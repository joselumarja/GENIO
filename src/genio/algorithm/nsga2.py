"""NSGA-II search backed by pymoo and evaluated through GENIO sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
from pymoo import __version__ as pymoo_version
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.problems.static import StaticProblem

from genio.algorithm.base import SearchAlgorithm, SearchContext
from genio.checkpoint.errors import (
    CheckpointFormatError,
    CheckpointStateError,
)
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.result import ResultStatus
from genio.objective.base import ObjectiveSchema
from genio.objective.runtime import EvaluatedBatch
from genio.search_space.space import SearchSpace


class _GenotypeSampling(Sampling):
    """Sample complete categorical genotypes, optionally without replacement."""

    def __init__(
        self,
        search_space: SearchSpace,
        *,
        balanced: bool,
        unique: bool,
    ) -> None:
        super().__init__()
        self.search_space = search_space
        self.balanced = balanced
        self.unique = unique

    def _do(
        self,
        problem: Problem,
        n_samples: int,
        *,
        random_state,
        **kwargs,
    ) -> np.ndarray:
        """Generate an exact-size matrix of valid categorical genotypes."""
        del problem, kwargs
        if self.unique and n_samples > self.search_space.search_space_size:
            raise ValueError(
                f"Cannot sample {n_samples} unique NSGA-II genotypes from a "
                f"search space of size {self.search_space.search_space_size}."
            )

        genotypes: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        max_attempts = max(100, n_samples * 20)
        while len(genotypes) < n_samples and attempts < max_attempts:
            genotype = self._sample_genotype(random_state)
            attempts += 1
            if self.unique and genotype in seen:
                continue
            seen.add(genotype)
            genotypes.append(genotype)

        if self.unique and len(genotypes) < n_samples:
            start = int(
                random_state.integers(0, self.search_space.search_space_size)
            )
            for offset in range(n_samples):
                genotype = self.search_space.index_to_genotype(
                    (start + offset) % self.search_space.search_space_size
                )
                if genotype in seen:
                    continue
                seen.add(genotype)
                genotypes.append(genotype)
                if len(genotypes) == n_samples:
                    break

        if len(genotypes) != n_samples:
            raise RuntimeError(
                f"NSGA-II initialization produced {len(genotypes)} of "
                f"{n_samples} requested genotypes."
            )
        return np.asarray(genotypes, dtype=int).reshape(
            n_samples,
            len(self.search_space.genotype_lengths),
        )

    def _sample_genotype(self, random_state) -> tuple[int, ...]:
        genes: list[int] = []
        slot_count = len(self.search_space.slot_lengths)
        for column, length in enumerate(self.search_space.genotype_lengths):
            if self.balanced and column < slot_count:
                slot = self.search_space.scenario.slots[column]
                group = slot.stage_groups[
                    int(random_state.integers(0, len(slot.stage_groups)))
                ]
                genes.append(group[int(random_state.integers(0, len(group)))])
            else:
                genes.append(int(random_state.integers(0, length)))
        return tuple(genes)


class _BalancedGenotypeSampling(_GenotypeSampling):
    """Sample stage groups uniformly before their concrete alternatives.

    Scenario slots use two-level sampling so stages with many parameterized
    alternatives do not receive more probability solely because their group is
    larger. Design genes remain uniform over their categorical domains.
    """

    def __init__(self, search_space: SearchSpace, *, unique: bool = False) -> None:
        """Initialize the pymoo sampler for a specific search space.

        Args:
            search_space: Source of stage groups and genotype domain lengths.
        """
        super().__init__(search_space, balanced=True, unique=unique)


class _CategoricalMutation(Mutation):
    """Replace selected bounded integer genes with another valid category.

    Unlike numeric perturbation, a selected gene with more than one category
    is guaranteed to change while remaining inside its inclusive bounds.
    Single-category genes cannot change.
    """

    def _do(
        self,
        problem: Problem,
        values: np.ndarray,
        *,
        random_state,
        **kwargs,
    ) -> np.ndarray:
        """Apply pymoo's per-variable mutation mask to categorical genes.

        Args:
            problem: Pymoo problem providing mutation probabilities and bounds.
            values: Population genotype matrix to mutate.
            random_state: Pymoo-managed NumPy random generator.
            **kwargs: Additional pymoo mutation arguments; ignored.

        Returns:
            A mutated integer copy; the input array is not modified.
        """
        del kwargs
        mutated = np.asarray(values, dtype=int).copy()
        probability = self.get_prob_var(problem, size=(len(mutated), 1))
        selected = random_state.random(mutated.shape) < probability
        lower = np.asarray(problem.xl, dtype=int)
        upper = np.asarray(problem.xu, dtype=int)

        for column, (minimum, maximum) in enumerate(zip(lower, upper, strict=True)):
            domain_size = int(maximum - minimum + 1)
            if domain_size <= 1:
                continue
            rows = np.flatnonzero(selected[:, column])
            if len(rows) == 0:
                continue
            offsets = random_state.integers(1, domain_size, size=len(rows))
            current = mutated[rows, column] - minimum
            mutated[rows, column] = minimum + (current + offsets) % domain_size
        return mutated


class NSGA2Search(SearchAlgorithm):
    """Adapt pymoo NSGA-II to GENIO's external generational ask/tell loop.

    Pymoo proposes integer genotype populations, while GENIO materializes and
    evaluates the corresponding individuals outside pymoo. Each ``ask`` must
    therefore be followed by one ``tell`` containing the complete matching
    population. Successful objective values are returned to pymoo using its
    minimization convention, so maximization objectives are negated. Failed
    evaluations are represented as infeasible and their objective metrics are
    not read.

    Initial stage-balanced sampling chooses a stage group uniformly and then
    an alternative inside it. Uniform crossover and categorical mutation are
    delegated to pymoo, as is optional duplicate elimination. The search space
    is bound on first use and the same ``SearchSpace`` object must be retained.

    Checkpoints store completed evaluations rather than pymoo internals.
    Restoration resets pymoo and deterministically replays every completed
    generation, rejecting state if replayed sizes or genotypes differ.
    """

    supports_checkpointing = True

    def __init__(
        self,
        *,
        population_size: int = 80,
        max_generations: int = 20,
        crossover_probability: float = 0.9,
        mutation_probability: float | None = None,
        eliminate_duplicates: bool = True,
        balanced_initialization: bool = True,
        initial_population: Sequence[Sequence[int]] | None = None,
        seed: int = 0,
    ) -> None:
        """Configure a multi-objective NSGA-II search.

        Args:
            population_size: Positive target population size. When duplicate
                elimination is enabled, it must not exceed the finite search
                space size checked on first ``ask``.
            max_generations: Maximum number of completed generations. Zero
                creates an already-stopped search.
            crossover_probability: Probability passed to pymoo's uniform
                crossover operator.
            mutation_probability: Per-variable probability passed to pymoo's
                mutation operator. ``None`` leaves pymoo's default in effect.
            eliminate_duplicates: Whether pymoo should eliminate duplicate
                genotypes. Duplicate supplied initial genotypes are rejected
                on first ``ask`` when enabled.
            balanced_initialization: Whether an unsupplied initial population
                uses stage-balanced sampling rather than integer-uniform
                sampling.
            initial_population: Optional integer genotypes used as pymoo's
                initial sample. Exactly ``population_size`` entries are
                required; bounds are validated against the search space on
                first ``ask``.
            seed: Non-negative seed supplied to pymoo setup and checkpoint
                replay.

        Raises:
            ValueError: If scalar or boolean configuration is invalid, initial
                genes are not integers, or the initial population has the
                wrong size.
        """
        self._validate_configuration(
            population_size=population_size,
            max_generations=max_generations,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            eliminate_duplicates=eliminate_duplicates,
            balanced_initialization=balanced_initialization,
            seed=seed,
        )
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = float(crossover_probability)
        self.mutation_probability = (
            float(mutation_probability)
            if mutation_probability is not None
            else None
        )
        self.eliminate_duplicates = eliminate_duplicates
        self.balanced_initialization = balanced_initialization
        self.initial_population = self._normalize_initial_population(
            initial_population
        )
        if (
            self.initial_population is not None
            and len(self.initial_population) != population_size
        ):
            raise ValueError(
                "initial_population must contain exactly population_size genotypes."
            )
        self.seed = seed

        self._objective_schema: ObjectiveSchema | None = None
        self._search_space: SearchSpace | None = None
        self._problem: Problem | None = None
        self._algorithm: NSGA2 | None = None
        self._pending_population: Any | None = None
        self._pending_individuals: tuple[Individual, ...] | None = None
        self._evaluations: list[Evaluation] = []
        self._evaluation_by_id: dict[str, Evaluation] = {}
        self._valid_evaluation_ids: set[str] = set()
        self._generation_sizes: list[int] = []
        self._completed_generations = 0
        self._exhausted = False

    def configure(self, context: SearchContext) -> None:
        """Bind the search space and multi-objective schema for this run."""

        if not isinstance(context, SearchContext):
            raise TypeError("context must be a SearchContext.")
        schema = context.objective_schema
        if schema is None:
            raise ValueError("NSGA2Search requires an objective schema.")
        if len(schema) < 2:
            raise ValueError("NSGA2Search requires at least two objectives.")
        super().configure(context)
        self._objective_schema = schema

    def ask(self) -> Sequence[Individual]:
        """Materialize the next integer-genotype population proposed by pymoo.

        Returns:
            Proposed individuals in pymoo population order, or an empty tuple
            after the generation budget or when pymoo yields no population.

        Raises:
            RuntimeError: If the previous population still awaits ``tell`` or
                the algorithm is reused with another ``SearchSpace`` object.
            ValueError: If genotype domains are empty, the population exceeds
                the finite space, an initial genotype is invalid, or duplicate
                initial genotypes conflict with duplicate elimination.

        Note:
            Generated metadata records the NSGA-II name, one-based generation,
            population position, and whether the proposal is initialization or
            offspring.
        """

        if self._pending_population is not None:
            raise RuntimeError("tell() is required before asking for another generation.")
        if self.should_stop():
            return ()

        search_space = self._bind_search_space(self.context.search_space)
        assert self._algorithm is not None
        population = self._algorithm.ask()
        if population is None or len(population) == 0:
            self._exhausted = True
            return ()

        generation = self._completed_generations + 1
        genotypes = self._population_genotypes(
            population,
            search_space,
            expected_size=self.population_size,
        )
        individuals = tuple(
            search_space.from_genotype(
                genotype,
                metadata={
                    "algorithm": {
                        "name": "nsga2",
                        "generation": generation,
                        "population_index": index,
                        "proposal_origin": (
                            "initialization" if generation == 1 else "offspring"
                        ),
                    }
                },
            )
            for index, genotype in enumerate(genotypes)
        )
        population.set("genio_id", [individual.id for individual in individuals])
        self._pending_population = population
        self._pending_individuals = individuals
        return individuals

    def tell(self, batch: EvaluatedBatch) -> None:
        """Validate and return one externally evaluated population to pymoo.

        Input evaluations and their precomputed objective rows may be unordered
        and are restored to proposal order by individual ID. Failed evaluations
        and invalid objective rows receive a positive constraint violation.

        Args:
            batch: Objective-aware result for every individual returned by the
                latest ``ask``.

        Raises:
            RuntimeError: If no pymoo population is awaiting evaluation.
            ValueError: If counts, identifiers, individuals, objective names,
                or objective rows do not match the pending population/schema.
        """

        population = self._pending_population
        pending = self._pending_individuals
        if population is None or pending is None:
            raise RuntimeError("tell() requires a pending generation from ask().")

        ordered = self._validate_and_order_evaluations(batch.evaluations, pending)
        objective_values, valid_mask = self._order_objective_data(batch, ordered)
        self._evaluate_population(population, objective_values, valid_mask)
        assert self._algorithm is not None
        self._algorithm.tell(infills=population)

        self._evaluations.extend(ordered)
        self._evaluation_by_id.update(
            (evaluation.individual.id, evaluation) for evaluation in ordered
        )
        self._valid_evaluation_ids.update(
            evaluation.individual.id
            for evaluation, valid in zip(ordered, valid_mask, strict=True)
            if valid
        )
        self._generation_sizes.append(len(ordered))
        self._completed_generations += 1
        self._pending_population = None
        self._pending_individuals = None

    def should_stop(self) -> bool:
        """Report whether the generation budget or pymoo proposals ended.

        Returns:
            ``True`` after ``max_generations`` populations have been completed
            through ``tell``, or after pymoo returns no population from
            ``ask``.
        """

        return self._exhausted or self._completed_generations >= self.max_generations

    def best_individuals(self) -> Sequence[Individual]:
        """Return successful individuals in pymoo's current optimal set.

        Returns:
            Individuals referenced by pymoo's optimal population, preserving
            its order. Missing or failed evaluations are excluded, as are
            repeated non-null search indexes after their first occurrence. An
            empty tuple is returned before pymoo has an optimal population.
        """

        if self._algorithm is None or self._algorithm.opt is None:
            return ()
        identifiers = self._algorithm.opt.get("genio_id")
        best: list[Individual] = []
        seen: set[int] = set()
        for identifier in identifiers:
            evaluation = self._evaluation_by_id.get(str(identifier))
            if (
                evaluation is None
                or evaluation.individual.id not in self._valid_evaluation_ids
                or evaluation.result.status is not ResultStatus.SUCCESS
            ):
                continue
            search_index = evaluation.individual.search_index
            if search_index is not None and search_index in seen:
                continue
            if search_index is not None:
                seen.add(search_index)
            best.append(evaluation.individual)
        return tuple(best)

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return immutable technical NSGA-II and pymoo configuration.

        Returns:
            Algorithm settings, initial genotypes, seed, installed pymoo
            version. Objective configuration belongs to the search context.
        """

        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "crossover_probability": self.crossover_probability,
            "mutation_probability": self.mutation_probability,
            "eliminate_duplicates": self.eliminate_duplicates,
            "balanced_initialization": self.balanced_initialization,
            "initial_population": (
                [list(genotype) for genotype in self.initial_population]
                if self.initial_population is not None
                else None
            ),
            "seed": self.seed,
            "pymoo_version": pymoo_version,
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Serialize only native state not owned by the session history.

        Returns:
            The pymoo exhaustion flag. Evaluations and generation boundaries
            are supplied by the session during restoration.

        Raises:
            CheckpointStateError: If an asked population still awaits
                ``tell``. Only completed-generation boundaries are safe.
        """

        if self._pending_population is not None:
            raise CheckpointStateError(
                "NSGA2Search cannot checkpoint a generation awaiting tell()."
            )
        return {
            "exhausted": self._exhausted,
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        search_space: SearchSpace,
        evaluated_batches: Sequence[EvaluatedBatch] = (),
    ) -> None:
        """Rebuild deterministic pymoo state by replaying completed generations.

        Runtime state is reset, pymoo is initialized with the configured seed,
        and each recorded generation is requested and told again. Replay must
        reproduce both the recorded generation size and exact genotype order.

        Args:
            state: Native exhaustion state.
            search_space: Search space used to initialize pymoo and verify
                replayed genotypes.
            evaluated_batches: Authoritative generations restored by the session.

        Raises:
            RuntimeError: If the algorithm has not been configured first.
            CheckpointFormatError: If state structure or counts are invalid, or
                replay produces a different population size or genotype
                sequence. A changed pymoo version is one possible cause of a
                replay mismatch.
            ValueError: If the supplied search space is incompatible with the
                configured population or initial genotypes.
        """

        if self._objective_schema is None:
            raise RuntimeError("NSGA2Search has not been configured for restoration.")
        if set(state) != {"exhausted"}:
            raise CheckpointFormatError("Invalid NSGA2Search checkpoint fields.")
        try:
            exhausted = state["exhausted"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid NSGA2Search checkpoint state.") from exc
        evaluations = [
            evaluation
            for batch in evaluated_batches
            for evaluation in batch.evaluations
        ]
        generation_sizes = [len(batch) for batch in evaluated_batches]
        completed_generations = len(evaluated_batches)
        if (
            completed_generations < 0
            or completed_generations > self.max_generations
            or len(generation_sizes) != completed_generations
            or any(size != self.population_size for size in generation_sizes)
            or sum(generation_sizes) != len(evaluations)
            or not isinstance(exhausted, bool)
        ):
            raise CheckpointFormatError("NSGA2Search checkpoint history is inconsistent.")
        if any(
            batch.objective_names != self._objective_schema.names
            for batch in evaluated_batches
        ):
            raise CheckpointFormatError(
                "NSGA2Search objective batches do not match its configured schema."
            )
        self._reset_runtime_state()
        self._bind_search_space(search_space)
        offset = 0
        for batch_index, generation_size in enumerate(generation_sizes):
            assert self._algorithm is not None
            population = self._algorithm.ask()
            generation = tuple(evaluations[offset : offset + generation_size])
            if population is None or len(population) != generation_size:
                raise CheckpointFormatError(
                    "NSGA2Search replay produced a different generation size."
                )
            expected = tuple(
                search_space.to_genotype(evaluation.individual)
                for evaluation in generation
            )
            actual = self._population_genotypes(
                population,
                search_space,
                expected_size=self.population_size,
            )
            if actual != expected:
                raise CheckpointFormatError(
                    "NSGA2Search replay produced different genotypes; check pymoo version."
                )
            population.set(
                "genio_id",
                [evaluation.individual.id for evaluation in generation],
            )
            batch = evaluated_batches[batch_index]
            ordered = self._validate_and_order_evaluations(
                batch.evaluations,
                tuple(evaluation.individual for evaluation in generation),
            )
            objective_values, valid_mask = self._order_objective_data(batch, ordered)
            self._evaluate_population(population, objective_values, valid_mask)
            self._algorithm.tell(infills=population)
            self._evaluations.extend(generation)
            self._evaluation_by_id.update(
                (evaluation.individual.id, evaluation) for evaluation in generation
            )
            self._valid_evaluation_ids.update(
                evaluation.individual.id
                for evaluation, valid in zip(generation, valid_mask, strict=True)
                if valid
            )
            self._generation_sizes.append(len(generation))
            self._completed_generations += 1
            offset += generation_size
        self._exhausted = exhausted

    def _bind_search_space(self, search_space: SearchSpace) -> SearchSpace:
        """Validate and bind the sole search space used by this instance."""

        if search_space is not self.context.search_space:
            raise RuntimeError(
                "NSGA2Search cannot use a search space outside its configured context."
            )
        if self._search_space is None:
            if not search_space.genotype_lengths or any(
                length <= 0 for length in search_space.genotype_lengths
            ):
                raise ValueError("NSGA2Search requires non-empty genotype domains.")
            if (
                self.eliminate_duplicates
                and self.population_size > search_space.search_space_size
            ):
                raise ValueError(
                    "population_size cannot exceed the finite search-space size "
                    "when duplicate elimination is enabled."
                )
            if self.initial_population is not None:
                for genotype in self.initial_population:
                    search_space.genotype_to_index(genotype)
                if self.eliminate_duplicates and len(set(self.initial_population)) != len(
                    self.initial_population
                ):
                    raise ValueError(
                        "initial_population contains duplicates while duplicate "
                        "elimination is enabled."
                    )
            self._search_space = search_space
            self._initialize_pymoo(search_space)
        elif self._search_space is not search_space:
            raise RuntimeError("NSGA2Search cannot be reused with another SearchSpace.")
        return search_space

    def _initialize_pymoo(self, search_space: SearchSpace) -> None:
        """Create the bounded integer problem and configured NSGA-II instance.

        The problem has one inequality constraint used to separate successful
        and failed external evaluations. ``NoTermination`` leaves stopping to
        this adapter's generation counter.
        """

        lower = np.zeros(len(search_space.genotype_lengths), dtype=int)
        upper = np.asarray(search_space.genotype_lengths, dtype=int) - 1
        assert self._objective_schema is not None
        self._problem = Problem(
            n_var=len(search_space.genotype_lengths),
            n_obj=len(self._objective_schema),
            n_ieq_constr=1,
            xl=lower,
            xu=upper,
            vtype=int,
        )
        sampling: Sampling | np.ndarray
        if self.initial_population is not None:
            sampling = np.asarray(self.initial_population, dtype=int)
        elif self.balanced_initialization:
            sampling = _BalancedGenotypeSampling(
                search_space,
                unique=self.eliminate_duplicates,
            )
        else:
            sampling = _GenotypeSampling(
                search_space,
                balanced=False,
                unique=self.eliminate_duplicates,
            )

        crossover: Crossover = UniformCrossover(
            prob=self.crossover_probability
        )
        mutation: Mutation = _CategoricalMutation(
            prob=1.0,
            prob_var=self.mutation_probability,
        )
        self._algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=self.eliminate_duplicates,
        )
        self._algorithm.setup(
            self._problem,
            termination=NoTermination(),
            seed=self.seed,
            verbose=False,
        )

    def _evaluate_population(
        self,
        population: Any,
        objective_values: Sequence[Sequence[float]],
        valid_mask: Sequence[bool],
    ) -> None:
        """Attach precomputed objectives and validity constraints to a population.

        Valid rows receive a negative constraint value. Failed evaluations and
        invalid objective rows retain a positive artificial constraint and the
        placeholders supplied by ``EvaluatedBatch.minimization_matrix``.
        """

        assert self._problem is not None
        assert self._objective_schema is not None
        values = np.asarray(objective_values, dtype=float)
        expected_shape = (len(valid_mask), len(self._objective_schema))
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "EvaluatedBatch minimization matrix must contain an exact, finite "
                f"{expected_shape} objective matrix."
            )
        validity = np.asarray(valid_mask)
        if validity.shape != (len(valid_mask),) or validity.dtype != np.bool_:
            raise ValueError("EvaluatedBatch valid mask must contain only booleans.")
        constraint_values = np.where(validity[:, None], -1.0, 1.0)
        assert self._algorithm is not None
        self._algorithm.evaluator.eval(
            StaticProblem(
                self._problem,
                F=values,
                G=constraint_values,
            ),
            population,
        )

    @staticmethod
    def _population_genotypes(
        population: Any,
        search_space: SearchSpace,
        *,
        expected_size: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Convert and validate pymoo's population matrix as GENIO genotypes."""

        if len(population) != expected_size:
            raise RuntimeError(
                f"NSGA-II expected exactly {expected_size} candidates, "
                f"but pymoo produced {len(population)}."
            )
        raw_values = population.get("X")
        if raw_values is None:
            raise RuntimeError("NSGA-II population has no genotype matrix X.")
        values = np.asarray(raw_values)
        expected_shape = (expected_size, len(search_space.genotype_lengths))
        if values.shape != expected_shape:
            raise RuntimeError(
                f"NSGA-II population X must have shape {expected_shape}, "
                f"got {values.shape}."
            )
        if not (
            np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.floating)
        ):
            raise RuntimeError("NSGA-II population X must contain numeric genes.")
        numeric = values.astype(float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise RuntimeError("NSGA-II population X must contain finite integer genes.")
        genotypes: list[tuple[int, ...]] = []
        for row in numeric:
            genotype = tuple(int(value) for value in row)
            search_space.genotype_to_index(genotype)
            genotypes.append(genotype)
        return tuple(genotypes)

    def _order_objective_data(
        self,
        batch: EvaluatedBatch,
        ordered: Sequence[Evaluation],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
        """Restore precomputed objective rows and validity flags by evaluation ID."""

        assert self._objective_schema is not None
        if tuple(batch.objective_names) != self._objective_schema.names:
            raise ValueError(
                "EvaluatedBatch objective names do not match the configured schema."
            )
        matrix = batch.minimization_matrix_with_placeholder(invalid_value=0.0)
        if len(matrix) != len(batch.items):
            raise ValueError("EvaluatedBatch objective rows do not match its items.")
        invalid_row = (0.0,) * len(self._objective_schema)

        values_by_id: dict[str, tuple[float, ...]] = {}
        valid_by_id: dict[str, bool] = {}
        for item, row in zip(batch.items, matrix, strict=True):
            identifier = item.evaluation.individual.id
            if item.individual != item.evaluation.individual:
                raise ValueError(
                    f"EvaluatedBatch item {identifier!r} does not match its evaluation."
                )
            if identifier in values_by_id:
                raise ValueError(
                    f"Duplicate evaluation individual ID: {identifier!r}."
                )
            values_by_id[identifier] = tuple(row) if item.valid else invalid_row
            valid_by_id[identifier] = item.valid

        return (
            tuple(values_by_id[evaluation.individual.id] for evaluation in ordered),
            tuple(valid_by_id[evaluation.individual.id] for evaluation in ordered),
        )

    @staticmethod
    def _validate_and_order_evaluations(
        evaluations: Sequence[Evaluation],
        pending: Sequence[Individual],
    ) -> tuple[Evaluation, ...]:
        """Validate a complete population and restore proposal order."""

        if len(evaluations) != len(pending):
            raise ValueError(
                f"Expected {len(pending)} evaluations, got {len(evaluations)}."
            )
        by_id: dict[str, Evaluation] = {}
        for evaluation in evaluations:
            if evaluation.individual.id in by_id:
                raise ValueError(
                    f"Duplicate evaluation individual ID: {evaluation.individual.id!r}."
                )
            by_id[evaluation.individual.id] = evaluation
        expected_ids = {individual.id for individual in pending}
        if set(by_id) != expected_ids:
            raise ValueError("Evaluation IDs do not match the pending NSGA-II generation.")
        ordered = tuple(by_id[individual.id] for individual in pending)
        for individual, evaluation in zip(pending, ordered, strict=True):
            if evaluation.individual != individual:
                raise ValueError(
                    f"Evaluation individual {individual.id!r} does not match its proposal."
                )
            if evaluation.result.individual_id != individual.id:
                raise ValueError(
                    f"Result individual ID {evaluation.result.individual_id!r} does not "
                    "match its proposal."
                )
        return ordered

    def _reset_runtime_state(self) -> None:
        """Discard initialized pymoo and all accumulated runtime history."""

        self._search_space = None
        self._problem = None
        self._algorithm = None
        self._pending_population = None
        self._pending_individuals = None
        self._evaluations = []
        self._evaluation_by_id = {}
        self._valid_evaluation_ids = set()
        self._generation_sizes = []
        self._completed_generations = 0
        self._exhausted = False

    @staticmethod
    def _validate_configuration(
        *,
        population_size: int,
        max_generations: int,
        crossover_probability: float,
        mutation_probability: float | None,
        eliminate_duplicates: bool,
        balanced_initialization: bool,
        seed: int,
    ) -> None:
        """Validate population, generation, probability, flag, and seed values."""

        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size <= 0
        ):
            raise ValueError("population_size must be a positive integer.")
        if (
            isinstance(max_generations, bool)
            or not isinstance(max_generations, int)
            or max_generations < 0
        ):
            raise ValueError("max_generations must be a non-negative integer.")
        for name, value, optional in (
            ("crossover_probability", crossover_probability, False),
            ("mutation_probability", mutation_probability, True),
        ):
            if value is None and optional:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be between 0 and 1.")
        if not isinstance(eliminate_duplicates, bool):
            raise ValueError("eliminate_duplicates must be a bool.")
        if not isinstance(balanced_initialization, bool):
            raise ValueError("balanced_initialization must be a bool.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")

    @staticmethod
    def _normalize_initial_population(
        initial_population: Sequence[Sequence[int]] | None,
    ) -> tuple[tuple[int, ...], ...] | None:
        """Copy initial genotypes into immutable tuples of plain integers."""

        if initial_population is None:
            return None
        normalized: list[tuple[int, ...]] = []
        for genotype in initial_population:
            genes: list[int] = []
            for gene in genotype:
                if isinstance(gene, bool) or not isinstance(gene, Integral):
                    raise ValueError(
                        "initial_population genotypes must contain only integers."
                    )
                genes.append(int(gene))
            normalized.append(tuple(genes))
        return tuple(normalized)


__all__ = ["NSGA2Search"]
