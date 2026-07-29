from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Any, TYPE_CHECKING

import numpy as np
from pymoo import __version__ as pymoo_version
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.problems.static import StaticProblem

from genio.algorithm.base import SearchAlgorithm
from genio.checkpoint.codec import decode_evaluation, encode_evaluation
from genio.checkpoint.errors import (
    CheckpointFormatError,
    CheckpointNotSupportedError,
    CheckpointStateError,
)
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.result import ResultStatus
from genio.objective.base import Objective, ObjectiveSet, OptimizationDirection
from genio.search_space.space import SearchSpace

if TYPE_CHECKING:
    from genio.session.optimization import OptimizationSession


class _BalancedGenotypeSampling(Sampling):
    """Sample stage choices uniformly before sampling their parameter variants."""

    def __init__(self, search_space: SearchSpace) -> None:
        super().__init__()
        self.search_space = search_space

    def _do(
        self,
        problem: Problem,
        n_samples: int,
        *,
        random_state,
        **kwargs,
    ) -> np.ndarray:
        del problem, kwargs
        values = np.empty(
            (n_samples, len(self.search_space.genotype_lengths)),
            dtype=int,
        )
        slot_count = len(self.search_space.slot_lengths)
        for row in range(n_samples):
            for column, slot in enumerate(self.search_space.scenario.slots):
                group = slot.stage_groups[
                    int(random_state.integers(0, len(slot.stage_groups)))
                ]
                values[row, column] = group[
                    int(random_state.integers(0, len(group)))
                ]
            for offset, length in enumerate(self.search_space.design_lengths):
                values[row, slot_count + offset] = int(
                    random_state.integers(0, length)
                )
        return values


class _CategoricalMutation(Mutation):
    """Replace selected bounded integer genes with another valid category."""

    def _do(
        self,
        problem: Problem,
        values: np.ndarray,
        *,
        random_state,
        **kwargs,
    ) -> np.ndarray:
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
    """Adapt pymoo NSGA-II to GENIO's external generational ask/tell loop."""

    supports_checkpointing = True

    def __init__(
        self,
        *,
        objectives: Objective | ObjectiveSet,
        population_size: int = 80,
        max_generations: int = 20,
        crossover_probability: float = 0.9,
        mutation_probability: float | None = None,
        eliminate_duplicates: bool = True,
        balanced_initialization: bool = True,
        initial_population: Sequence[Sequence[int]] | None = None,
        seed: int = 0,
    ) -> None:
        self._validate_configuration(
            population_size=population_size,
            max_generations=max_generations,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            eliminate_duplicates=eliminate_duplicates,
            balanced_initialization=balanced_initialization,
            seed=seed,
        )
        self.objectives = (
            objectives
            if isinstance(objectives, ObjectiveSet)
            else ObjectiveSet((objectives,))
        )
        if len(self.objectives.objectives) < 2:
            raise ValueError("NSGA2Search requires at least two objectives.")

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

        self._search_space: SearchSpace | None = None
        self._problem: Problem | None = None
        self._algorithm: NSGA2 | None = None
        self._pending_population: Any | None = None
        self._pending_individuals: tuple[Individual, ...] | None = None
        self._evaluations: list[Evaluation] = []
        self._evaluation_by_id: dict[str, Evaluation] = {}
        self._generation_sizes: list[int] = []
        self._completed_generations = 0
        self._exhausted = False

    def ask(self, session: OptimizationSession) -> Sequence[Individual]:
        """Return the next integer-genotype population proposed by pymoo."""

        if self._pending_population is not None:
            raise RuntimeError("tell() is required before asking for another generation.")
        if self.should_stop():
            return ()

        search_space = self._bind_search_space(session.search_space)
        assert self._algorithm is not None
        population = self._algorithm.ask()
        if population is None or len(population) == 0:
            self._exhausted = True
            return ()

        generation = self._completed_generations + 1
        genotypes = self._population_genotypes(population, search_space)
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

    def tell(self, evaluations: Sequence[Evaluation]) -> None:
        """Return one complete externally evaluated population to pymoo."""

        population = self._pending_population
        pending = self._pending_individuals
        if population is None or pending is None:
            raise RuntimeError("tell() requires a pending generation from ask().")

        ordered = self._validate_and_order_evaluations(evaluations, pending)
        self._evaluate_population(population, ordered)
        assert self._algorithm is not None
        self._algorithm.tell(infills=population)

        self._evaluations.extend(ordered)
        self._evaluation_by_id.update(
            (evaluation.individual.id, evaluation) for evaluation in ordered
        )
        self._generation_sizes.append(len(ordered))
        self._completed_generations += 1
        self._pending_population = None
        self._pending_individuals = None

    def should_stop(self) -> bool:
        """Return whether the generation budget or finite space is exhausted."""

        return self._exhausted or self._completed_generations >= self.max_generations

    def best_individuals(self) -> Sequence[Individual]:
        """Return the successful rank-zero population maintained by NSGA-II."""

        if self._algorithm is None or self._algorithm.opt is None:
            return ()
        identifiers = self._algorithm.opt.get("genio_id")
        best: list[Individual] = []
        seen: set[int] = set()
        for identifier in identifiers:
            evaluation = self._evaluation_by_id.get(str(identifier))
            if (
                evaluation is None
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
        """Return immutable NSGA-II, objective and pymoo configuration."""

        unsupported = [
            objective.name
            for objective in self.objectives.objectives
            if type(objective).checkpoint_signature is Objective.checkpoint_signature
        ]
        if unsupported:
            raise CheckpointNotSupportedError(
                "Objectives must override checkpoint_signature(): "
                + ", ".join(unsupported)
            )
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
            "objectives": [
                dict(objective.checkpoint_signature())
                for objective in self.objectives.objectives
            ],
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return replayable completed-generation state."""

        if self._pending_population is not None:
            raise CheckpointStateError(
                "NSGA2Search cannot checkpoint a generation awaiting tell()."
            )
        return {
            "evaluations": [
                encode_evaluation(evaluation) for evaluation in self._evaluations
            ],
            "generation_sizes": list(self._generation_sizes),
            "completed_generations": self._completed_generations,
            "exhausted": self._exhausted,
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
        search_space: SearchSpace,
    ) -> None:
        """Rebuild deterministic pymoo state by replaying completed generations."""

        if version != self.checkpoint_version:
            raise CheckpointFormatError(
                f"Unsupported NSGA2Search checkpoint version {version}."
            )
        try:
            evaluations = [
                decode_evaluation(value, search_space)
                for value in state.get("evaluations", [])
            ]
            generation_sizes = [
                int(value) for value in state.get("generation_sizes", [])
            ]
            completed_generations = int(state["completed_generations"])
            exhausted = state["exhausted"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid NSGA2Search checkpoint state.") from exc
        if (
            completed_generations < 0
            or completed_generations > self.max_generations
            or len(generation_sizes) != completed_generations
            or any(size <= 0 for size in generation_sizes)
            or sum(generation_sizes) != len(evaluations)
            or not isinstance(exhausted, bool)
        ):
            raise CheckpointFormatError("NSGA2Search checkpoint history is inconsistent.")

        self._reset_runtime_state()
        self._bind_search_space(search_space)
        offset = 0
        for generation_size in generation_sizes:
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
            actual = self._population_genotypes(population, search_space)
            if actual != expected:
                raise CheckpointFormatError(
                    "NSGA2Search replay produced different genotypes; check pymoo version."
                )
            population.set(
                "genio_id",
                [evaluation.individual.id for evaluation in generation],
            )
            self._evaluate_population(population, generation)
            self._algorithm.tell(infills=population)
            self._evaluations.extend(generation)
            self._evaluation_by_id.update(
                (evaluation.individual.id, evaluation) for evaluation in generation
            )
            self._generation_sizes.append(generation_size)
            self._completed_generations += 1
            offset += generation_size
        self._exhausted = exhausted

    def _bind_search_space(self, search_space: SearchSpace) -> SearchSpace:
        if self._search_space is None:
            if not search_space.genotype_lengths or any(
                length <= 0 for length in search_space.genotype_lengths
            ):
                raise ValueError("NSGA2Search requires non-empty genotype domains.")
            if self.population_size > search_space.search_space_size:
                raise ValueError(
                    "population_size cannot exceed the finite search-space size."
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
        lower = np.zeros(len(search_space.genotype_lengths), dtype=int)
        upper = np.asarray(search_space.genotype_lengths, dtype=int) - 1
        self._problem = Problem(
            n_var=len(search_space.genotype_lengths),
            n_obj=len(self.objectives.objectives),
            n_ieq_constr=1,
            xl=lower,
            xu=upper,
            vtype=int,
        )
        sampling: Sampling | np.ndarray
        if self.initial_population is not None:
            sampling = np.asarray(self.initial_population, dtype=int)
        elif self.balanced_initialization:
            sampling = _BalancedGenotypeSampling(search_space)
        else:
            sampling = IntegerRandomSampling()

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
        evaluations: Sequence[Evaluation],
    ) -> None:
        assert self._problem is not None
        objective_values = np.zeros(
            (len(evaluations), len(self.objectives.objectives)),
            dtype=float,
        )
        constraint_values = np.ones((len(evaluations), 1), dtype=float)
        for row, evaluation in enumerate(evaluations):
            if evaluation.result.status is not ResultStatus.SUCCESS:
                continue
            constraint_values[row, 0] = -1.0
            for column, objective in enumerate(self.objectives.objectives):
                value = objective.value(evaluation)
                if not isfinite(value):
                    raise ValueError(
                        f"Objective {objective.name!r} produced a non-finite value."
                    )
                objective_values[row, column] = (
                    -value
                    if objective.direction is OptimizationDirection.MAXIMIZE
                    else value
                )
        assert self._algorithm is not None
        self._algorithm.evaluator.eval(
            StaticProblem(
                self._problem,
                F=objective_values,
                G=constraint_values,
            ),
            population,
        )

    @staticmethod
    def _population_genotypes(
        population: Any,
        search_space: SearchSpace,
    ) -> tuple[tuple[int, ...], ...]:
        values = np.asarray(population.get("X"))
        genotypes: list[tuple[int, ...]] = []
        for row in values:
            genotype = tuple(int(value) for value in row)
            search_space.genotype_to_index(genotype)
            genotypes.append(genotype)
        return tuple(genotypes)

    @staticmethod
    def _validate_and_order_evaluations(
        evaluations: Sequence[Evaluation],
        pending: Sequence[Individual],
    ) -> tuple[Evaluation, ...]:
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
        self._search_space = None
        self._problem = None
        self._algorithm = None
        self._pending_population = None
        self._pending_individuals = None
        self._evaluations = []
        self._evaluation_by_id = {}
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
