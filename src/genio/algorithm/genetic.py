"""Generational genetic search over GENIO's categorical genotypes.

The implementation preserves the selection, crossover, and mutation behavior
of GENIO's legacy genetic search while exposing it through the session's
external ask/tell evaluation protocol.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from random import Random
from statistics import median

from genio.algorithm.base import SearchAlgorithm, SearchContext
from genio.checkpoint.codec import (
    decode_random_state,
    encode_random_state,
)
from genio.checkpoint.errors import (
    CheckpointFormatError,
    CheckpointStateError,
)
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.objective.runtime import EvaluatedBatch
from genio.search_space.space import SearchSpace

@dataclass(frozen=True, slots=True)
class _Candidate:
    """Unmaterialized genotype and the provenance attached to its individual.

    Attributes:
        genotype: Categorical gene indexes understood by the search space.
        origin: Proposal source recorded in algorithm metadata.
        parent_ids: Identifiers of the two selected parents, when applicable.
        mutation_applied: Whether the candidate was selected for mutation.
        mutation_changed: Whether resampling actually changed its genotype.
    """

    genotype: tuple[int, ...]
    origin: str
    parent_ids: tuple[str, ...] = ()
    mutation_applied: bool = False
    mutation_changed: bool = False


class GeneticSearch(SearchAlgorithm):
    """Run the legacy generational GA through a strict ask/tell contract.

    Each ``ask`` returns one complete population and must be followed by a
    ``tell`` containing exactly one matching evaluation per proposal. The first
    population is supplied by ``initial_population`` or sampled from the search
    space. Later populations fully replace their parents: successful members
    of the previous generation are selected by a median-filtered roulette,
    crossed uniformly, and optionally mutated. If the complete previous
    generation failed, a fresh population is sampled instead.

    Selection fitness comes from the aggregate scores prepared by the session's
    objective runtime. Failed evaluations and evaluations with invalid
    objective values receive zero fitness. Generation and global bests use only
    scores already committed through ``tell``.

    The search space is bound on first use and the same ``SearchSpace`` object
    must be used for the lifetime of the algorithm.
    """

    supports_checkpointing = True

    def __init__(
        self,
        *,
        population_size: int = 80,
        mutation_probability: float = 0.05,
        max_generations: int = 20,
        start_generation: int = 1,
        balanced_initialization: bool = True,
        initial_population: Sequence[Sequence[int]] | None = None,
        random: Random | None = None,
    ) -> None:
        """Configure the generational genetic search.

        Args:
            population_size: Positive even number of individuals in every
                generation.
            mutation_probability: Probability that each offspring candidate
                has one randomly selected gene resampled after crossover.
            max_generations: Inclusive highest generation number that may be
                proposed. Zero disables all proposals with the default start.
            start_generation: Label of the first generation to propose.
            balanced_initialization: Whether sampled initial and restart
                populations choose stage groups uniformly before alternatives.
            initial_population: Optional sequence containing exactly
                ``population_size`` genotypes. It is required when
                ``start_generation`` is not one and is validated against the
                search space on first ``ask``.
            random: Pseudo-random generator used by sampling, selection,
                crossover, and mutation. A new unseeded generator is created
                when omitted.

        Raises:
            ValueError: If scalar configuration or initial population size is
                invalid, or if a non-default starting generation lacks an
                initial population.
        """
        self._validate_configuration(
            population_size=population_size,
            mutation_probability=mutation_probability,
            max_generations=max_generations,
            start_generation=start_generation,
            balanced_initialization=balanced_initialization,
        )
        self.population_size = population_size
        self.mutation_probability = float(mutation_probability)
        self.max_generations = max_generations
        self.start_generation = start_generation
        self.balanced_initialization = balanced_initialization
        self.initial_population = (
            tuple(tuple(genotype) for genotype in initial_population)
            if initial_population is not None
            else None
        )
        if (
            self.initial_population is not None
            and len(self.initial_population) != population_size
        ):
            raise ValueError(
                "initial_population must contain exactly population_size genotypes."
            )
        if self.initial_population is not None and any(
            isinstance(gene, bool) or not isinstance(gene, int)
            for genotype in self.initial_population
            for gene in genotype
        ):
            raise ValueError("initial_population genotypes must contain only integers.")
        if start_generation != 1 and self.initial_population is None:
            raise ValueError(
                "initial_population is required when start_generation is not 1."
            )

        self.random = random or Random()
        self._search_space: SearchSpace | None = None
        self._pending_generation: tuple[Individual, ...] | None = None
        self._last_evaluations: tuple[Evaluation, ...] = ()
        self._last_fitnesses: tuple[float, ...] = ()
        self._last_valid_ids: tuple[str, ...] = ()
        self._evaluations: list[Evaluation] = []
        self._generation_bests: list[Individual] = []
        self._generation_fitnesses: list[dict[str, float]] = []
        self._global_best: Individual | None = None
        self._global_best_fitness: float | None = None
        self._asked_generations = 0
        self._next_generation = start_generation

    def configure(self, context: SearchContext) -> None:
        """Attach objective configuration required for scalar fitness."""

        if context.objective_schema is None:
            raise ValueError("GeneticSearch requires an objective schema.")
        if not context.has_scalarizer:
            raise ValueError("GeneticSearch requires an objective scalarizer.")
        generation_count = max(0, self.max_generations - self.start_generation + 1)
        if generation_count > 1 and context.normalization_scope in {
            "batch",
            "cumulative",
        }:
            raise ValueError(
                f"{context.normalization_scope.upper()} normalization cannot produce "
                "fitness values comparable across multiple GeneticSearch generations; "
                "use fixed normalization or disable normalization."
            )
        super().configure(context)

    def ask(self) -> Sequence[Individual]:
        """Propose one complete initial, offspring, or restart generation.

        Args:
        Returns:
            A tuple of exactly ``population_size`` individuals, or an empty
            tuple when the inclusive generation limit has been passed.

        Raises:
            RuntimeError: If the previous generation still awaits ``tell`` or
                the algorithm is reused with another ``SearchSpace`` object.
            ValueError: If the search space has an empty genotype domain or a
                supplied initial genotype is invalid for that space.

        Note:
            Individual metadata records the generation, population position,
            proposal origin, parent identifiers, and mutation outcome.
        """

        if self._pending_generation is not None:
            raise RuntimeError("tell() is required before asking for another generation.")
        if self.should_stop():
            return ()

        search_space = self._bind_search_space(self.context.search_space)
        generation = self._next_generation
        if self._asked_generations == 0:
            candidates = self._initial_candidates(search_space)
        elif self._last_valid_ids:
            candidates = self._breed_generation(search_space)
        else:
            candidates = self._sample_candidates(search_space, origin="restart")

        population = self._materialize_population(
            search_space,
            candidates,
            generation=generation,
        )
        self._pending_generation = population
        self._asked_generations += 1
        self._next_generation += 1
        return population

    def tell(self, batch: EvaluatedBatch) -> None:
        """Validate, score, and commit one complete evaluated generation.

        Evaluations may arrive in any order; they are reordered to match the
        pending population before their precomputed aggregate scores are
        committed. Validation failures leave the generation pending.

        Args:
            batch: Objective-aware results for every individual returned by the
                latest ``ask``.

        Raises:
            RuntimeError: If no generation is awaiting evaluation.
            ValueError: If counts or identifiers do not match the pending
                population, an evaluation differs from its proposal, or an
                aggregate score is missing or non-finite.
        """

        pending = self._pending_generation
        if pending is None:
            raise RuntimeError("tell() requires a pending generation from ask().")

        assert self.context.objective_schema is not None
        expected_names = self.context.objective_schema.names
        if batch.objective_names != expected_names:
            raise ValueError(
                "Evaluated batch objective names do not match configured objectives; "
                f"expected={expected_names!r}, got={batch.objective_names!r}."
            )
        ordered = self._validate_and_order_evaluations(batch.evaluations, pending)
        fitnesses, valid_indexes = self._fitnesses(batch, ordered)
        generation_best = (
            ordered[max(valid_indexes, key=fitnesses.__getitem__)].individual
            if valid_indexes
            else None
        )
        generation_best_fitness = (
            max(fitnesses[index] for index in valid_indexes)
            if valid_indexes
            else None
        )
        global_best = self._global_best
        global_best_fitness = self._global_best_fitness
        if (
            generation_best is not None
            and generation_best_fitness is not None
            and (
                global_best_fitness is None
                or generation_best_fitness > global_best_fitness
            )
        ):
            global_best = generation_best
            global_best_fitness = generation_best_fitness

        if generation_best is not None:
            self._generation_bests.append(generation_best)
        self._generation_fitnesses.append(
            {
                evaluation.individual.id: fitness
                for evaluation, fitness in zip(ordered, fitnesses, strict=True)
            }
        )
        self._global_best = global_best
        self._global_best_fitness = global_best_fitness
        self._evaluations.extend(ordered)
        self._last_evaluations = ordered
        self._last_fitnesses = fitnesses
        self._last_valid_ids = tuple(
            ordered[index].individual.id for index in valid_indexes
        )
        self._pending_generation = None

    def should_stop(self) -> bool:
        """Report whether all configured generation numbers were proposed.

        Returns:
            ``True`` when the next generation number is greater than
            ``max_generations``. The final proposed generation may still be
            pending when this becomes true.
        """

        return self._next_generation > self.max_generations

    def best_individuals(self) -> Sequence[Individual]:
        """Return the globally best valid individual by committed fitness.

        Returns:
            A one-item tuple containing the best individual selected from
            scores committed by ``tell``, or an empty tuple if no valid
            evaluation has been committed.
        """

        return (self._global_best,) if self._global_best is not None else ()

    def generation_best_individuals(self) -> Sequence[Individual]:
        """Return each committed generation's best successful individual.

        Returns:
            Generation bests in commit order. Generations with no successful
            evaluations contribute no item.
        """

        return tuple(self._generation_bests)

    def generation_fitnesses(self) -> Sequence[Mapping[str, float]]:
        """Return per-generation selection fitness keyed by individual ID.

        Returns:
            Copies of the fitness mappings in committed generation order.
            Failed and objective-invalid individuals are present with zero
            fitness.
        """

        return tuple(dict(fitnesses) for fitnesses in self._generation_fitnesses)

    def checkpoint_signature(self) -> Mapping[str, object]:
        """Return immutable genetic strategy configuration.

        Returns:
            Genetic parameters and the optional initial population. Objective
            configuration is checkpointed by the session objective runtime.
        """

        return {
            "population_size": self.population_size,
            "mutation_probability": self.mutation_probability,
            "max_generations": self.max_generations,
            "start_generation": self.start_generation,
            "balanced_initialization": self.balanced_initialization,
            "initial_population": (
                [list(genotype) for genotype in self.initial_population]
                if self.initial_population is not None
                else None
            ),
        }

    def checkpoint_state(self) -> Mapping[str, object]:
        """Serialize only the RNG state not owned by session history.

        Returns:
            JSON-compatible state sufficient for deterministic continuation.

        Raises:
            CheckpointStateError: If an asked generation still awaits
                ``tell``. Only boundaries after a completed generation are
                checkpoint-safe.
        """

        if self._pending_generation is not None:
            raise CheckpointStateError(
                "GeneticSearch cannot checkpoint a generation awaiting tell()."
            )
        return {
            "random_state": encode_random_state(self.random.getstate()),
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, object],
        *,
        search_space: SearchSpace,
        evaluated_batches: Sequence[EvaluatedBatch] = (),
    ) -> None:
        """Restore genetic history, RNG, and the next generation number.

        Restored state is always at an ask/tell boundary with no pending
        generation and is bound to the supplied search space.

        Args:
            state: Encoded pseudo-random state.
            search_space: Search space used to continue the search.
            evaluated_batches: Authoritative generations restored by the session.

        Raises:
            CheckpointFormatError: If encoded values, population counts, or
                objective batches are inconsistent.
            ValueError: If the decoded object is not a valid ``Random`` state.
        """

        try:
            random_state = decode_random_state(state["random_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid GeneticSearch checkpoint state.") from exc
        if set(state) != {"random_state"}:
            raise CheckpointFormatError("Invalid GeneticSearch checkpoint fields.")
        asked_generations = len(evaluated_batches)
        next_generation = self.start_generation + asked_generations
        maximum_generations = max(
            0,
            self.max_generations - self.start_generation + 1,
        )
        if asked_generations > maximum_generations or any(
            len(batch.items) != self.population_size for batch in evaluated_batches
        ):
            raise CheckpointFormatError("GeneticSearch fitness history is inconsistent.")
        if self.context.objective_schema is None or any(
            batch.objective_names != self.context.objective_schema.names
            for batch in evaluated_batches
        ):
            raise CheckpointFormatError(
                "GeneticSearch objective batch schemas are inconsistent."
            )

        evaluations = [
            evaluation
            for batch in evaluated_batches
            for evaluation in batch.evaluations
        ]
        last_evaluations = (
            tuple(evaluated_batches[-1].evaluations) if evaluated_batches else ()
        )
        generation_fitnesses = [
            {
                item.individual.id: (
                    item.objective_values.aggregate_score
                    if item.valid
                    and item.objective_values is not None
                    and item.objective_values.aggregate_score is not None
                    else 0.0
                )
                for item in batch.items
            }
            for batch in evaluated_batches
        ]
        last_fitnesses = (
            tuple(
                generation_fitnesses[-1][item.individual.id]
                for item in evaluated_batches[-1].items
            )
            if evaluated_batches
            else ()
        )
        last_valid_ids = (
            tuple(
                item.individual.id
                for item in evaluated_batches[-1].items
                if item.valid
            )
            if evaluated_batches
            else ()
        )

        generation_bests: list[Individual] = []
        global_best: Individual | None = None
        global_best_fitness: float | None = None
        for batch, fitnesses in zip(
            evaluated_batches, generation_fitnesses, strict=True
        ):
            valid_items = tuple(item for item in batch.items if item.valid)
            if not valid_items:
                continue
            best_item = max(
                valid_items,
                key=lambda item: fitnesses[item.individual.id],
            )
            best_fitness = fitnesses[best_item.individual.id]
            generation_bests.append(best_item.individual)
            if global_best_fitness is None or best_fitness > global_best_fitness:
                global_best = best_item.individual
                global_best_fitness = best_fitness

        self.random.setstate(random_state)
        self._search_space = search_space
        self._pending_generation = None
        self._last_evaluations = last_evaluations
        self._last_fitnesses = last_fitnesses
        self._last_valid_ids = last_valid_ids
        self._evaluations = evaluations
        self._generation_bests = generation_bests
        self._generation_fitnesses = generation_fitnesses
        self._global_best = global_best
        self._global_best_fitness = global_best_fitness
        self._asked_generations = asked_generations
        self._next_generation = next_generation

    def _bind_search_space(self, search_space: SearchSpace) -> SearchSpace:
        """Bind once to a search space with non-empty genotype domains."""

        if self._search_space is None:
            if not search_space.genotype_lengths or any(
                length <= 0 for length in search_space.genotype_lengths
            ):
                raise ValueError("GeneticSearch requires non-empty genotype domains.")
            self._search_space = search_space
        elif self._search_space is not search_space:
            raise RuntimeError("GeneticSearch cannot be reused with another SearchSpace.")
        return search_space

    def _initial_candidates(self, search_space: SearchSpace) -> tuple[_Candidate, ...]:
        """Return configured genotypes or sample the first candidates."""

        if self.initial_population is not None:
            return tuple(
                _Candidate(genotype=genotype, origin="initial_population")
                for genotype in self.initial_population
            )
        return self._sample_candidates(search_space, origin="initialization")

    def _sample_candidates(
        self,
        search_space: SearchSpace,
        *,
        origin: str,
    ) -> tuple[_Candidate, ...]:
        """Sample a full population with the requested provenance label."""

        return tuple(
            _Candidate(
                genotype=self._sample_genotype(
                    search_space,
                    balanced=self.balanced_initialization,
                ),
                origin=origin,
            )
            for _ in range(self.population_size)
        )

    def _sample_genotype(
        self,
        search_space: SearchSpace,
        *,
        balanced: bool,
    ) -> tuple[int, ...]:
        """Sample one genotype, optionally balancing scenario stage groups.

        Slot genes use two-level stage-group sampling in balanced mode. Design
        genes are sampled uniformly regardless of that mode.
        """

        genes: list[int] = []
        slot_count = len(search_space.slot_lengths)
        for position, length in enumerate(search_space.genotype_lengths):
            if balanced and position < slot_count:
                _, gene = search_space.sample_slot_balanced(
                    position,
                    random=self.random,
                )
            else:
                gene = self.random.randrange(length)
            genes.append(gene)
        return tuple(genes)

    def _breed_generation(self, search_space: SearchSpace) -> tuple[_Candidate, ...]:
        """Create a full replacement generation from the last evaluations.

        Parents are selected in pairs, uniform crossover yields two children,
        and mutation is considered independently for each child. No parent is
        copied directly into the next generation.
        """

        roulette_weights = self._roulette_weights()
        candidates: list[_Candidate] = []
        for _ in range(self.population_size // 2):
            first_index, second_index = self._select_parent_indexes(roulette_weights)
            first = self._last_evaluations[first_index].individual
            second = self._last_evaluations[second_index].individual
            first_genotype = search_space.to_genotype(first)
            second_genotype = search_space.to_genotype(second)
            first_child, second_child = self._uniform_crossover(
                first_genotype,
                second_genotype,
            )
            parent_ids = (first.id, second.id)
            candidates.append(
                self._mutate_candidate(
                    search_space,
                    _Candidate(first_child, "crossover", parent_ids),
                )
            )
            candidates.append(
                self._mutate_candidate(
                    search_space,
                    _Candidate(second_child, "crossover", parent_ids),
                )
            )
        return tuple(candidates)

    def _roulette_weights(self) -> tuple[float, ...]:
        """Build non-negative parent weights from last-generation fitness.

        Fitness below the generation median is discarded. If the retained
        fitness sums to zero, every objective-valid evaluation receives unit
        weight and failed or invalid evaluations remain ineligible.
        """

        threshold = median(self._last_fitnesses)
        valid_ids = set(self._last_valid_ids)
        eligible = tuple(
            index
            for index, (evaluation, fitness) in enumerate(
                zip(self._last_evaluations, self._last_fitnesses, strict=True)
            )
            if evaluation.individual.id in valid_ids and fitness >= threshold
        )
        if not eligible:
            raise RuntimeError("Cannot select genetic parents without valid fitness.")
        minimum = min(self._last_fitnesses[index] for index in eligible)
        return tuple(
            self._last_fitnesses[index] - minimum + 1.0
            if index in eligible
            else 0.0
            for index in range(len(self._last_fitnesses))
        )

    def _select_parent_indexes(self, weights: Sequence[float]) -> tuple[int, int]:
        """Select two roulette points separated by half the total weight."""

        cumulative: list[float] = []
        total = 0.0
        for weight in weights:
            total += weight
            cumulative.append(total)
        if total <= 0:
            raise RuntimeError("Cannot select genetic parents without successful evaluations.")

        first_point = self.random.random() * total
        second_point = (first_point + total / 2.0) % total
        return (
            min(bisect_right(cumulative, first_point), len(cumulative) - 1),
            min(bisect_right(cumulative, second_point), len(cumulative) - 1),
        )

    def _uniform_crossover(
        self,
        first: tuple[int, ...],
        second: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Swap each pair of parent genes independently between two children."""

        first_child: list[int] = []
        second_child: list[int] = []
        for first_gene, second_gene in zip(first, second, strict=True):
            if self.random.randrange(2) == 0:
                first_child.append(first_gene)
                second_child.append(second_gene)
            else:
                first_child.append(second_gene)
                second_child.append(first_gene)
        return tuple(first_child), tuple(second_child)

    def _mutate_candidate(
        self,
        search_space: SearchSpace,
        candidate: _Candidate,
    ) -> _Candidate:
        """Possibly resample one gene and record whether its value changed.

        When mutation is selected, slot genes are resampled with stage-group
        balancing and design genes uniformly. Resampling can reproduce the
        current value, in which case ``mutation_changed`` is false.
        """

        if self.random.random() >= self.mutation_probability:
            return candidate

        position = self.random.randrange(len(candidate.genotype))
        genes = list(candidate.genotype)
        if position < len(search_space.slot_lengths):
            _, genes[position] = search_space.sample_slot_balanced(
                position,
                random=self.random,
            )
        else:
            genes[position] = self.random.randrange(
                search_space.genotype_lengths[position]
            )
        return _Candidate(
            genotype=tuple(genes),
            origin=candidate.origin,
            parent_ids=candidate.parent_ids,
            mutation_applied=True,
            mutation_changed=tuple(genes) != candidate.genotype,
        )

    def _materialize_population(
        self,
        search_space: SearchSpace,
        candidates: Sequence[_Candidate],
        *,
        generation: int,
    ) -> tuple[Individual, ...]:
        """Create individuals and attach generation provenance metadata."""

        if len(candidates) != self.population_size:
            raise RuntimeError(
                f"Expected {self.population_size} candidates, got {len(candidates)}."
            )
        return tuple(
            search_space.from_genotype(
                candidate.genotype,
                metadata={
                    "algorithm": {
                        "generation": generation,
                        "population_index": population_index,
                        "proposal_origin": candidate.origin,
                        "parent_ids": list(candidate.parent_ids),
                        "mutation_applied": candidate.mutation_applied,
                        "mutation_changed": candidate.mutation_changed,
                    }
                },
            )
            for population_index, candidate in enumerate(candidates)
        )

    def _validate_and_order_evaluations(
        self,
        evaluations: Sequence[Evaluation],
        pending: Sequence[Individual],
    ) -> tuple[Evaluation, ...]:
        """Validate one complete generation and restore proposal order."""

        if len(evaluations) != len(pending):
            raise ValueError(
                f"Expected {len(pending)} evaluations, got {len(evaluations)}."
            )

        evaluation_ids = [evaluation.individual.id for evaluation in evaluations]
        duplicates = sorted(
            identifier
            for identifier in set(evaluation_ids)
            if evaluation_ids.count(identifier) > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate evaluation individual IDs: {duplicates!r}.")

        expected = {individual.id: individual for individual in pending}
        received = set(evaluation_ids)
        if received != set(expected):
            missing = sorted(set(expected) - received)
            unexpected = sorted(received - set(expected))
            raise ValueError(
                f"Evaluation IDs do not match pending generation; "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )

        by_id = {evaluation.individual.id: evaluation for evaluation in evaluations}
        ordered = tuple(by_id[individual.id] for individual in pending)
        for individual, evaluation in zip(pending, ordered, strict=True):
            if evaluation.individual != individual:
                raise ValueError(
                    f"Evaluation individual {individual.id!r} does not match its proposal."
                )
            if evaluation.result.individual_id != individual.id:
                raise ValueError(
                    f"Result individual ID {evaluation.result.individual_id!r} does not "
                    f"match evaluation individual {individual.id!r}."
                )
        return ordered

    @staticmethod
    def _fitnesses(
        batch: EvaluatedBatch,
        evaluations: Sequence[Evaluation],
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        """Return aggregate scores and valid indexes in proposal order."""

        items_by_id = {
            item.evaluation.individual.id: item
            for item in batch.items
        }
        fitnesses: list[float] = []
        valid_indexes: list[int] = []
        for index, evaluation in enumerate(evaluations):
            item = items_by_id[evaluation.individual.id]
            if item.individual != item.evaluation.individual:
                raise ValueError(
                    f"Evaluated item {evaluation.individual.id!r} does not match "
                    "its evaluation."
                )
            if not item.valid or item.objective_values is None:
                fitnesses.append(0.0)
                continue
            score = item.objective_values.aggregate_score
            if (
                score is None
                or isinstance(score, bool)
                or not isinstance(score, Real)
                or not isfinite(float(score))
            ):
                raise ValueError(
                    f"Aggregate score for individual {evaluation.individual.id!r} "
                    "must be a finite real number."
                )
            fitnesses.append(float(score))
            valid_indexes.append(index)
        return tuple(fitnesses), tuple(valid_indexes)

    @staticmethod
    def _validate_configuration(
        *,
        population_size: int,
        mutation_probability: float,
        max_generations: int,
        start_generation: int,
        balanced_initialization: bool,
    ) -> None:
        """Validate generation, population, and mutation scalar settings."""

        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size <= 0
            or population_size % 2 != 0
        ):
            raise ValueError("population_size must be a positive even integer.")
        if (
            isinstance(max_generations, bool)
            or not isinstance(max_generations, int)
            or max_generations < 0
        ):
            raise ValueError("max_generations must be a non-negative integer.")
        if (
            isinstance(start_generation, bool)
            or not isinstance(start_generation, int)
            or start_generation <= 0
        ):
            raise ValueError("start_generation must be a positive integer.")
        if (
            isinstance(mutation_probability, bool)
            or not isinstance(mutation_probability, Real)
            or not isfinite(float(mutation_probability))
            or not 0 <= mutation_probability <= 1
        ):
            raise ValueError("mutation_probability must be between 0 and 1.")
        if not isinstance(balanced_initialization, bool):
            raise ValueError("balanced_initialization must be a boolean.")

__all__ = ["GeneticSearch"]
