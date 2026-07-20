from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from random import Random
from statistics import median
from typing import TYPE_CHECKING

from genio.algorithm.base import SearchAlgorithm
from genio.checkpoint.codec import (
    decode_evaluation,
    decode_individual,
    decode_random_state,
    encode_evaluation,
    encode_individual,
    encode_random_state,
)
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


@dataclass(frozen=True, slots=True)
class _Candidate:
    genotype: tuple[int, ...]
    origin: str
    parent_ids: tuple[str, ...] = ()
    mutation_applied: bool = False
    mutation_changed: bool = False


class GeneticSearch(SearchAlgorithm):
    """Run the legacy generational genetic search through the ask/tell contract."""

    supports_checkpointing = True

    def __init__(
        self,
        *,
        objectives: Objective | ObjectiveSet,
        weights: Mapping[str, float] | None = None,
        population_size: int = 80,
        mutation_probability: float = 0.05,
        max_generations: int = 20,
        start_generation: int = 1,
        balanced_initialization: bool = True,
        initial_population: Sequence[Sequence[int]] | None = None,
        random: Random | None = None,
    ) -> None:
        self._validate_configuration(
            population_size=population_size,
            mutation_probability=mutation_probability,
            max_generations=max_generations,
            start_generation=start_generation,
        )
        self.objectives = (
            objectives
            if isinstance(objectives, ObjectiveSet)
            else ObjectiveSet((objectives,))
        )
        self.weights = self._validate_weights(weights)
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
        if start_generation != 1 and self.initial_population is None:
            raise ValueError(
                "initial_population is required when start_generation is not 1."
            )

        self.random = random or Random()
        self._search_space: SearchSpace | None = None
        self._pending_generation: tuple[Individual, ...] | None = None
        self._last_evaluations: tuple[Evaluation, ...] = ()
        self._last_fitnesses: tuple[float, ...] = ()
        self._evaluations: list[Evaluation] = []
        self._generation_bests: list[Individual] = []
        self._generation_fitnesses: list[dict[str, float]] = []
        self._global_best: Individual | None = None
        self._asked_generations = 0
        self._next_generation = start_generation

    def ask(self, session: OptimizationSession) -> Sequence[Individual]:
        """Return the complete initial population or next offspring generation."""

        if self._pending_generation is not None:
            raise RuntimeError("tell() is required before asking for another generation.")
        if self.should_stop():
            return ()

        search_space = self._bind_search_space(session.search_space)
        generation = self._next_generation
        if self._asked_generations == 0:
            candidates = self._initial_candidates(search_space)
        elif self._successful_indexes(self._last_evaluations):
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

    def tell(self, evaluations: Sequence[Evaluation]) -> None:
        """Record and score one complete evaluated generation."""

        pending = self._pending_generation
        if pending is None:
            raise RuntimeError("tell() requires a pending generation from ask().")

        ordered = self._validate_and_order_evaluations(evaluations, pending)
        fitnesses = self._fitnesses(ordered)
        successful = self._successful_indexes(ordered)
        generation_best = (
            ordered[max(successful, key=fitnesses.__getitem__)].individual
            if successful
            else None
        )

        complete_history = (*self._evaluations, *ordered)
        history_fitnesses = self._fitnesses(complete_history)
        history_successful = self._successful_indexes(complete_history)
        global_best = (
            complete_history[
                max(history_successful, key=history_fitnesses.__getitem__)
            ].individual
            if history_successful
            else None
        )

        if generation_best is not None:
            self._generation_bests.append(generation_best)
        self._generation_fitnesses.append(
            {
                evaluation.individual.id: fitness
                for evaluation, fitness in zip(ordered, fitnesses, strict=True)
            }
        )
        self._global_best = global_best
        self._evaluations.extend(ordered)
        self._last_evaluations = ordered
        self._last_fitnesses = fitnesses
        self._pending_generation = None

    def should_stop(self) -> bool:
        """Return whether all configured generations have been proposed."""

        return self._next_generation > self.max_generations

    def best_individuals(self) -> Sequence[Individual]:
        """Return the globally best successful individual under weighted fitness."""

        return (self._global_best,) if self._global_best is not None else ()

    def generation_best_individuals(self) -> Sequence[Individual]:
        """Return the best successful individual recorded for each generation."""

        return tuple(self._generation_bests)

    def generation_fitnesses(self) -> Sequence[Mapping[str, float]]:
        """Return selection fitness values keyed by individual for each generation."""

        return tuple(dict(fitnesses) for fitnesses in self._generation_fitnesses)

    def checkpoint_signature(self) -> Mapping[str, object]:
        """Return immutable genetic configuration and objective definitions."""

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
            "mutation_probability": self.mutation_probability,
            "max_generations": self.max_generations,
            "start_generation": self.start_generation,
            "balanced_initialization": self.balanced_initialization,
            "initial_population": (
                [list(genotype) for genotype in self.initial_population]
                if self.initial_population is not None
                else None
            ),
            "objectives": [
                dict(objective.checkpoint_signature())
                for objective in self.objectives.objectives
            ],
            "weights": list(self.weights),
        }

    def checkpoint_state(self) -> Mapping[str, object]:
        """Return committed genetic state when no generation awaits evaluation."""

        if self._pending_generation is not None:
            raise CheckpointStateError(
                "GeneticSearch cannot checkpoint a generation awaiting tell()."
            )
        return {
            "random_state": encode_random_state(self.random.getstate()),
            "last_evaluations": [
                encode_evaluation(evaluation) for evaluation in self._last_evaluations
            ],
            "last_fitnesses": list(self._last_fitnesses),
            "evaluations": [
                encode_evaluation(evaluation) for evaluation in self._evaluations
            ],
            "generation_bests": [
                encode_individual(individual) for individual in self._generation_bests
            ],
            "generation_fitnesses": self._generation_fitnesses,
            "global_best": (
                encode_individual(self._global_best)
                if self._global_best is not None
                else None
            ),
            "asked_generations": self._asked_generations,
            "next_generation": self._next_generation,
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, object],
        *,
        version: int,
        search_space: SearchSpace,
    ) -> None:
        """Restore genetic history, RNG and next generation at a safe boundary."""

        if version != self.checkpoint_version:
            raise CheckpointFormatError(
                f"Unsupported GeneticSearch checkpoint version {version}."
            )
        try:
            random_state = decode_random_state(state["random_state"])
            last_evaluations = tuple(
                decode_evaluation(value, search_space)
                for value in state.get("last_evaluations", [])
            )
            last_fitnesses = tuple(
                float(value) for value in state.get("last_fitnesses", [])
            )
            evaluations = [
                decode_evaluation(value, search_space)
                for value in state.get("evaluations", [])
            ]
            generation_bests = [
                decode_individual(value, search_space)
                for value in state.get("generation_bests", [])
            ]
            generation_fitnesses = [
                {str(identifier): float(fitness) for identifier, fitness in value.items()}
                for value in state.get("generation_fitnesses", [])
            ]
            global_best_value = state.get("global_best")
            global_best = (
                decode_individual(global_best_value, search_space)
                if global_best_value is not None
                else None
            )
            asked_generations = int(state["asked_generations"])
            next_generation = int(state["next_generation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid GeneticSearch checkpoint state.") from exc
        if len(last_evaluations) != len(last_fitnesses):
            raise CheckpointFormatError(
                "GeneticSearch last evaluations and fitnesses have different lengths."
            )
        if asked_generations < 0 or next_generation <= 0:
            raise CheckpointFormatError("Invalid GeneticSearch generation counters.")
        if len(evaluations) != asked_generations * self.population_size:
            raise CheckpointFormatError("GeneticSearch evaluation history is inconsistent.")
        if next_generation != self.start_generation + asked_generations:
            raise CheckpointFormatError("GeneticSearch next generation is inconsistent.")
        if len(generation_fitnesses) != asked_generations:
            raise CheckpointFormatError("GeneticSearch fitness history is inconsistent.")
        if len(generation_bests) > asked_generations:
            raise CheckpointFormatError("GeneticSearch best history is inconsistent.")
        if asked_generations == 0 and (last_evaluations or last_fitnesses):
            raise CheckpointFormatError("GeneticSearch has unexpected last-generation state.")
        if asked_generations > 0 and len(last_evaluations) != self.population_size:
            raise CheckpointFormatError("GeneticSearch last generation has invalid size.")

        self.random.setstate(random_state)
        self._search_space = search_space
        self._pending_generation = None
        self._last_evaluations = last_evaluations
        self._last_fitnesses = last_fitnesses
        self._evaluations = evaluations
        self._generation_bests = generation_bests
        self._generation_fitnesses = generation_fitnesses
        self._global_best = global_best
        self._asked_generations = asked_generations
        self._next_generation = next_generation

    def _bind_search_space(self, search_space: SearchSpace) -> SearchSpace:
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
        threshold = median(self._last_fitnesses)
        retained = tuple(
            fitness if fitness >= threshold else 0.0
            for fitness in self._last_fitnesses
        )
        if sum(retained) > 0:
            return retained

        successful = set(self._successful_indexes(self._last_evaluations))
        return tuple(
            1.0 if index in successful else 0.0
            for index in range(len(self._last_evaluations))
        )

    def _select_parent_indexes(self, weights: Sequence[float]) -> tuple[int, int]:
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

    def _fitnesses(self, evaluations: Sequence[Evaluation]) -> tuple[float, ...]:
        fitnesses = [0.0] * len(evaluations)
        successful = self._successful_indexes(evaluations)
        if not successful:
            return tuple(fitnesses)

        for objective, weight in zip(
            self.objectives.objectives,
            self.weights,
            strict=True,
        ):
            values = [objective.value(evaluations[index]) for index in successful]
            if any(not isfinite(value) for value in values):
                raise ValueError(
                    f"Objective {objective.name!r} produced a non-finite value."
                )
            minimum = min(values)
            maximum = max(values)
            for index, value in zip(successful, values, strict=True):
                if maximum == minimum:
                    normalized = 0.0
                elif objective.direction is OptimizationDirection.MAXIMIZE:
                    normalized = (value - minimum) / (maximum - minimum)
                else:
                    normalized = (maximum - value) / (maximum - minimum)
                fitnesses[index] += weight * normalized
        return tuple(fitnesses)

    @staticmethod
    def _successful_indexes(evaluations: Sequence[Evaluation]) -> tuple[int, ...]:
        return tuple(
            index
            for index, evaluation in enumerate(evaluations)
            if evaluation.result.status is ResultStatus.SUCCESS
        )

    @staticmethod
    def _validate_configuration(
        *,
        population_size: int,
        mutation_probability: float,
        max_generations: int,
        start_generation: int,
    ) -> None:
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

    def _validate_weights(self, weights: Mapping[str, float] | None) -> tuple[float, ...]:
        names = tuple(objective.name for objective in self.objectives.objectives)
        configured = dict(weights) if weights is not None else dict.fromkeys(names, 1.0)
        if set(configured) != set(names):
            missing = sorted(set(names) - set(configured))
            unexpected = sorted(set(configured) - set(names))
            raise ValueError(
                f"weights must match objective names; missing={missing!r}, "
                f"unexpected={unexpected!r}."
            )

        configured_weights: list[float] = []
        for name in names:
            weight = configured[name]
            if (
                isinstance(weight, bool)
                or not isinstance(weight, Real)
                or not isfinite(float(weight))
                or weight < 0
            ):
                raise ValueError(f"Weight for objective {name!r} must be finite and non-negative.")
            configured_weights.append(float(weight))
        if not any(weight > 0 for weight in configured_weights):
            raise ValueError("At least one objective weight must be positive.")

        scale = max(configured_weights)
        scaled_weights = [weight / scale for weight in configured_weights]
        total = sum(scaled_weights)
        return tuple(weight / total for weight in scaled_weights)


__all__ = ["GeneticSearch"]
