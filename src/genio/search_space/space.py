from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from genio.core import Individual, StageChoice

from .spec import SearchScenarioSpec


@dataclass(slots=True)
class SearchSpace:
    """Factory and context for Individuals belonging to one search scenario.

    Each slot contains only valid concrete alternatives. A genotype is a tuple
    of local sequence numbers, one per slot. The global search_index is the
    mixed-radix encoding of that genotype using each slot cardinality as base.
    """

    scenario: SearchScenarioSpec
    _next_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.scenario.slots:
            raise ValueError("SearchSpace requires at least one slot.")

        expected = tuple(range(len(self.scenario.slots)))
        actual = tuple(slot.index for slot in self.scenario.slots)
        if actual != expected:
            raise ValueError(
                "Slot indexes must be contiguous and ordered from 0; "
                f"got {actual!r}."
            )

        empty_slots = [slot.index for slot in self.scenario.slots if not slot.alternatives]
        if empty_slots:
            raise ValueError(f"Slots without alternatives: {empty_slots!r}.")

        for slot in self.scenario.slots:
            for alternative in slot.alternatives:
                if alternative.slot != slot.index:
                    raise ValueError(
                        f"Alternative {alternative!r} belongs to slot "
                        f"{alternative.slot}, but is stored in slot {slot.index}."
                    )

    @property
    def scenario_id(self) -> str:
        return self.scenario.id

    @property
    def scenario_version(self) -> str | None:
        return self.scenario.version

    @property
    def slot_lengths(self) -> tuple[int, ...]:
        return tuple(len(slot.alternatives) for slot in self.scenario.slots)

    @property
    def cardinality(self) -> int:
        total = 1
        for length in self.slot_lengths:
            total *= length
        return total

    def from_genotype(
        self,
        genotype: tuple[int, ...] | list[int],
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Individual:
        normalized_genotype = tuple(genotype)
        self._validate_genotype(normalized_genotype)

        slots = tuple(
            self.scenario.slots[slot_index].alternatives[gene]
            for slot_index, gene in enumerate(normalized_genotype)
        )

        return Individual.from_slots(
            id=id or self._new_id(),
            scenario=self.scenario.id,
            slots=slots,
            genotype=normalized_genotype,
            search_index=self.genotype_to_index(normalized_genotype),
            metadata=metadata,
        )

    def from_index(
        self,
        search_index: int,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Individual:
        return self.from_genotype(
            self.index_to_genotype(search_index),
            id=id,
            metadata=metadata,
        )

    def from_slots(
        self,
        slots: list[StageChoice] | tuple[StageChoice, ...],
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Individual:
        genotype = self.slots_to_genotype(tuple(slots))
        return self.from_genotype(genotype, id=id, metadata=metadata)

    def sample(
        self,
        *,
        random: Random | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Individual:
        rng = random or Random()
        genotype = tuple(rng.randrange(length) for length in self.slot_lengths)
        return self.from_genotype(genotype, metadata=metadata)

    def sample_population(
        self,
        size: int,
        *,
        unique: bool = True,
        random: Random | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Individual]:
        if size < 0:
            raise ValueError("Population size cannot be negative.")
        if unique and size > self.cardinality:
            raise ValueError(
                f"Cannot sample {size} unique individuals from a search space "
                f"with cardinality {self.cardinality}."
            )

        population: list[Individual] = []
        seen: set[int] = set()
        rng = random or Random()

        while len(population) < size:
            individual = self.sample(random=rng, metadata=metadata)
            if unique:
                assert individual.search_index is not None
                if individual.search_index in seen:
                    continue
                seen.add(individual.search_index)
            population.append(individual)

        return population

    def genotype_to_index(self, genotype: tuple[int, ...] | list[int]) -> int:
        normalized_genotype = tuple(genotype)
        self._validate_genotype(normalized_genotype)

        index = 0
        for gene, base in zip(normalized_genotype, self.slot_lengths):
            index = index * base + gene
        return index

    def index_to_genotype(self, search_index: int) -> tuple[int, ...]:
        if search_index < 0:
            raise ValueError("search_index cannot be negative.")
        if search_index >= self.cardinality:
            raise ValueError(
                f"search_index {search_index} is out of range for cardinality "
                f"{self.cardinality}."
            )

        remaining = search_index
        genes: list[int] = []

        for base in reversed(self.slot_lengths):
            genes.append(remaining % base)
            remaining //= base

        return tuple(reversed(genes))

    def slots_to_genotype(self, slots: tuple[StageChoice, ...]) -> tuple[int, ...]:
        if len(slots) != len(self.scenario.slots):
            raise ValueError(
                f"Expected {len(self.scenario.slots)} slots, got {len(slots)}."
            )

        by_slot = {choice.slot: choice for choice in slots}
        if len(by_slot) != len(slots):
            raise ValueError("Duplicate slot choices are not allowed.")

        genes: list[int] = []
        for slot in self.scenario.slots:
            try:
                choice = by_slot[slot.index]
            except KeyError as exc:
                raise ValueError(f"Missing slot {slot.index}.") from exc

            try:
                genes.append(slot.alternatives.index(choice))
            except ValueError as exc:
                raise ValueError(
                    f"Choice {choice!r} is not a valid alternative for slot "
                    f"{slot.index}."
                ) from exc

        return tuple(genes)

    def to_index(self, individual: Individual) -> int:
        if individual.scenario != self.scenario.id:
            raise ValueError(
                f"Individual scenario {individual.scenario!r} does not match "
                f"search space scenario {self.scenario.id!r}."
            )
        return self.genotype_to_index(self.to_genotype(individual))

    def to_genotype(self, individual: Individual) -> tuple[int, ...]:
        genotype = self.slots_to_genotype(individual.slots)
        if individual.genotype is not None and individual.genotype != genotype:
            raise ValueError(
                f"Individual genotype {individual.genotype!r} does not match "
                f"its slots genotype {genotype!r}."
            )
        return genotype

    def _validate_genotype(self, genotype: tuple[int, ...]) -> None:
        if len(genotype) != len(self.slot_lengths):
            raise ValueError(
                f"Expected genotype length {len(self.slot_lengths)}, "
                f"got {len(genotype)}."
            )

        for slot_index, (gene, length) in enumerate(zip(genotype, self.slot_lengths)):
            if gene < 0 or gene >= length:
                raise ValueError(
                    f"Gene {gene} for slot {slot_index} is out of range "
                    f"[0, {length})."
                )

    def _new_id(self) -> str:
        value = self._next_id
        self._next_id += 1
        return f"{self.scenario.id}_{value:06d}"
