from __future__ import annotations

import json
import ast
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

from genio.core import Individual, StageChoice

from .spec import SearchScenarioSpec
from .spec import SlotSpec


def _is_odd(value: int) -> bool:
    return value % 2 == 1


def _is_even(value: int) -> bool:
    return value % 2 == 0


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _divisible_by(value: int, divisor: int) -> bool:
    return divisor != 0 and value % divisor == 0


def _square(rows: int, cols: int) -> bool:
    return rows == cols


@dataclass(init=False, slots=True)
class SearchSpace:
    """Factory and context for Individuals belonging to one search scenario.

    Each slot contains only valid concrete alternatives. A genotype is a tuple
    of local sequence numbers, one per slot. The global search_index is the
    mixed-radix encoding of that genotype using each slot length as base.
    """

    scenario: SearchScenarioSpec
    _next_id: int = field(default=0, init=False, repr=False)
    _ALLOWED_CONSTRAINT_FUNCTIONS = {
        "abs": abs,
        "divisible_by": _divisible_by,
        "is_even": _is_even,
        "is_odd": _is_odd,
        "is_power_of_two": _is_power_of_two,
        "max": max,
        "min": min,
        "round": round,
        "square": _square,
    }

    def __init__(
        self,
        test_file: str | Path,
        stages_definitions_path: str | Path,
    ) -> None:
        self.scenario = self._load_scenario(
            Path(test_file),
            Path(stages_definitions_path),
        )
        self._next_id = 0
        self.__post_init__()

    @classmethod
    def from_scenario(cls, scenario: SearchScenarioSpec) -> "SearchSpace":
        search_space = cls.__new__(cls)
        search_space.scenario = scenario
        search_space._next_id = 0
        search_space.__post_init__()
        return search_space

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
    def slot_lengths(self) -> tuple[int, ...]:
        return tuple(len(slot.alternatives) for slot in self.scenario.slots)

    @property
    def search_space_size(self) -> int:
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
        if unique and size > self.search_space_size:
            raise ValueError(
                f"Cannot sample {size} unique individuals from a search space "
                f"with size {self.search_space_size}."
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
        if search_index >= self.search_space_size:
            raise ValueError(
                f"search_index {search_index} is out of range for search space "
                f"size {self.search_space_size}."
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

    @classmethod
    def _load_scenario(
        cls,
        test_file: Path,
        stages_definitions_path: Path,
    ) -> SearchScenarioSpec:
        with test_file.open("r", encoding="utf-8") as file:
            test_config = json.load(file)

        scenario_id = test_config["id"]
        stage_definitions = cls._load_stage_definitions(stages_definitions_path)
        slots = tuple(
            cls._build_slot_spec(slot_config, stage_definitions)
            for slot_config in test_config["pipeline"]
        )

        metadata = {
            key: value
            for key, value in test_config.items()
            if key not in {"id", "pipeline"}
        }

        return SearchScenarioSpec(id=scenario_id, slots=slots, metadata=metadata)

    @staticmethod
    def _load_stage_definitions(
        stages_definitions_path: Path,
    ) -> dict[str, dict[str, Any]]:
        definitions: dict[str, dict[str, Any]] = {}

        for definition_file in stages_definitions_path.glob("*/*.json"):
            with definition_file.open("r", encoding="utf-8") as file:
                definition = json.load(file)
            definitions[definition["id"]] = definition

        return definitions

    @classmethod
    def _build_slot_spec(
        cls,
        slot_config: dict[str, Any],
        stage_definitions: dict[str, dict[str, Any]],
    ) -> SlotSpec:

        slot_index = slot_config["slot"]
        alternatives: list[StageChoice] = []

        for candidate_config in slot_config["candidates"]:
            stage = candidate_config["stage"]
            if stage not in stage_definitions:
                raise ValueError(
                    f"Stage {stage!r} is not defined in stages definitions path."
                )
            alternatives.extend(
                cls._expand_candidate(
                    slot_index,
                    candidate_config,
                    stage_definitions[stage],
                )
            )

        return SlotSpec(index=slot_index, alternatives=tuple(alternatives))

    @classmethod
    def _expand_candidate(
        cls,
        slot_index: int,
        candidate_config: dict[str, Any],
        stage_definition: dict[str, Any],
    ) -> list[StageChoice]:
        stage = candidate_config["stage"]
        parameter_specs = candidate_config.get("parameters", {})
        wrapper_inputs = candidate_config.get("wrapper_inputs", {})

        if not parameter_specs:
            return [
                StageChoice(
                    slot=slot_index,
                    stage=stage,
                    wrapper_inputs=dict(wrapper_inputs),
                )
            ]

        parameter_names = tuple(parameter_specs)
        parameter_values = tuple(
            cls._parameter_values(parameter_specs[name]) for name in parameter_names
        )

        alternatives: list[StageChoice] = []
        for values in product(*parameter_values):
            parameters = dict(zip(parameter_names, values))
            if not cls._satisfies_constraints(parameters, stage_definition):
                continue

            alternatives.append(
                StageChoice(
                    slot=slot_index,
                    stage=stage,
                    parameters=parameters,
                    wrapper_inputs=dict(wrapper_inputs),
                )
            )

        return alternatives

    @staticmethod
    def _parameter_values(parameter_spec: dict[str, Any]) -> tuple[Any, ...]:
        parameter_type = parameter_spec["type"]

        if parameter_type == "constant":
            return (parameter_spec["value"],)

        if parameter_type == "choice":
            return tuple(parameter_spec["values"])

        if parameter_type == "integer_range":
            return tuple(
                range(
                    parameter_spec["start"],
                    parameter_spec["stop"],
                    parameter_spec.get("step", 1),
                )
            )

        if parameter_type == "number_range":
            values: list[float] = []
            current = parameter_spec["start"]
            stop = parameter_spec["stop"]
            step = parameter_spec["step"]
            while current < stop:
                values.append(current)
                current += step
            return tuple(values)

        raise ValueError(f"Unsupported parameter spec type: {parameter_type!r}.")

    @classmethod
    def _satisfies_constraints(
        cls,
        parameters: dict[str, Any],
        stage_definition: dict[str, Any],
    ) -> bool:
        constraints = stage_definition.get("constraints", [])
        if not constraints:
            return True

        token_values = cls._constraint_token_values(parameters, stage_definition)
        return all(cls._evaluate_constraint(constraint, token_values) for constraint in constraints)

    @staticmethod
    def _constraint_token_values(
        parameters: dict[str, Any],
        stage_definition: dict[str, Any],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for parameter_definition in stage_definition.get("parameters", []):
            name = parameter_definition["name"]
            token = parameter_definition["token"]
            if name in parameters:
                values[token] = parameters[name]
        return values

    @classmethod
    def _evaluate_constraint(
        cls,
        constraint: str,
        token_values: dict[str, Any],
    ) -> bool:
        expression = constraint
        for token, value in sorted(token_values.items(), key=lambda item: len(item[0]), reverse=True):
            expression = expression.replace(token, repr(value))

        if "@" in expression:
            raise ValueError(f"Constraint {constraint!r} contains unresolved tokens.")

        parsed = ast.parse(expression, mode="eval")
        return bool(cls._eval_constraint_node(parsed.body))

    @classmethod
    def _eval_constraint_node(cls, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.List):
            return [cls._eval_constraint_node(element) for element in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(cls._eval_constraint_node(element) for element in node.elts)

        if isinstance(node, ast.Set):
            return {cls._eval_constraint_node(element) for element in node.elts}

        if isinstance(node, ast.Compare):
            left = cls._eval_constraint_node(node.left)
            for operator, comparator_node in zip(node.ops, node.comparators):
                right = cls._eval_constraint_node(comparator_node)
                if not cls._compare(left, operator, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.BoolOp):
            values = [cls._eval_constraint_node(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -cls._eval_constraint_node(node.operand)

        if isinstance(node, ast.BinOp):
            left = cls._eval_constraint_node(node.left)
            right = cls._eval_constraint_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed in constraints.")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed in constraints.")

            function_name = node.func.id
            try:
                function = cls._ALLOWED_CONSTRAINT_FUNCTIONS[function_name]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported constraint function: {function_name!r}."
                ) from exc

            arguments = [cls._eval_constraint_node(argument) for argument in node.args]
            return function(*arguments)

        raise ValueError(f"Unsupported constraint expression: {ast.dump(node)}")

    @staticmethod
    def _compare(left: Any, operator: ast.cmpop, right: Any) -> bool:
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.Gt):
            return left > right
        if isinstance(operator, ast.GtE):
            return left >= right
        if isinstance(operator, ast.In):
            return left in right
        if isinstance(operator, ast.NotIn):
            return left not in right
        raise ValueError(f"Unsupported constraint operator: {operator!r}.")
