from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from genio.core import Individual, StageChoice
from genio.search_space import SearchSpace


class ComposerError(Exception):
    """Base error for composer failures."""


class StageDefinitionNotFoundError(ComposerError):
    """Raised when an Individual references an unknown stage."""


class Composer(ABC):
    """Base class for backend-specific composers.

    The base composer owns common traversal and definition-loading logic.
    Subclasses decide what execution artifacts to generate for each backend.
    """

    def __init__(self, stages_definitions_path: str | Path) -> None:
        self.stages_definitions_path = Path(stages_definitions_path)
        self.stage_definitions = SearchSpace._load_stage_definitions(
            self.stages_definitions_path
        )

    @abstractmethod
    def compose(self, individual: Individual) -> Any:
        """Generate backend-specific execution artifacts from an Individual."""

    def active_choices(self, individual: Individual) -> Iterator[StageChoice]:
        for choice in individual.slots:
            if not self.should_skip(choice):
                yield choice

    def active_stage_definitions(
        self,
        individual: Individual,
    ) -> Iterator[tuple[StageChoice, dict[str, Any]]]:
        for choice in self.active_choices(individual):
            yield choice, self.stage_definition(choice.stage)

    def should_skip(self, choice: StageChoice) -> bool:
        return choice.stage == "nop"

    def stage_definition(self, stage: str) -> dict[str, Any]:
        try:
            return self.stage_definitions[stage]
        except KeyError as exc:
            raise StageDefinitionNotFoundError(
                f"Stage {stage!r} is not defined in {self.stages_definitions_path}."
            ) from exc

    def artifact_metadata(self, individual: Individual) -> dict[str, Any]:
        return {
            "composer": self.__class__.__name__,
            "scenario": individual.scenario,
            "search_index": individual.search_index,
        }
