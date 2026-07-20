from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from genio.core.evaluation import Evaluation
from genio.core.individual import Individual, StageChoice
from genio.core.result import Result, ResultStatus
from genio.search_space.space import SearchSpace

from .errors import CheckpointFormatError


def encode_random_state(value: object) -> object:
    """Encode nested tuples from Random.getstate() into JSON-compatible lists."""

    if isinstance(value, tuple):
        return [encode_random_state(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported random state value {value!r}.")


def decode_random_state(value: object) -> object:
    """Restore nested tuples required by Random.setstate()."""

    if isinstance(value, list):
        return tuple(decode_random_state(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise CheckpointFormatError(f"Invalid encoded random state value {value!r}.")


def encode_individual(individual: Individual) -> dict[str, Any]:
    """Serialize an individual without relying on Python object pickling."""

    return {
        "id": individual.id,
        "scenario": individual.scenario,
        "slots": [
            {
                "slot": choice.slot,
                "stage": choice.stage,
                "parameters": choice.parameters,
                "wrapper_inputs": choice.wrapper_inputs,
            }
            for choice in individual.slots
        ],
        "genotype": list(individual.genotype) if individual.genotype is not None else None,
        "search_index": individual.search_index,
        "design": individual.design,
        "metadata": individual.metadata,
    }


def decode_individual(value: Mapping[str, Any], search_space: SearchSpace) -> Individual:
    """Deserialize and validate an individual against the configured search space."""

    try:
        identifier = str(value["id"])
        scenario = str(value["scenario"])
        genotype_value = value["genotype"]
        metadata = dict(value.get("metadata", {}))
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError("Invalid individual checkpoint payload.") from exc
    if scenario != search_space.scenario_id:
        raise CheckpointFormatError(
            f"Individual scenario {scenario!r} does not match {search_space.scenario_id!r}."
        )

    if genotype_value is not None:
        if not isinstance(genotype_value, list):
            raise CheckpointFormatError("Individual genotype must be a list or null.")
        individual = search_space.from_genotype(
            tuple(genotype_value),
            id=identifier,
            metadata=metadata,
        )
        if individual.search_index != value.get("search_index"):
            raise CheckpointFormatError(
                f"Individual {identifier!r} has an inconsistent search index."
            )
        if individual.design != value.get("design", {}):
            raise CheckpointFormatError(
                f"Individual {identifier!r} has inconsistent design values."
            )
        return individual

    try:
        slots = tuple(
            StageChoice(
                slot=int(choice["slot"]),
                stage=str(choice["stage"]),
                parameters=dict(choice.get("parameters", {})),
                wrapper_inputs=dict(choice.get("wrapper_inputs", {})),
            )
            for choice in value["slots"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError("Invalid individual slot payload.") from exc
    return Individual.from_slots(
        id=identifier,
        scenario=scenario,
        slots=slots,
        genotype=None,
        search_index=value.get("search_index"),
        design=dict(value.get("design", {})),
        metadata=metadata,
    )


def encode_evaluation(evaluation: Evaluation) -> dict[str, Any]:
    """Serialize an evaluation and its normalized result."""

    return {
        "individual": encode_individual(evaluation.individual),
        "result": {
            "individual_id": evaluation.result.individual_id,
            "status": evaluation.result.status.value,
            "metrics": evaluation.result.metrics,
            "error": evaluation.result.error,
            "metadata": evaluation.result.metadata,
        },
        "metadata": evaluation.metadata,
    }


def decode_evaluation(value: Mapping[str, Any], search_space: SearchSpace) -> Evaluation:
    """Deserialize an evaluation and validate individual/result identity."""

    try:
        individual = decode_individual(value["individual"], search_space)
        result_value = value["result"]
        result = Result(
            individual_id=str(result_value["individual_id"]),
            status=ResultStatus(str(result_value["status"])),
            metrics={
                str(name): float(metric)
                for name, metric in dict(result_value.get("metrics", {})).items()
            },
            error=(
                str(result_value["error"])
                if result_value.get("error") is not None
                else None
            ),
            metadata=dict(result_value.get("metadata", {})),
        )
        metadata = dict(value.get("metadata", {}))
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError("Invalid evaluation checkpoint payload.") from exc
    if result.individual_id != individual.id:
        raise CheckpointFormatError(
            f"Result ID {result.individual_id!r} does not match individual "
            f"{individual.id!r}."
        )
    return Evaluation(individual=individual, result=result, metadata=metadata)


def qualified_name(value: object | type[object]) -> str:
    """Return a stable qualified class name for compatibility signatures."""

    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def signature_value(value: Any) -> Any:
    """Convert supported configuration values to canonical JSON structures."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, type):
        return qualified_name(value)
    if isinstance(value, Mapping):
        return {str(key): signature_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [signature_value(item) for item in value]
    checkpoint_signature = getattr(value, "checkpoint_signature", None)
    if callable(checkpoint_signature):
        return signature_value(checkpoint_signature())
    raise TypeError(
        f"Configuration object {qualified_name(value)} must implement "
        "checkpoint_signature()."
    )


__all__ = [
    "decode_evaluation",
    "decode_individual",
    "decode_random_state",
    "encode_evaluation",
    "encode_individual",
    "encode_random_state",
    "qualified_name",
    "signature_value",
]
