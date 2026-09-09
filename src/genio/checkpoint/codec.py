from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from genio.core.evaluation import Evaluation
from genio.core.individual import Individual, StageChoice
from genio.core.result import Result, ResultStatus
from genio.objective.normalization import Normalizer
from genio.objective.runtime import (
    EvaluatedBatch,
    EvaluatedIndividual,
    ObjectiveEvaluationStatus,
    ObjectiveValues,
)
from genio.search_space.space import SearchSpace

from .errors import CheckpointFormatError


def encode_random_state(value: object) -> object:
    """Encode nested tuples from Random.getstate() into JSON-compatible lists."""

    if isinstance(value, tuple):
        return [encode_random_state(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported random state value {value!r}.")


def decode_random_state(value: object) -> tuple[Any, ...]:
    """Restore nested tuples required by Random.setstate()."""

    decoded = _decode_random_state_value(value)
    if not isinstance(decoded, tuple):
        raise CheckpointFormatError("Encoded random state root must be a sequence.")
    return decoded


def _decode_random_state_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_decode_random_state_value(item) for item in value)
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


def encode_evaluated_batch(batch: EvaluatedBatch) -> dict[str, Any]:
    """Serialize objective values without duplicating their raw evaluations."""

    return {
        "batch_index": batch.batch_index,
        "objective_names": list(batch.objective_names),
        "normalization_state": (
            dict(batch.normalization_state.checkpoint_state())
            if batch.normalization_state is not None
            else None
        ),
        "items": [
            {
                "individual_id": item.individual.id,
                "status": item.status.value,
                "error": item.error,
                "objective_values": _encode_objective_values(item.objective_values),
            }
            for item in batch.items
        ],
    }


def decode_evaluated_batch(
    value: Mapping[str, Any],
    *,
    evaluations: Sequence[Evaluation],
    normalizer: Normalizer | None = None,
) -> EvaluatedBatch:
    """Restore objective values and bind them to authoritative evaluations."""

    try:
        names = _string_tuple(value["objective_names"], "objective_names")
        raw_items = value["items"]
        batch_index = value.get("batch_index")
        raw_normalization = value.get("normalization_state")
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError("Invalid evaluated batch payload.") from exc
    if batch_index is not None and (
        isinstance(batch_index, bool) or not isinstance(batch_index, int)
    ):
        raise CheckpointFormatError(
            "Evaluated batch index must be an integer or null."
        )
    if isinstance(raw_items, (str, bytes)) or not isinstance(raw_items, Sequence):
        raise CheckpointFormatError("Evaluated batch items must be a sequence.")
    if len(raw_items) != len(evaluations):
        raise CheckpointFormatError(
            "Evaluated batch item count does not match its evaluations."
        )
    try:
        if raw_normalization is not None and normalizer is None:
            raise ValueError(
                "A configured normalizer is required to restore batch state."
            )
        normalization_state = (
            normalizer.restore_state(raw_normalization)
            if raw_normalization is not None and normalizer is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointFormatError("Invalid batch normalization state.") from exc
    if normalization_state is not None and normalization_state.objective_names != names:
        raise CheckpointFormatError(
            "Batch normalization objective names are inconsistent."
        )

    items: list[EvaluatedIndividual] = []
    for index, (raw_item, evaluation) in enumerate(
        zip(raw_items, evaluations, strict=True)
    ):
        if not isinstance(raw_item, Mapping):
            raise CheckpointFormatError(
                f"Evaluated batch item {index} must be a mapping."
            )
        if raw_item.get("individual_id") != evaluation.individual.id:
            raise CheckpointFormatError(
                f"Evaluated batch item {index} does not match its evaluation."
            )
        try:
            status = ObjectiveEvaluationStatus(raw_item["status"])
            objective_values = _decode_objective_values(
                raw_item.get("objective_values"),
                names=names,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError(
                f"Invalid objective data for evaluated batch item {index}."
            ) from exc
        error = raw_item.get("error")
        if error is not None and not isinstance(error, str):
            raise CheckpointFormatError(
                f"Evaluated batch item {index} error must be text or null."
            )
        if (status is ObjectiveEvaluationStatus.VALID) != (
            objective_values is not None
        ):
            raise CheckpointFormatError(
                f"Evaluated batch item {index} status and values are inconsistent."
            )
        items.append(
            EvaluatedIndividual(
                evaluation=evaluation,
                objective_values=objective_values,
                status=status,
                error=error,
            )
        )
    return EvaluatedBatch(
        items=tuple(items),
        objective_names=names,
        batch_index=batch_index,
        normalization_state=normalization_state,
    )


def _encode_objective_values(value: ObjectiveValues | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "names": list(value.names),
        "raw": list(value.raw),
        "minimize": list(value.minimize),
        "maximize": list(value.maximize),
        "normalized_minimize": (
            list(value.normalized_minimize)
            if value.normalized_minimize is not None
            else None
        ),
        "normalized_maximize": (
            list(value.normalized_maximize)
            if value.normalized_maximize is not None
            else None
        ),
        "aggregate_score": value.aggregate_score,
    }


def _decode_objective_values(
    value: object,
    *,
    names: tuple[str, ...],
) -> ObjectiveValues | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("objective_values must be a mapping or null.")
    value_names = _string_tuple(value.get("names"), "objective value names")
    if value_names != names:
        raise ValueError("Objective value names do not match the batch schema.")
    score = value.get("aggregate_score")
    return ObjectiveValues(
        names=names,
        raw=_float_tuple(value.get("raw"), len(names), "raw"),
        minimize=_float_tuple(value.get("minimize"), len(names), "minimize"),
        maximize=_float_tuple(value.get("maximize"), len(names), "maximize"),
        normalized_minimize=_optional_float_tuple(
            value.get("normalized_minimize"), len(names), "normalized_minimize"
        ),
        normalized_maximize=_optional_float_tuple(
            value.get("normalized_maximize"), len(names), "normalized_maximize"
        ),
        aggregate_score=(
            None if score is None else _finite_float(score, "aggregate_score")
        ),
    )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence.")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{label} must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique.")
    return result


def _float_tuple(value: object, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence.")
    result = tuple(_finite_float(item, label) for item in value)
    if len(result) != length:
        raise ValueError(f"{label} does not match the objective count.")
    return result


def _optional_float_tuple(
    value: object,
    length: int,
    label: str,
) -> tuple[float, ...] | None:
    return None if value is None else _float_tuple(value, length, label)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must contain finite real numbers.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must contain finite real numbers.")
    return result


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
    "decode_evaluated_batch",
    "decode_individual",
    "decode_random_state",
    "encode_evaluation",
    "encode_evaluated_batch",
    "encode_individual",
    "encode_random_state",
    "qualified_name",
    "signature_value",
]
