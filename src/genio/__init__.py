"""Public API for building and running GENIO optimization searches.

GENIO separates candidate generation, evaluation, execution, and observation.
A typical application creates a :class:`SearchSpace`, selects a
:class:`SearchAlgorithm`, combines one or more :class:`EvaluationStep` objects
in an :class:`EvaluationWorkflow`, and passes those components to an
:class:`OptimizationSession`.

The names re-exported here form the supported convenience API. Subpackages can
still be imported directly when implementing custom algorithms, evaluators,
 composers, artifacts, or statistics collectors.
"""

from genio.algorithm import (
    GeneticSearch,
    GridSearch,
    NSGA2Search,
    RandomSearch,
    SearchAlgorithm,
    SearchContext,
)
from genio.evaluation.artifact import Artifact
from genio.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointFormatError,
    CheckpointNotSupportedError,
    CheckpointPolicy,
    CheckpointStateError,
    JSONCheckpointStore,
)
from genio.composer import Composer, ComposerError, ExecutionPackage, StageDefinitionNotFoundError
from genio.core import (
    Evaluation,
    Individual,
    Proposal,
    Result,
    ResultStatus,
    SearchResult,
    StageChoice,
)
from genio.evaluation import (
    ArtifactInput,
    EvaluationExecutionError,
    EvaluationExecutor,
    EvaluationOutput,
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    EvaluationWorkflowError,
    TaskContext,
)
from genio.objective import (
    IdentityScalarizer,
    MetricObjective,
    MinMaxNormalizer,
    NormalizationScope,
    Objective,
    ObjectiveError,
    ObjectiveSet,
    OptimizationDirection,
    WeightedMeanScalarizer,
)
from genio.search_space import DesignDimensionSpec, SearchScenarioSpec, SearchSpace, SlotSpec
from genio.session import OptimizationSession
from genio.statistics import (
    CSVStatisticsCollector,
    CompositeStatisticsCollector,
    InMemoryStatistics,
    PopulationAnalysisCollector,
    PopulationAnalysisError,
    PopulationPlotConfig,
    PopulationStatisticsCollector,
    StatisticsCollector,
)

__all__ = [
    "Artifact",
    "ArtifactInput",
    "Composer",
    "ComposerError",
    "CompositeStatisticsCollector",
    "CSVStatisticsCollector",
    "DesignDimensionSpec",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointFormatError",
    "CheckpointNotSupportedError",
    "CheckpointPolicy",
    "CheckpointStateError",
    "Evaluation",
    "EvaluationExecutionError",
    "EvaluationExecutor",
    "EvaluationOutput",
    "EvaluationStep",
    "EvaluationTask",
    "EvaluationWorkflow",
    "EvaluationWorkflowError",
    "ExecutionPackage",
    "TaskContext",
    "GeneticSearch",
    "GridSearch",
    "IdentityScalarizer",
    "InMemoryStatistics",
    "Individual",
    "JSONCheckpointStore",
    "MetricObjective",
    "MinMaxNormalizer",
    "NSGA2Search",
    "NormalizationScope",
    "Objective",
    "ObjectiveError",
    "ObjectiveSet",
    "OptimizationSession",
    "OptimizationDirection",
    "PopulationAnalysisCollector",
    "PopulationAnalysisError",
    "PopulationPlotConfig",
    "PopulationStatisticsCollector",
    "Proposal",
    "Result",
    "ResultStatus",
    "RandomSearch",
    "SearchAlgorithm",
    "SearchContext",
    "SearchScenarioSpec",
    "SearchSpace",
    "SearchResult",
    "SlotSpec",
    "StatisticsCollector",
    "StageDefinitionNotFoundError",
    "StageChoice",
    "WeightedMeanScalarizer",
]
