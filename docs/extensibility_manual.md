# Manual De Extensibilidad De GENIO

Este documento describe como extender GENIO a partir de sus clases abstractas y contratos base. El objetivo es servir como manual de usuario para implementar nuevos dominios de evaluacion, algoritmos de busqueda, backends, artefactos, objetivos, compositores y recolectores de estadisticas.

GENIO separa tres ideas principales:

- El framework define contratos estables.
- Las extensiones concretas implementan logica de dominio.
- Los algoritmos de busqueda consumen `EvaluatedBatch`, no artefactos ni detalles de ejecucion.

## Vista General

El flujo extensible es:

```text
OptimizationSession configura SearchAlgorithm con SearchContext
SearchAlgorithm.ask()
        -> Individual[]
EvaluationWorkflow
        -> EvaluationStep.create_task(...)
        -> EvaluationTask.run(context)
        -> Artifact[] / MetricArtifact[]
EvaluationExecutor
        -> Result.metrics
ObjectiveRuntime interno
        -> EvaluatedBatch
SearchAlgorithm.tell(batch)
```

Las extensiones principales se apoyan en estas clases:

```text
SearchAlgorithm
SearchContext
EvaluationStep
EvaluationTask
Artifact
MetricArtifact
Backend
Composer
Objective
ObjectiveSet
Normalizer
Scalarizer
StatisticsCollector
```

## Principios De Diseño

GENIO espera que cada extension respete estas reglas:

- `SearchAlgorithm` decide que individuos evaluar y como usar los resultados.
- `EvaluationStep` declara un paso logico del workflow y crea una task.
- `EvaluationTask` ejecuta trabajo real usando un `ExecutionContext`.
- `Backend` proporciona infraestructura, no logica de dominio.
- `Artifact` transporta salidas entre steps.
- `MetricArtifact` expone metricas numericas que se agregan en `Result.metrics`.
- `Result` no contiene artefactos; solo estado, metricas y error.
- `ObjectiveSet` configura en la sesion la extraccion, normalizacion y scalarizacion.
- `ObjectiveRuntime` mantiene estado aislado de una sesion y entrega `EvaluatedBatch`.
- Cada algoritmo decide la semantica de `best_individuals()`.
- El analisis historico pertenece a las estadisticas y herramientas posteriores, no a `ObjectiveSet`.
- `Composer` ayuda a traducir individuos a representaciones de dominio.
- `StatisticsCollector` observa eventos de sesion sin modificar el flujo.

## 1. Extender Algoritmos De Busqueda

Clase base:

```python
from genio import SearchAlgorithm
```

Contrato:

```python
class SearchAlgorithm(ABC):
    def configure(self, context: SearchContext) -> None:
        ...

    def ask(self) -> Sequence[Individual]:
        ...

    def tell(self, batch: EvaluatedBatch) -> None:
        ...

    def should_stop(self) -> bool:
        ...

    def best_individuals(self) -> Sequence[Individual]:
        return ()
```

Responsabilidades:

- Validar en `configure(...)` el esquema y las transformaciones requeridas.
- Proponer individuos desde `self.context.search_space`.
- Recibir resultados objetivos completos en `tell(batch)`.
- Mantener estado interno.
- Decidir cuando detenerse.
- Devolver los mejores individuos segun su propio criterio.

No debe:

- Ejecutar tasks directamente.
- Leer artefactos intermedios del backend.
- Conocer detalles de Vitis, OpenCV, datasets o plantillas.

### `SearchContext`

La sesion construye y fija este contexto inmutable antes de iniciar el primer batch:

```python
SearchContext(
    search_space: SearchSpace,
    objective_schema: ObjectiveSchema | None = None,
    has_normalizer: bool = False,
    has_scalarizer: bool = False,
    normalization_scope: str | None = None,
)
```

No expone el `ObjectiveSet` ejecutable, backend, workflow ni sesion completa.
`configure()` es idempotente si recibe un contexto igual y rechaza cambiarlo despues.
Una subclase debe validar sus requisitos antes de llamar a `super().configure(context)`.

### Algoritmos Sin Objetivo

Algunos algoritmos no necesitan score para explorar, por ejemplo grid search o random search. Pueden ignorar `Objective` y simplemente almacenar evaluaciones.

```python
from genio import SearchAlgorithm

class FirstNAlgorithm(SearchAlgorithm):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.next_index = 0
        self.evaluations = []

    def ask(self):
        if self.next_index >= self.limit:
            return []
        individual = self.context.search_space.from_index(self.next_index)
        self.next_index += 1
        return [individual]

    def tell(self, batch):
        self.evaluations.extend(batch.evaluations)

    def should_stop(self):
        return self.next_index >= self.limit
```

### Algoritmos Con Score

Los algoritmos escalares no extraen objetivos por su cuenta. Declaran que requieren
un scalarizer y consumen `ObjectiveValues.aggregate_score` ya calculado:

```python
from genio import SearchAlgorithm

class BestScoreAlgorithm(SearchAlgorithm):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.next_index = 0
        self.scored = []

    def configure(self, context):
        if not context.has_scalarizer:
            raise ValueError("BestScoreAlgorithm requires a scalarizer")
        super().configure(context)

    def ask(self):
        if self.next_index >= self.limit:
            return []
        individual = self.context.search_space.from_index(self.next_index)
        self.next_index += 1
        return [individual]

    def tell(self, batch):
        for item in batch.valid_items:
            score = item.objective_values.aggregate_score
            if score is None:
                raise ValueError("Missing aggregate score")
            self.scored.append((score, item.individual))

    def should_stop(self):
        return self.next_index >= self.limit

    def best_individuals(self):
        if not self.scored:
            return ()
        return (max(self.scored, key=lambda entry: entry[0])[1],)
```

### Algoritmos Multiobjetivo

Los algoritmos multiobjetivo validan `context.objective_schema` y consumen matrices
orientadas de `EvaluatedBatch`. Por ejemplo, un adaptador que usa convencion de
minimizacion puede obtener:

```python
class ExternalMultiObjectiveAdapter(SearchAlgorithm):
    def configure(self, context):
        if context.objective_schema is None or len(context.objective_schema) < 2:
            raise ValueError("At least two objectives are required")
        super().configure(context)

    def tell(self, batch):
        valid_items = batch.valid_items
        objective_rows = batch.minimization_matrix()
        self.external_optimizer.tell(
            individuals=[item.individual for item in valid_items],
            objectives=objective_rows,
        )
```

La alineacion de fallos debe ser explicita si una libreria externa exige una fila por
propuesta. No se deben interpretar placeholders numericos como objetivos validos.

### Frontera De Fallos

- Un fallo esperado de una task se convierte en `Result.failed`; los demas individuos del batch continuan.
- El runtime lo entrega como `EVALUATION_FAILED`, sin usar metricas parciales como objetivos.
- Un `Result.success` con objetivos ausentes, booleanos, no numericos o no finitos se entrega como `INVALID_OBJECTIVES`.
- Cada algoritmo decide como tratar items invalidos. Genetic les asigna fitness cero y NSGA-II los marca como no factibles.
- Un error de contrato, normalizacion, scalarizacion, `tell()` o hook despues de `ask()` deja la sesion fallida y no reanudable. Las llamadas posteriores a `run()` o `save_checkpoint()` fallan claramente.
- Un `ask()` vacio es valido solo si el algoritmo confirma `should_stop()`; vacio sin parada es un error de contrato.

## 2. Extender Objetivos De Optimizacion

Clases base:

```python
from genio import (
    Objective,
    MetricObjective,
    ObjectiveSet,
    ObjectiveError,
    MinMaxNormalizer,
    NormalizationScope,
    OptimizationDirection,
    WeightedMeanScalarizer,
)
```

### `OptimizationDirection`

Define la direccion de mejora:

```python
OptimizationDirection.MAXIMIZE
OptimizationDirection.MINIMIZE
```

### `Objective`

Contrato para interpretar un valor numerico desde una `Evaluation`.

```python
class Objective(ABC):
    @property
    def name(self) -> str:
        ...

    @property
    def direction(self) -> OptimizationDirection:
        ...

    @property
    def normalization_bounds(self) -> tuple[float, float] | None:
        return None

    def value(self, evaluation: Evaluation) -> float:
        ...

    def score(self, evaluation: Evaluation) -> float:
        ...

    def checkpoint_signature(self) -> Mapping[str, Any]:
        ...
```

`score(...)` convierte el objetivo a una convencion uniforme: mayor score es mejor. Si la direccion es `MINIMIZE`, devuelve `-value`.

### `MetricObjective`

Usa una clave de `Result.metrics`.

```python
from genio import MetricObjective, OptimizationDirection

objective = MetricObjective(
    metric="functional.f1",
    direction=OptimizationDirection.MAXIMIZE,
)
```

Ejemplo de minimizacion:

```python
latency = MetricObjective(
    metric="hls.latency",
    direction=OptimizationDirection.MINIMIZE,
    name="latency",
    normalization_bounds=(0.0, 1_000_000.0),
)
```

La firma final es:

```python
MetricObjective(
    metric,
    direction,
    *,
    name=None,
    normalization_bounds=None,
)
```

`direction` acepta el enum o `"maximize"`/`"minimize"`. `name` usa `metric` por
defecto. Los bounds solo configuran normalizacion y deben ser finitos con minimo menor
que maximo.

### Crear Un Objetivo Personalizado

Un objetivo personalizado puede combinar varias metricas.

```python
from dataclasses import dataclass
from genio import Objective, OptimizationDirection, ObjectiveError

@dataclass(frozen=True, slots=True)
class WeightedQualityObjective(Objective):
    f1_metric: str
    latency_metric: str

    @property
    def name(self):
        return "weighted_quality"

    @property
    def direction(self):
        return OptimizationDirection.MAXIMIZE

    @property
    def normalization_bounds(self):
        return (-1_000.0, 1.0)

    def value(self, evaluation):
        metrics = evaluation.result.metrics
        try:
            f1 = metrics[self.f1_metric]
            latency = metrics[self.latency_metric]
        except KeyError as exc:
            raise ObjectiveError("Missing metric for weighted quality") from exc
        return f1 - 0.001 * latency

    def checkpoint_signature(self):
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "f1_metric": self.f1_metric,
            "latency_metric": self.latency_metric,
            "normalization_bounds": list(self.normalization_bounds),
        }
```

La firma es obligatoria para una extension: debe incluir todos los campos que cambian
la extraccion o transformacion. No basta el nombre de la clase.

### `ObjectiveSet`

Agrupa la configuracion de uno o varios objetivos para la sesion.

```python
from genio import ObjectiveSet, MetricObjective, OptimizationDirection

objectives = ObjectiveSet((
    MetricObjective(
        metric="functional.f1",
        direction=OptimizationDirection.MAXIMIZE,
    ),
    MetricObjective(
        metric="hls.latency",
        direction=OptimizationDirection.MINIMIZE,
    ),
    MetricObjective(
        metric="hls.lut",
        direction=OptimizationDirection.MINIMIZE,
    ),
))
```

Los defaults son deliberadamente neutros:

```python
ObjectiveSet(
    objectives: Sequence[Objective],
    normalizer: Normalizer | None = None,
    scalarizer: Scalarizer | None = None,
)
```

Sin normalizer no existen vectores normalizados; sin scalarizer no existe
`aggregate_score`. `ObjectiveSet` valida nombres, orden, esquema y estrategias, pero no
guarda historia, no calcula un Pareto historico y no decide `best_individuals()`.

### Normalizacion Min-Max

```python
MinMaxNormalizer(scope=NormalizationScope.BATCH)
MinMaxNormalizer(scope=NormalizationScope.CUMULATIVE)
MinMaxNormalizer(scope=NormalizationScope.FIXED)
```

| Scope | Uso | Comparabilidad |
| --- | --- | --- |
| `BATCH` | Ajusta columnas sin bounds al batch valido actual. | Solo dentro del mismo batch. |
| `CUMULATIVE` | Amplia minimos y maximos con cada batch. | Los scores de versiones distintas no son directamente comparables. |
| `FIXED` | Usa exclusivamente `normalization_bounds`. | Comparable entre batches y sesiones con la misma firma. |

`FIXED` requiere bounds para todos los objetivos. La transformacion no hace clipping;
un valor fuera de rango puede quedar fuera de `[0, 1]`. Una columna constante vale
cero porque no discrimina candidatos.

### Scalarizacion

```python
WeightedMeanScalarizer(weights=None)  # media uniforme explicita
WeightedMeanScalarizer({"quality": 0.7, "latency": 0.3})
```

Los pesos opcionales se identifican por `Objective.name`, deben cubrir exactamente el
esquema, ser finitos y no negativos, y contener al menos uno positivo. Se normalizan
para sumar uno. El scalarizer recibe valores orientados a maximizacion: normalizados
si hay normalizer, y brutos en caso contrario.

### Runtime Y Resultados Objetivos

`OptimizationSession` crea internamente un `ObjectiveRuntime` mediante
`ObjectiveSet.bind()`. Una instancia pertenece a una sola sesion y mantiene el estado
de normalizacion sin contaminar otras ejecuciones que reutilicen el mismo
`ObjectiveSet`.

```python
ObjectiveValues(
    names,
    raw,
    minimize,
    maximize,
    normalized_minimize=None,
    normalized_maximize=None,
    aggregate_score=None,
)

EvaluatedIndividual(
    evaluation,
    objective_values,
    status,
    error=None,
)

EvaluatedBatch(
    items,
    objective_names,
    batch_index=None,
    normalization_state=None,
)
```

`EvaluatedIndividual.individual` es una propiedad derivada de `evaluation`, no un
campo duplicado. `EvaluatedBatch` implementa `Sequence[EvaluatedIndividual]` y expone
`evaluations`, `individuals`, `valid_items`, `valid_indices`, mascaras, matrices y
scores. Las matrices normales incluyen solo items validos, sin ceros implicitos.

Los estados son `VALID`, `EVALUATION_FAILED`, `INVALID_OBJECTIVES` y `NOT_CONFIGURED`.
Una evaluacion fallida no usa metricas parciales. Un resultado exitoso con una metrica
ausente, booleana, no numerica o no finita queda como `INVALID_OBJECTIVES`. La
normalizacion y scalarizacion se ajustan solo con filas validas. El runtime prepara
todo el batch y publica el nuevo estado de forma transaccional.

### Extender Normalizers Y Scalarizers

Un `Normalizer` implementa `validate(schema)`, `fit(values, *, schema, previous)`,
`transform(values, state)`, `restore_state(data)` y `checkpoint_signature()`. Su estado
concreto implementa `NormalizationState.checkpoint_state()`.

Un `Scalarizer` implementa `validate(schema)`, `scalarize(objective_names, values)` y
`checkpoint_signature()`. Ambas firmas deben ser completas, deterministas y
serializables como JSON; las clases base rechazan una firma implicita.

## 3. Extender Artefactos

Clases base:

```python
from genio import Artifact, MetricArtifact, ArtifactError
```

### `Artifact`

Representa una salida producida por una task y consumible por steps posteriores.

Campos:

```python
name: str
producer: str
individual_id: str
objective: str | None
metadata: dict[str, Any]
```

Metodo obligatorio:

```python
load() -> Sequence[Any]
```

Ejemplo:

```python
from dataclasses import dataclass
from pathlib import Path
from genio import Artifact

@dataclass(frozen=True, slots=True)
class FileArtifact(Artifact):
    path: Path

    def load(self):
        return [self.path.read_text()]
```

### `MetricArtifact`

Subclase de `Artifact` para artefactos que exponen metricas numericas.

Metodo obligatorio adicional:

```python
metrics() -> Mapping[str, float]
```

Ejemplo:

```python
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any
from genio import MetricArtifact

@dataclass(frozen=True, slots=True)
class ReportMetrics(MetricArtifact):
    values: Mapping[str, float]

    def load(self) -> Sequence[Any]:
        return [dict(self.values)]

    def metrics(self) -> Mapping[str, float]:
        return self.values
```

Si este artefacto lo devuelve un step con id `hls`, las metricas se agregan como:

```python
{
    "hls.latency": 120.0,
    "hls.lut": 3021.0,
}
```

Reglas importantes:

- `metrics()` debe devolver valores numericos.
- Los booleanos no se aceptan como metricas.
- Las claves duplicadas provocan `EvaluationExecutionError`.
- Los artefactos no metricos no aparecen en `Result`.
- Los artefactos no metricos solo sirven para steps posteriores o persistencia externa futura.

## 4. Extender Tareas Ejecutables

Clase base:

```python
from genio import EvaluationTask, ExecutionContext
```

Contrato:

```python
@dataclass(frozen=True, slots=True)
class EvaluationTask(ABC):
    individual: Individual
    id: str | None = None
    step_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        ...

    def run(self, context: ExecutionContext) -> list[Artifact]:
        ...
```

Responsabilidades:

- Ejecutar una unidad concreta de trabajo.
- Usar `ExecutionContext` para filesystem, logs, comandos y rutas.
- Crear y devolver artefactos.
- Encapsular logica de dominio.

No debe:

- Modificar directamente el estado del algoritmo.
- Decidir objetivos de optimizacion.
- Depender de un backend concreto si puede evitarse.

### Ejemplo De Task

```python
from genio import EvaluationTask, ExecutionContext, Artifact

class FunctionalEvaluationTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        task_dir = context.ensure_dir(context.task_dir(self))
        report_path = context.write_json(task_dir / "report.json", {"f1": 0.91})
        return [
            ReportMetrics(
                name="metrics",
                producer=self.step_id or "functional",
                individual_id=self.individual.id,
                values={"f1": 0.91},
            )
        ]
```

## 5. Usar `ExecutionContext`

`ExecutionContext` lo crea el backend y se entrega a cada task.

Campos:

```python
base_work_dir: Path
run_id: str | None
backend_id: str | None
metadata: Mapping[str, Any]
```

Helpers disponibles:

```python
resolve_path(path)
task_dir(task, *parts)
artifact_path(task, *parts)
log_path(task, *parts)
ensure_dir(path)
ensure_parent(path)
write_text(path, content)
read_text(path)
write_bytes(path, content)
read_bytes(path)
write_json(path, data)
read_json(path)
copy_file(source, target)
copy_tree(source, target)
write_log(task, name, content)
merged_env(env=None)
run_command(command, cwd=None, env=None, timeout=None, check=True)
```

Ejemplo:

```python
class CommandTask(EvaluationTask):
    def run(self, context):
        workdir = context.ensure_dir(context.task_dir(self))
        result = context.run_command(
            ["python", "-c", "print('ok')"],
            cwd=workdir,
        )
        context.write_log(self, "stdout.log", result.stdout)
        return []
```

Regla de diseño:

- `ExecutionContext` debe seguir siendo generico.
- No se deben anadir helpers especificos como `run_vitis_hls()` o `load_dataset()`.
- Esos detalles pertenecen a tasks, composers o configs de dominio.

## 6. Extender Steps De Evaluacion

Clase base:

```python
from genio import EvaluationStep
```

Contrato:

```python
class EvaluationStep(ABC):
    id: str
    depends_on: tuple[str, ...] = ()
    required_artifacts: Mapping[str, type[Artifact]] = {}
    produced_artifacts: Mapping[str, type[Artifact]] = {}
    task_type: type[EvaluationTask] = EvaluationTask

    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        ...
```

Responsabilidades:

- Declarar el id logico del paso.
- Declarar dependencias con `depends_on`.
- Declarar artefactos consumidos con claves cualificadas en `required_artifacts`.
- Declarar todos los artefactos que una task exitosa puede devolver mediante
  nombres locales en `produced_artifacts`.
- Declarar el tipo de task que produce mediante `task_type`.
- Crear una task concreta para un individuo.
- Usar artefactos acumulados de pasos anteriores si el step depende de ellos.

Ejemplo:

```python
from genio import EvaluationStep, Individual

class FunctionalStep(EvaluationStep):
    id = "functional"
    task_type = FunctionalEvaluationTask
    produced_artifacts = {"metrics": FunctionalMetricsArtifact}

    def create_task(self, individual: Individual, artifacts):
        return FunctionalEvaluationTask(individual=individual, step_id=self.id)
```

Ejemplo con dependencia:

```python
class HlsStep(EvaluationStep):
    id = "hls"
    depends_on = ("compose",)
    task_type = HlsEvaluationTask
    required_artifacts = {"compose.project": ProjectArtifact}
    produced_artifacts = {"rtl": RTLArtifact}

    def create_task(self, individual, artifacts):
        project = artifacts["compose.project"]
        return HlsEvaluationTask(
            individual=individual,
            step_id=self.id,
            metadata={"project_path": str(project.load()[0])},
        )
```

Reglas:

- `create_task(...)` debe devolver una instancia de `task_type`.
- `id` debe ser unico dentro del workflow.
- `depends_on` debe referenciar steps existentes.
- Los nombres de `produced_artifacts` son locales, no vacios y no contienen `.`.
- Cada requisito debe estar declarado por su productor y su tipo debe ser compatible.
- Una task exitosa no puede devolver artifacts no declarados o de otro individuo.

## 7. Declarar Workflows De Evaluacion

Clase:

```python
from genio import EvaluationWorkflow
```

Uso:

```python
workflow = EvaluationWorkflow((
    ComposeStep(),
    FunctionalStep(),
    HlsStep(),
))
```

`EvaluationWorkflow` valida:

- ids duplicados;
- dependencias inexistentes;
- ciclos.

El orden de ejecucion se obtiene con:

```python
workflow.execution_order()
```

Los artefactos se acumulan con claves:

```text
step_id.artifact_name
```

Las metricas se acumulan con claves:

```text
step_id.metric_name
```

## 8. Extender Backends

Clase base:

```python
from genio import Backend, EvaluationHandle, EvaluationState
```

Contrato:

```python
class Backend(ABC):
    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        ...

    def submit_batch(self, tasks: Sequence[EvaluationTask]) -> list[EvaluationHandle]:
        ...

    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        ...

    def status(self, handle: EvaluationHandle) -> EvaluationState:
        ...

    def error(self, handle: EvaluationHandle) -> str | None:
        return None

    def cancel(self, handle: EvaluationHandle) -> None:
        ...
```

Responsabilidades:

- Recibir `EvaluationTask`.
- Crear contexto de ejecucion.
- Ejecutar o enviar la task.
- Devolver `EvaluationHandle`.
- Permitir recoger artefactos.
- Exponer estado, error y cancelacion.

No debe:

- Saber que es Vitis, OpenCV, Git, un dataset o una plantilla.
- Crear tasks.
- Interpretar metricas.
- Decidir objetivos de optimizacion.

### `EvaluationHandle`

Campos:

```python
id: str
task_id: str | None
backend_id: str | None
metadata: Mapping[str, Any]
payload: Any
```

### `EvaluationState`

Estados:

```python
PENDING
RUNNING
DONE
FAILED
CANCELLED
```

### Cuándo Crear Un Backend Nuevo

Crear un backend nuevo si se necesita:

- ejecucion remota;
- colas de trabajos;
- paralelismo real;
- integracion con Slurm, Kubernetes, Ray, Celery u otro scheduler;
- persistencia externa de artefactos;
- cancelacion asincrona.

Para ejecucion local sincrona ya existe `LocalBackend`.

## 9. Extender Composers

Clase base:

```python
from genio import Composer
```

Contrato principal:

```python
class Composer(ABC):
    def compose(self, individual: Individual) -> Any:
        ...
```

Helpers disponibles:

```python
active_choices(individual)
active_stage_definitions(individual)
should_skip(choice)
stage_definition(stage)
artifact_metadata(individual)
```

Responsabilidades:

- Traducir un `Individual` a una representacion de dominio.
- Reutilizar definiciones de stages.
- Saltar etapas `nop` si aplica.
- Consumir solo los dominios de `individual.design` que correspondan a su backend, por ejemplo `hls` para síntesis o `system` para integración de plataforma.
- Preparar informacion que una task puede materializar.

No debe:

- Ejecutar herramientas externas por si mismo si eso requiere runtime.
- Depender del backend.
- Actualizar resultados de busqueda.

Ejemplo:

```python
from genio import Composer

class PipelineComposer(Composer):
    def compose(self, individual):
        return [
            {
                "stage": choice.stage,
                "parameters": choice.parameters,
            }
            for choice in self.active_choices(individual)
        ]
```

Uso recomendado dentro de una task:

```python
class ComposeTask(EvaluationTask):
    def run(self, context):
        composer = PipelineComposer("search_space/stages/definitions")
        pipeline = composer.compose(self.individual)
        path = context.write_json(context.artifact_path(self, "pipeline.json"), pipeline)
        return [PipelineArtifact("pipeline", path, self.individual.id)]
```

## 10. Extender Estadisticas

Clase base:

```python
from genio import StatisticsCollector
```

Contrato:

```python
class StatisticsCollector(ABC):
    def on_session_started(self, session: OptimizationSession) -> None:
        pass

    def on_batch_started(self, batch_index: int, individuals: Sequence[Individual]) -> None:
        pass

    def on_proposals_generated(self, proposals: Sequence[Proposal]) -> None:
        pass

    def on_evaluation_completed(self, evaluation: Evaluation) -> None:
        pass

    def on_evaluated_batch(self, batch: EvaluatedBatch) -> None:
        pass

    def on_batch_completed(self, batch_index: int, evaluations: Sequence[Evaluation]) -> None:
        pass

    def on_session_completed(self, result: SearchResult) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {}
```

Responsabilidades:

- Observar eventos de sesion.
- Observar batches completos, que pueden representar generaciones en algoritmos evolutivos.
- Acumular metricas crudas, estados objetivos, vectores orientados/normalizados y scores.
- Devolver un snapshot serializable.

`OptimizationSession.run()` anota `batch_index` en `Evaluation.metadata`, por lo que cada evaluacion del resultado final puede asociarse al batch en el que fue producida.

Para un batch normal, la sesion emite propuestas, evaluaciones individuales y, tras
aceptar `tell(batch)` y comprometer su historia, `on_evaluated_batch(batch)` y
`on_batch_completed(...)`. Un collector que no pueda persistir datos debe propagar el
error; la sesion quedara en estado fallido terminal.

Ejemplo:

```python
from genio import StatisticsCollector

class MetricHistory(StatisticsCollector):
    def __init__(self) -> None:
        self.history = []
        self.batches = []
        self.objective_batches = []

    def on_batch_completed(self, batch_index, evaluations):
        self.batches.append({
            "batch_index": batch_index,
            "individuals": [
                {
                    "id": evaluation.individual.id,
                    "metrics": dict(evaluation.result.metrics),
                }
                for evaluation in evaluations
            ],
        })

    def on_evaluation_completed(self, evaluation):
        self.history.append(dict(evaluation.result.metrics))

    def on_evaluated_batch(self, batch):
        self.objective_batches.append({
            "batch_index": batch.batch_index,
            "items": [
                {
                    "id": item.individual.id,
                    "status": item.status.value,
                    "error": item.error,
                    "raw": (
                        list(item.objective_values.raw)
                        if item.objective_values is not None
                        else None
                    ),
                    "aggregate_score": (
                        item.objective_values.aggregate_score
                        if item.objective_values is not None
                        else None
                    ),
                }
                for item in batch
            ],
        })

    def snapshot(self):
        return {
            "metrics": self.history,
            "batches": self.batches,
            "objective_batches": self.objective_batches,
        }
```

`CSVStatisticsCollector` implementa este hook y añade a cada fila `objective_status`,
`objective_error`, `normalization_version`, `aggregate_score` y columnas
`objective.<name>.*` para los valores raw, orientados y normalizados. El manifest
incluye la firma del conjunto objetivo y el summary agrega estadisticas objetivas.
Este CSV/JSON es el lugar previsto para analisis historico a posteriori.

## 11. Montar Una Sesion Completa

Una sesion conecta espacio de busqueda, algoritmo, backend, workflow y la configuracion
objetiva opcional. Este ejemplo usa bounds fijos y scalarizer explicito para mantener
scores comparables durante varias generaciones:

```python
from random import Random

from genio import (
    GeneticSearch,
    MetricObjective,
    MinMaxNormalizer,
    NSGA2Search,
    NormalizationScope,
    ObjectiveSet,
    OptimizationDirection,
    OptimizationSession,
    WeightedMeanScalarizer,
)

objective_set = ObjectiveSet(
    objectives=(
        MetricObjective(
            "python_image_functional.mask_f1",
            OptimizationDirection.MAXIMIZE,
            name="quality",
            normalization_bounds=(0.0, 1.0),
        ),
        MetricObjective(
            "hls_image_pipeline_synthesis.hls_synthesis.lut",
            OptimizationDirection.MINIMIZE,
            name="lut",
            normalization_bounds=(0.0, 100_000.0),
        ),
        MetricObjective(
            "xheep_verilator_simulation.xheep_verilator.application_cycles",
            OptimizationDirection.MINIMIZE,
            name="cycles",
            normalization_bounds=(0.0, 10_000_000.0),
        ),
    ),
    normalizer=MinMaxNormalizer(NormalizationScope.FIXED),
    scalarizer=WeightedMeanScalarizer(
        {"quality": 0.5, "lut": 0.2, "cycles": 0.3}
    ),
)

genetic_result = OptimizationSession(
    search_space=search_space,
    algorithm=GeneticSearch(
        population_size=32,
        max_generations=12,
        random=Random(7),
    ),
    backend=genetic_backend,
    evaluation_workflow=workflow,
    objective_set=objective_set,
).run()

nsga2_result = OptimizationSession(
    search_space=search_space,
    algorithm=NSGA2Search(
        population_size=32,
        max_generations=12,
        seed=7,
    ),
    backend=nsga2_backend,
    evaluation_workflow=workflow,
    objective_set=objective_set,
).run()
```

Genetic exige el scalarizer y usa su score. Para varias generaciones no acepta scopes
`BATCH` ni `CUMULATIVE`, ya que no producen scores estacionarios; con normalizacion se
debe usar `FIXED`. NSGA-II usa el vector orientado a minimizacion y no necesita el
scalarizer para evolucionar, aunque en este ejemplo se conserva para estadisticas
comparables. Genetic decide su mejor fitness y NSGA-II conserva el frente final nativo
de `pymoo` en sus respectivos `best_individuals`.

## 12. Extension Por Tipo De Necesidad

### Necesito Una Nueva Metrica

Implementar o modificar una `EvaluationTask` para devolver un `MetricArtifact`.

```text
EvaluationTask.run -> MetricArtifact.metrics -> Result.metrics
```

### Necesito Un Nuevo Criterio De Optimizacion

Usar `MetricObjective` si basta una metrica. Crear una subclase de `Objective` si hay combinacion o penalizacion, y configurar la normalizacion separadamente.

```text
Result.metrics -> ObjectiveRuntime -> EvaluatedBatch -> SearchAlgorithm
```

### Necesito Un Nuevo Algoritmo

Crear una subclase de `SearchAlgorithm`, validar sus necesidades en `configure(context)` y consumir `EvaluatedBatch`. El `ObjectiveSet` se configura en `OptimizationSession`, no se inyecta en el algoritmo.

### Necesito Un Nuevo Paso De Evaluacion

Crear una subclase de `EvaluationStep` y una subclase de `EvaluationTask` asociada.

```text
EvaluationStep.create_task -> EvaluationTask.run
```

El framework incluye como referencia un workflow de tres fases:

```text
PythonImageFunctionalEvaluationStep
    -> HLSImagePipelineSynthesisEvaluationStep
    -> XHeepVerilatorSimulationEvaluationStep
```

El último step demuestra cómo consumir un artefacto RTL, materializar un checkout
externo aislado, preservar symlinks, ejecutar comandos dentro de Conda y convertir
líneas `GENIO_METRIC:nombre:valor` en métricas. Véase
[Evaluación de pipelines HLS en X-HEEP con SAFA](XHEEP_SAFA_EVALUATION.md).
El compositor selecciona exclusivamente mediante `application_name` los flujos
`genio_trans_mem_mem`, `genio_trans_mem_flash` o `genio_trans_flash_mem`.

### Necesito Un Nuevo Backend

Crear una subclase de `Backend` si la ejecucion no cabe en `LocalBackend`.

### Necesito Generar Codigo O Configuracion

Crear una subclase de `Composer` y usarla dentro de una `EvaluationTask`.

### Necesito Guardar Estadisticas

Crear una subclase de `StatisticsCollector`.

## 13. Reglas De Integracion

- Los objetivos proceden siempre de `Result.metrics`, pero el algoritmo los recibe ya transformados en `EvaluatedBatch`.
- Los nombres de metricas agregadas siguen la forma `step_id.metric_name`.
- Los nombres de artefactos acumulados siguen la forma `step_id.artifact_name`.
- Los artefactos no metricos no se devuelven en `Result`.
- `ObjectiveSet` es opcional en `OptimizationSession`; sus estrategias tienen defaults `None`.
- Los algoritmos declaran sus requisitos mediante `SearchContext` y deciden su propio `best_individuals()`.
- `ObjectiveSet` no almacena historia ni selecciona resultados globales.
- La ponderacion de objetivos pertenece a `WeightedMeanScalarizer`.
- Para comparar scores entre generaciones se necesita una escala estable, normalmente `MinMaxNormalizer(FIXED)` con bounds completos.
- `batch_index` en `Evaluation.metadata` permite reconstruir poblaciones o generaciones evaluadas.
- El backend no debe contener logica de dominio.
- Las tasks son el lugar correcto para ejecutar herramientas, materializar archivos y producir artefactos.

### Checkpoints De Extensiones

Toda extension que participe en la firma de una sesion debe declarar de forma
determinista y serializable como JSON la configuracion que afecta a su comportamiento:

- `Objective`, `Normalizer` y `Scalarizer` deben implementar siempre `checkpoint_signature()`; sus bases rechazan la firma implicita.
- `SearchAlgorithm`, `StatisticsCollector`, `EvaluationStep`, `Composer` y `Backend` deben incluir todos sus parametros relevantes en `checkpoint_signature()`.
- La configuracion de una `EvaluationTask` queda cubierta por el `EvaluationStep` que la construye.
- Un algoritmo con `supports_checkpointing=True` implementa `checkpoint_state()` y `restore_checkpoint_state(state, *, search_space, evaluated_batches)`.
- Un normalizer con estado implementa `NormalizationState.checkpoint_state()` y `restore_state(data)`.
- El estado mutable no debe duplicar la historia: los `EvaluatedBatch` de la sesion son la fuente autoritativa para restaurar algoritmos.
- `OptimizationSession` es el unico lector y escritor del formato de checkpoint; no existen schemas versionados ni formatos por algoritmo.

Cambiar una metrica, bound, peso, template, dataset, backend o parametro algorítmico
debe cambiar la firma y hacer incompatible la reanudacion.

## 14. Mapa Rapido De Clases Abstractas

| Clase | Se extiende para | Metodo clave |
| --- | --- | --- |
| `SearchAlgorithm` | Nuevas estrategias de busqueda | `configure`, `ask`, `tell`, `should_stop` |
| `Objective` | Nuevos criterios escalares o compuestos | `value`, `checkpoint_signature` |
| `Normalizer` | Nuevas transformaciones vectoriales con estado propio | `validate`, `fit`, `transform`, `restore_state` |
| `Scalarizer` | Nuevas agregaciones a score | `validate`, `scalarize` |
| `Artifact` | Nuevas salidas consumibles por steps | `load` |
| `MetricArtifact` | Nuevas salidas metricas | `metrics` |
| `EvaluationTask` | Nuevas unidades ejecutables | `run` |
| `EvaluationStep` | Nuevos pasos de workflow | `create_task` |
| `Backend` | Nuevos mecanismos de ejecucion | `submit`, `collect`, `status` |
| `Composer` | Nuevas traducciones de individuos | `compose` |
| `StatisticsCollector` | Nuevos recolectores de eventos | hooks de sesion |

## 15. Checklist Para Una Extension De Dominio

1. Definir que artefactos produce cada fase.
2. Implementar `Artifact` o `MetricArtifact` para esas salidas.
3. Implementar una `EvaluationTask` por unidad ejecutable.
4. Implementar un `EvaluationStep` por paso logico.
5. Construir un `EvaluationWorkflow` con dependencias explicitas.
6. Elegir `LocalBackend` o implementar un `Backend` propio.
7. Implementar o configurar un `SearchAlgorithm` con `configure`, `ask()` y `tell(batch)`.
8. Definir `ObjectiveSet` en la sesion si el algoritmo necesita comparar resultados.
9. Elegir explicitamente normalizer y scalarizer; usar bounds fijos si se comparan generaciones.
10. Implementar firmas completas de checkpoint para todas las extensiones configurables.
11. Crear un `StatisticsCollector` si se necesitan historicos o trazas.
12. Ejecutar la sesion con `OptimizationSession`.
