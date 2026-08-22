# Manual De Extensibilidad De GENIO

Este documento describe como extender GENIO a partir de sus clases abstractas y contratos base. El objetivo es servir como manual de usuario para implementar nuevos dominios de evaluacion, algoritmos de busqueda, backends, artefactos, objetivos, compositores y recolectores de estadisticas.

GENIO separa tres ideas principales:

- El framework define contratos estables.
- Las extensiones concretas implementan logica de dominio.
- Los algoritmos de busqueda consumen metricas normalizadas, no artefactos ni detalles de ejecucion.

## Vista General

El flujo extensible es:

```text
SearchAlgorithm.ask(session)
        -> Individual[]
EvaluationWorkflow
        -> EvaluationStep.create_task(...)
        -> EvaluationTask.run(context)
        -> Artifact[] / MetricArtifact[]
EvaluationExecutor
        -> Result.metrics
SearchAlgorithm.tell(evaluations)
        -> usa Objective u ObjectiveSet si lo necesita
```

Las extensiones principales se apoyan en estas clases:

```text
SearchAlgorithm
EvaluationStep
EvaluationTask
Artifact
MetricArtifact
Backend
Composer
Objective
ObjectiveSet
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
- `Objective` interpreta metricas y direccion de optimizacion.
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
    def ask(self, session: OptimizationSession) -> Sequence[Individual]:
        ...

    def tell(self, evaluations: Sequence[Evaluation]) -> None:
        ...

    def should_stop(self) -> bool:
        ...

    def best_individuals(self) -> Sequence[Individual]:
        return ()
```

Responsabilidades:

- Proponer individuos desde `session.search_space`.
- Recibir evaluaciones completas en `tell(...)`.
- Mantener estado interno.
- Decidir cuando detenerse.
- Devolver mejores individuos si tiene criterio para ello.

No debe:

- Ejecutar tasks directamente.
- Leer artefactos intermedios del backend.
- Conocer detalles de Vitis, OpenCV, datasets o plantillas.

### Algoritmos Sin Objetivo

Algunos algoritmos no necesitan score para explorar, por ejemplo grid search o random search. Pueden ignorar `Objective` y simplemente almacenar evaluaciones.

```python
from genio import SearchAlgorithm

class FirstNAlgorithm(SearchAlgorithm):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.next_index = 0
        self.evaluations = []

    def ask(self, session):
        if self.next_index >= self.limit:
            return []
        individual = session.search_space.from_index(self.next_index)
        self.next_index += 1
        return [individual]

    def tell(self, evaluations):
        self.evaluations.extend(evaluations)

    def should_stop(self):
        return self.next_index >= self.limit
```

### Algoritmos Monoobjetivo

Los algoritmos monoobjetivo pueden recibir un `Objective` en su constructor y usar `objective.score(evaluation)`.

```python
from genio import SearchAlgorithm, Objective

class BestScoreAlgorithm(SearchAlgorithm):
    def __init__(self, objective: Objective, limit: int) -> None:
        self.objective = objective
        self.limit = limit
        self.next_index = 0
        self.evaluations = []

    def ask(self, session):
        if self.next_index >= self.limit:
            return []
        individual = session.search_space.from_index(self.next_index)
        self.next_index += 1
        return [individual]

    def tell(self, evaluations):
        self.evaluations.extend(evaluations)

    def should_stop(self):
        return self.next_index >= self.limit

    def best_individuals(self):
        if not self.evaluations:
            return ()
        best = max(self.evaluations, key=self.objective.score)
        return (best.individual,)
```

### Algoritmos Multiobjetivo

Los algoritmos multiobjetivo pueden recibir un `ObjectiveSet` y usar `values(...)`, `scores(...)` o `dominates(...)`.

```python
from genio import ObjectiveSet, dominates

class ParetoArchive:
    def __init__(self, objectives: ObjectiveSet) -> None:
        self.objectives = objectives
        self.front = []

    def add(self, evaluation):
        if any(dominates(existing, evaluation, self.objectives) for existing in self.front):
            return
        self.front = [
            existing
            for existing in self.front
            if not dominates(evaluation, existing, self.objectives)
        ]
        self.front.append(evaluation)
```

## 2. Extender Objetivos De Optimizacion

Clases base:

```python
from genio import (
    Objective,
    MetricObjective,
    ObjectiveSet,
    ObjectiveError,
    OptimizationDirection,
    dominates,
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

    def value(self, evaluation: Evaluation) -> float:
        ...

    def score(self, evaluation: Evaluation) -> float:
        ...
```

`score(...)` convierte el objetivo a una convencion uniforme: mayor score es mejor. Si la direccion es `MINIMIZE`, devuelve `-value`.

### `MetricObjective`

Usa una clave de `Result.metrics`.

```python
from genio import MetricObjective, OptimizationDirection

objective = MetricObjective(
    metric="functional.f1",
    optimization_direction=OptimizationDirection.MAXIMIZE,
)
```

Ejemplo de minimizacion:

```python
latency = MetricObjective(
    metric="hls.latency",
    optimization_direction=OptimizationDirection.MINIMIZE,
    id="latency",
)
```

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

    def value(self, evaluation):
        metrics = evaluation.result.metrics
        try:
            f1 = metrics[self.f1_metric]
            latency = metrics[self.latency_metric]
        except KeyError as exc:
            raise ObjectiveError("Missing metric for weighted quality") from exc
        return f1 - 0.001 * latency
```

### `ObjectiveSet`

Agrupa varios objetivos para algoritmos multiobjetivo.

```python
from genio import ObjectiveSet, MetricObjective, OptimizationDirection

objectives = ObjectiveSet((
    MetricObjective(
        metric="functional.f1",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    ),
    MetricObjective(
        metric="hls.latency",
        optimization_direction=OptimizationDirection.MINIMIZE,
    ),
    MetricObjective(
        metric="hls.lut",
        optimization_direction=OptimizationDirection.MINIMIZE,
    ),
))
```

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
- Declarar el tipo de task que produce mediante `task_type`.
- Crear una task concreta para un individuo.
- Usar artefactos acumulados de pasos anteriores si el step depende de ellos.

Ejemplo:

```python
from genio import EvaluationStep, Individual

class FunctionalStep(EvaluationStep):
    id = "functional"
    task_type = FunctionalEvaluationTask

    def create_task(self, individual: Individual, artifacts):
        return FunctionalEvaluationTask(individual=individual, step_id=self.id)
```

Ejemplo con dependencia:

```python
class HlsStep(EvaluationStep):
    id = "hls"
    depends_on = ("compose",)
    task_type = HlsEvaluationTask

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

    def on_evaluation_completed(self, evaluation: Evaluation) -> None:
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
- Acumular metricas de seguimiento.
- Devolver un snapshot serializable.

`OptimizationSession.run()` anota `batch_index` en `Evaluation.metadata`, por lo que cada evaluacion del resultado final puede asociarse al batch en el que fue producida.

Ejemplo:

```python
from genio import StatisticsCollector

class MetricHistory(StatisticsCollector):
    def __init__(self) -> None:
        self.history = []
        self.batches = []

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

    def snapshot(self):
        return {"metrics": self.history, "batches": self.batches}
```

## 11. Montar Una Sesion Completa

Una sesion conecta espacio de busqueda, algoritmo, backend y workflow.

```python
from genio import (
    EvaluationWorkflow,
    LocalBackend,
    MetricObjective,
    OptimizationDirection,
    OptimizationSession,
)

objective = MetricObjective(
    metric="functional.f1",
    optimization_direction=OptimizationDirection.MAXIMIZE,
)

algorithm = BestScoreAlgorithm(objective=objective, limit=10)

workflow = EvaluationWorkflow((
    FunctionalStep(),
))

session = OptimizationSession(
    search_space=search_space,
    algorithm=algorithm,
    backend=LocalBackend(),
    evaluation_workflow=workflow,
)

result = session.run()
```

## 12. Extension Por Tipo De Necesidad

### Necesito Una Nueva Metrica

Implementar o modificar una `EvaluationTask` para devolver un `MetricArtifact`.

```text
EvaluationTask.run -> MetricArtifact.metrics -> Result.metrics
```

### Necesito Un Nuevo Criterio De Optimizacion

Usar `MetricObjective` si basta una metrica. Crear una subclase de `Objective` si hay combinacion, penalizacion o normalizacion.

```text
Result.metrics -> Objective.value/score -> SearchAlgorithm
```

### Necesito Un Nuevo Algoritmo

Crear una subclase de `SearchAlgorithm`. Si requiere ranking, inyectar `Objective`. Si requiere Pareto, inyectar `ObjectiveSet`.

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

### Necesito Un Nuevo Backend

Crear una subclase de `Backend` si la ejecucion no cabe en `LocalBackend`.

### Necesito Generar Codigo O Configuracion

Crear una subclase de `Composer` y usarla dentro de una `EvaluationTask`.

### Necesito Guardar Estadisticas

Crear una subclase de `StatisticsCollector`.

## 13. Reglas De Integracion

- Las metricas que consume el algoritmo siempre deben venir de `Result.metrics`.
- Los nombres de metricas agregadas siguen la forma `step_id.metric_name`.
- Los nombres de artefactos acumulados siguen la forma `step_id.artifact_name`.
- Los artefactos no metricos no se devuelven en `Result`.
- `Objective` y `ObjectiveSet` no son obligatorios en `OptimizationSession`.
- Los algoritmos concretos deciden si necesitan objetivo, conjunto de objetivos o ninguno.
- `batch_index` en `Evaluation.metadata` permite reconstruir poblaciones o generaciones evaluadas.
- El backend no debe contener logica de dominio.
- Las tasks son el lugar correcto para ejecutar herramientas, materializar archivos y producir artefactos.

## 14. Mapa Rapido De Clases Abstractas

| Clase | Se extiende para | Metodo clave |
| --- | --- | --- |
| `SearchAlgorithm` | Nuevas estrategias de busqueda | `ask`, `tell`, `should_stop` |
| `Objective` | Nuevos criterios escalares o compuestos | `value` |
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
7. Implementar o configurar un `SearchAlgorithm`.
8. Definir `MetricObjective` u `ObjectiveSet` si el algoritmo necesita comparar resultados.
9. Crear un `StatisticsCollector` si se necesitan historicos o trazas.
10. Ejecutar la sesion con `OptimizationSession`.
