# Manual De Extensibilidad De GENIO

Este documento describe como extender GENIO a partir de sus clases abstractas y contratos base. El objetivo es servir como manual de usuario para implementar nuevos dominios de evaluacion, algoritmos de busqueda, tareas, artefactos, objetivos, compositores y recolectores de estadisticas.

GENIO separa tres ideas principales:

- El framework define contratos estables.
- Las extensiones concretas implementan logica de dominio.
- Los algoritmos de busqueda consumen `EvaluatedBatch`, no artefactos ni detalles de ejecucion.

Las bases y los algoritmos genericos se importan de `genio` (`src/genio`). Las extensiones instalables de imagen y X-HEEP estan en `genio_use_cases.image_pipeline` y `genio_use_cases.xheep` (`src/genio_use_cases`), respectivamente. `PythonImagePipelineComposer`/`PythonExecutionPackage` viven en `genio_use_cases.image_pipeline.python`, `HLSImagePipelineComposer`/`HLSExecutionPackage` en `.hls`, `PythonImageFunctionalTask` en `.image_functional`, `HLSImagePipelineSynthesisTask` en `.hls_image_pipeline_synthesis`, `HLSRTLArtifact` en `.hls_synthesis`, los parsers HLS en `.hls_reports`, y `GRHeepConfigurationComposer`/`GRHeepConfigurationPackage` y `XHeepVerilatorSimulationTask` en `genio_use_cases.xheep.gr_heep` y `.xheep_verilator_simulation`. Las clases tambien se reexportan desde los paquetes de dominio.

Los runners y escenarios de `use_cases/insect_segmentation` y `use_cases/date_experiments` son experimentos en la raiz del repositorio, no forman parte de los paquetes instalables. Los JSON de escenario estan en `use_cases/insect_segmentation/scenarios/`. Desde la raiz se pueden ejecutar con `python -m use_cases.insect_segmentation.run_random_search`, `python -m use_cases.insect_segmentation.run_population_search`, `python -m use_cases.insect_segmentation.run_genetic_search` y `python -m use_cases.date_experiments.run`. Instalar las dependencias opcionales con `pip install -e '.[image_pipeline]'` o `pip install -e '.[image_pipeline,xheep]'`; los toolchains externos requieren configuracion aparte. El paquete instalado solo contiene codigo: el llamador debe proporcionar los directorios de recursos `search_space/stages`, `hls_templates`, `hls_implementations` y `gr_heep_templates` del checkout.

## Vista General

El flujo extensible es:

```text
OptimizationSession configura SearchAlgorithm con SearchContext
SearchAlgorithm.ask()
        -> Individual[]
EvaluationWorkflow
        -> EvaluationStep(task_class=..., config=...).create_task(...)
        -> EvaluationTask.run(context)
        -> EvaluationOutput(metrics, artifacts)
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
EvaluationOutput
TaskContext
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
- `EvaluationStep` configura una clase de task y crea una instancia por individuo.
- `EvaluationTask` ejecuta trabajo real usando un `TaskContext`.
- La sesion supervisa procesos locales; no hay backends configurables.
- `Artifact` transporta salidas entre steps.
- `EvaluationOutput` separa las metricas numericas de los artefactos; el executor agrega las primeras en `Result.metrics`.
- `Result` no contiene artefactos; solo estado, metricas y error.
- `ObjectiveSet` configura en la sesion la extraccion, normalizacion y scalarizacion.
- `ObjectiveRuntime` mantiene estado aislado de una sesion y entrega `EvaluatedBatch`.
- Cada algoritmo decide la semantica de `best_individuals()`.
- El analisis historico pertenece a las estadisticas y herramientas posteriores, no a `ObjectiveSet`.
- `Composer` ayuda a traducir individuos a representaciones de dominio.
- `ExecutionPackage` define la base generica para el paquete materializable que puede producir un composer; sus subclases Python, HLS y GR-HEEP estan en `genio_use_cases`.
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
- Leer artefactos intermedios de otras tareas.
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

No expone el `ObjectiveSet` ejecutable, workflow ni sesion completa.
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
from genio import Artifact, EvaluationOutput
```

### `Artifact`

`Artifact[T]` es una dataclass concreta, generica, frozen y slotted. Transporta
datos estructurados pequenos y archivos producidos por una task para steps posteriores.

Campos:

```python
data: T | None = None
files: Mapping[str, Path] = field(default_factory=dict)
```

`data` debe poder transportarse con `pickle`; `files` asocia nombres logicos a
rutas de archivos existentes dentro del workspace de la task productora. No hay
`load()`, ni campos `name`, `producer`, `individual_id`, `objective` o `metadata`.
No es necesario crear una subclase para representar un archivo:

```python
from genio import Artifact

# Dentro de run(context), despues de crear el archivo:
path = context.workspace / "pipeline.json"
path.write_text('{"ok": true}', encoding="utf-8")
artifact = Artifact[dict[str, str]](data={"format": "json"}, files={"pipeline": path})
```

`HLSRTLArtifact` es una subclase marcador de `Artifact[Mapping[str, Any]]`.
Se importa de `genio_use_cases.image_pipeline`, no de `genio`.
Su `data` contiene `origin`, `top_function`, `interface` y metadatos de imagen;
sus `files` usan nombres `verilog/<path>` o `vhdl/<path>` para el RTL que consume
X-HEEP. No anade campos ni metodos de carga.

### `EvaluationOutput`

Dataclass inmutable definida en `genio.evaluation.task`. Una task devuelve metricas
locales y, por separado, los artefactos persistentes que necesitan otros steps:

```python
metrics: Mapping[str, float] = field(default_factory=dict)
artifacts: Mapping[str, Artifact] = field(default_factory=dict)
```

Por ejemplo, `EvaluationOutput(metrics={"latency": 120.0, "lut": 3021.0})`
devuelto por el step `hls` produce `hls.latency` y `hls.lut` en `Result.metrics`.
Las tasks concretas HLS y X-HEEP conservan ademas sus espacios locales
`hls_synthesis.*` y `xheep_verilator.*`, respectivamente.

Reglas importantes:

- `EvaluationOutput.metrics` debe ser un mapa de nombres locales no vacios a valores numericos.
- Los booleanos no se aceptan como metricas.
- Las claves duplicadas provocan `EvaluationExecutionError`.
- `EvaluationOutput.artifacts` es un mapa dinamico de nombres locales no vacios y sin puntos
  a instancias de `Artifact`; el executor los acumula como `step_id.nombre` por individuo.
- Los artefactos no aparecen en `Result`.
- Los artefactos sirven para steps posteriores o persistencia externa futura.

El descriptor de clase `ArtifactInput(source, artifact_type)` define una entrada
cualificada y su tipo; el step deriva `required_artifacts` de `requires`. No se
declaran las salidas por clase: `run` las devuelve en `EvaluationOutput.artifacts`.
Si un productor termina con exito pero no entrega una entrada requerida o entrega
un tipo incompatible, `EvaluationExecutionError` aborta la sesion. Una salida no
transportable por `pickle` tambien es un error de contrato; en cambio, una excepcion
durante `run()` solo falla al individuo correspondiente.

`OptimizationSession` elimina por defecto los workspaces de `work_dir/workspaces`
al cerrar cada batch; `keep_all_individuals=True` los conserva para depuracion.
Los logs se conservan por separado bajo `work_dir/logs`.
Las referencias a archivos generados solo son validas durante el batch por defecto;
para conservarlos despues hay que activar `keep_all_individuals=True` o copiarlos
fuera del workspace por separado.

## 4. Extender Tareas Ejecutables

Clase base:

```python
from genio import EvaluationTask, TaskContext
```

Contrato:

```python
@dataclass(frozen=True, slots=True)
class EvaluationTask(ABC):
    individual: Individual
    id: str | None = None
    step_id: str | None = None
    artifacts: Mapping[str, Artifact] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_state: dict[str, Any] = field(default_factory=dict, init=False)
    requires: ClassVar[tuple[ArtifactInput, ...]] = ()

    @property
    def task_id(self) -> str:
        ...

    def run(self, context: TaskContext) -> EvaluationOutput:
        ...

    def on_timeout(self, context: TaskContext) -> None:
        pass
```

Responsabilidades:

- Ejecutar una unidad concreta de trabajo.
- Declarar entradas cualificadas con `ArtifactInput(source, artifact_type)` en `requires`.
- Leer los artefactos requeridos en `self.artifacts[source]`, inyectados por el step.
- Usar `context.workspace` para archivos y `context.resources` para recursos de la sesion; gestionar filesystem, logs y comandos con la biblioteca estandar.
- Devolver `EvaluationOutput` con metricas y artefactos nombrados dinamicamente.
- Encapsular logica de dominio.

Opcionalmente, una subclase declara `uses_domains = ("pipeline", "hls")` como
atributo de clase **sin anotacion de dataclass** para limitar los dominios de
`Individual.values` relevantes para su cache. Si se omite, se asume que usa todos.
Las tasks integradas declaran `("pipeline",)` para funcional, `("pipeline", "hls")`
para sintesis y `("system",)` para X-HEEP. El executor anade transitivamente los
dominios de los productores de artefactos requeridos; por ello la cache de X-HEEP
tambien depende de `pipeline` y `hls` a traves del RTL HLS. Declare solo los
dominios que realmente afectan a la salida propia de la task.

`on_timeout` es opcional y solo se llama dentro del proceso hijo cuando el supervisor
envia `SIGTERM` por vencimiento del plazo. Puede usar `runtime_state` para acceder a
datos que `run` guardo en ese mismo proceso (por ejemplo, el identificador de un
contenedor). La clase base gestiona la interrupcion; las extensiones no instalan
handlers ni sobrescriben `_execute`. El supervisor espera hasta tres segundos antes
de terminar el grupo con `SIGKILL` si la tarea o el hook no concluyen. El hook no
puede garantizar limpieza si el proceso no atiende la senal, por lo que recursos
externos criticos pueden requerir una via adicional basada en IDs persistidos.

Por ejemplo, una task que usa Docker puede guardar
`self.runtime_state["container_name"]` antes de esperar a `docker run` y definir
un hook sin instalar ninguna senal por su cuenta:

```python
def on_timeout(self, context: TaskContext) -> None:
    name = self.runtime_state.get("container_name")
    if name:
        subprocess.run(["docker", "rm", "-f", name], timeout=2, check=False)
```

No debe:

- Modificar directamente el estado del algoritmo.
- Decidir objetivos de optimizacion.
- Depender de transporte remoto o de un backend eliminado.

### Ejemplo De Task

```python
import json
from genio import EvaluationTask, EvaluationOutput, TaskContext

class FunctionalEvaluationTask(EvaluationTask):
    def run(self, context: TaskContext) -> EvaluationOutput:
        (context.workspace / "report.json").write_text(
            json.dumps({"f1": 0.91}), encoding="utf-8"
        )
        return EvaluationOutput(metrics={"f1": 0.91})
```

## 5. Usar `TaskContext`

`TaskContext` lo crea el worker local supervisado y se entrega a cada task.
El executor ya ha creado `workspace` para ese individuo y step.

Campos:

```python
workspace: Path
resources: Mapping[str, Any] = field(default_factory=dict)
```

`resources` contiene el `execution_metadata` de la sesion; no debe confundirse
con `EvaluationTask.metadata`. `TaskContext` no ofrece helpers ni `CommandResult`.
Usar `pathlib` y `json` para archivos, `shutil` para copias y `subprocess` para
comandos. Colocar los archivos referenciados por `Artifact.files` dentro del
workspace de la task productora y crearlos antes de devolver `EvaluationOutput`.

Ejemplo:

```python
import subprocess
import sys

class CommandTask(EvaluationTask):
    def run(self, context: TaskContext) -> EvaluationOutput:
        log_dir = context.workspace / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "task.log").write_text("started\n", encoding="utf-8")
        subprocess.run(
            [sys.executable, "-c", "print('ok')"],
            cwd=context.workspace,
            check=True,
        )
        return EvaluationOutput()
```

Por defecto `subprocess.run` hereda stdout/stderr y el grupo de procesos
supervisado; usar `capture_output=True, text=True` solo si la task necesita
inspeccionar la salida. El timeout del step se aplica a la task completa.

Regla de diseño:

- `TaskContext` expone solo el workspace y los recursos compartidos de ejecucion.
- No se deben anadir helpers especificos como `run_vitis_hls()` o `load_dataset()`.
- Esos detalles pertenecen a tasks, composers o configs de dominio.

## 6. Configurar Steps De Evaluacion

Clase base:

```python
from genio import EvaluationStep
```

Contrato:

```python
@dataclass(frozen=True, slots=True)
class EvaluationStep:
    id: str
    task_class: type[EvaluationTask]
    config: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
```

Responsabilidades:

- Declarar el id logico del paso.
- Asociar `task_class` y sus parametros de constructor en `config` (excepto los
  campos inyectados `individual`, `step_id` y `artifacts`).
- Derivar `required_artifacts` de `requires`; el workflow ordena automaticamente
  los steps productores por su id antes de los consumidores.
- Crear una task concreta con `individual`, `step_id`, `artifacts` y `config`.

Ejemplo:

```python
from genio import EvaluationStep

functional_step = EvaluationStep(
    id="functional",
    task_class=FunctionalEvaluationTask,
    config={},
)
```

Ejemplo con dependencia:

```python
from dataclasses import dataclass
from typing import ClassVar
from genio import ArtifactInput, EvaluationOutput, EvaluationTask

@dataclass(frozen=True, slots=True)
class HlsEvaluationTask(EvaluationTask):
    requires: ClassVar[tuple[ArtifactInput, ...]] = (
        ArtifactInput("compose.project", ProjectArtifact),
    )
    hls_tool: str = "v++"

    def run(self, context):
        project = self.artifacts["compose.project"]
        # Ejecutar self.hls_tool sobre project.files dentro de context.workspace.
        rtl = RTLArtifact(data={"tool": self.hls_tool}, files={})
        return EvaluationOutput(artifacts={"rtl": rtl})

hls_step = EvaluationStep(
    id="hls",
    task_class=HlsEvaluationTask,
    config={"hls_tool": "v++"},
    timeout_seconds=1800,
)
```

Reglas:

- `create_task(...)` pasa solo los artefactos requeridos (mapa con claves cualificadas)
  al constructor tipado de la task, junto con individuo, id del step y configuracion.
- `checkpoint_signature()` incluye id, clase de task, config, timeout y requisitos
  de entrada; el workflow firma los steps en orden efectivo.
- `id` debe ser unico dentro del workflow.
- El id productor de cada artefacto requerido debe corresponder a otro step del workflow.
- Los nombres devueltos en `EvaluationOutput.artifacts` son locales, no vacios y no contienen `.`.
- El workflow comprueba la existencia del step productor y los ciclos, no si este
  declara el nombre o tipo de salida; el executor valida presencia y tipo al consumirla.
- Cada archivo referenciado debe existir dentro del workspace de la task productora.

## 7. Declarar Workflows De Evaluacion

Clase:

```python
from genio import EvaluationWorkflow
```

Uso:

```python
workflow = EvaluationWorkflow((
    compose_step,
    functional_step,
    hls_step,
))
```

`EvaluationWorkflow` valida:

- ids duplicados;
- requisitos cuyo id de step productor no existe o con dependencias circulares;
- claves de requisitos invalidas o tipos de entrada que no son `Artifact`.

Si un productor termina con exito pero omite un artefacto requerido, o entrega un
tipo incorrecto, el executor lanza `EvaluationExecutionError` y aborta toda la
sesion de optimizacion. Si la ejecucion de la task productora falla, solo falla
ese individuo y los demas pueden continuar.

El orden de ejecucion se obtiene con `execution_order()`; solo difiere del orden declarado cuando un step necesita un artefacto de otro situado mas tarde:

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

## 8. Ejecucion Local Supervisada

`OptimizationSession` ejecuta las tasks en procesos locales supervisados, incluso
con `max_workers=1`. `max_workers` controla la concurrencia; `work_dir` es la raiz
de los workspaces y `execution_metadata` transporta recursos como rutas de Vitis.
Cada task solo implementa `run(context: TaskContext) -> EvaluationOutput`.

Los logs se guardan bajo `work_dir/logs/<run_id>/<individual>/<step>`, por separado
del workspace `work_dir/workspaces/<individual>/<step>`. El supervisor captura
stdout/stderr del proceso Python en `stdout.log` y `stderr.log`. Los subprocesos
heredan esos descriptores y el grupo supervisado salvo que la task capture su
salida deliberadamente. Las tasks integradas escriben archivos especificos en
`workspace/logs`; al limpiar el workspace tras un batch completado, se copian a
`work_dir/logs/<run_id>/<individual>/<step>/task`.
Un `EvaluationStep` puede declarar `timeout_seconds=None` (sin limite) o un
timeout positivo para cada task. El supervisor controla la finalizacion y la
cancelacion de procesos. No existe API de extensiones para backends o handles;
el soporte SSH ha sido eliminado.

Las clases de tasks deben poder importarse desde su modulo y sus campos deben
admitir serializacion con `pickle`: el executor valida esto antes de lanzarlas.
Abrir conexiones o crear locks dentro de `run()` evita transportarlos al worker;
`TaskContext` se crea alli y no forma parte de los datos serializados.

### Cache De Evaluaciones

La `EvaluationCache` pertenece a `OptimizationSession`, no a las extensiones:
almacena salidas exitosas por step durante esa sesion. Se configura con
`cache_capacity={"functional": 100, "hls": 20, "simulation": None}`, usando IDs
de steps como claves. Un step omitido o con `0` no usa cache, un valor positivo
limita sus entradas mediante LRU y `None` no limita su numero. La cache comienza
vacia al iniciar o reanudar y se elimina al terminar `run()`; no forma parte de
los checkpoints. El executor agrupa evaluaciones equivalentes dentro de cada ola,
ademas de reutilizar entradas de olas anteriores de la misma sesion.

Una task cacheable debe producir la misma salida para los dominios declarados y los
artefactos de entrada equivalentes; se presupone que recursos externos como datasets,
toolchains y templates son inmutables durante la sesion. Los archivos declarados en
`Artifact.files` se copian al almacenamiento de cache y se restauran como copias
separadas dentro del workspace de cada individuo, sin reutilizar paths del
productor. Los archivos no declarados no forman parte de las salidas cacheadas;
`Artifact.data` no debe guardar rutas a archivos generados porque no se reubican.
Esas rutas deben estar en `Artifact.files`. Un acierto no ejecuta la task y por
tanto tampoco genera nuevos logs de esa task.
`keep_all_individuals=True` sigue siendo necesario para retener workspaces completos.

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
- Consumir solo los dominios de `individual.values` que correspondan a su backend, por ejemplo `hls` para síntesis o `system` para integración de plataforma. Los consumidores actuales deben migrar a esta representación.
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
import json

class ComposeTask(EvaluationTask):
    def run(self, context: TaskContext) -> EvaluationOutput:
        composer = PipelineComposer(context.resources["stages_definitions_path"])
        pipeline = composer.compose(self.individual)
        path = context.workspace / "pipeline.json"
        path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
        return EvaluationOutput(artifacts={
            "pipeline": Artifact(data={"format": "json"}, files={"pipeline": path}),
        })
```

El consumidor recibe este artefacto con la clave cualificada `compose.pipeline`.
El llamador inyecta `stages_definitions_path` en `execution_metadata`, por ejemplo
`Path.cwd().resolve() / "search_space/stages/definitions"` al ejecutarse desde la
raiz del checkout; en otras ubicaciones debe indicar su ruta absoluta.

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

Cuando una sesion necesita varios destinos, `CompositeStatisticsCollector` propaga
todos los hooks en orden y mantiene snapshots y estados sin mezclar claves:

```python
from genio import (
    CompositeStatisticsCollector,
    CSVStatisticsCollector,
    InMemoryStatistics,
)

statistics = CompositeStatisticsCollector((
    CSVStatisticsCollector("results"),
    InMemoryStatistics(),
))
```

La entrega es fail-fast: si un collector falla, los collectors posteriores no reciben
ese evento. El checkpoint compuesto solo esta disponible cuando todos sus hijos lo
soportan; configuracion y estado preservan el orden declarado.

Para el caso habitual CSV + analisis poblacional se recomienda la fachada:

```python
from genio import PopulationPlotConfig, PopulationStatisticsCollector

statistics = PopulationStatisticsCollector(
    "results",
    plots=PopulationPlotConfig(
        every_batches=5,
        tracked_genes=(0, 2, 4),
    ),
)
```

Esta fachada expone `csv_collector` y `analysis_collector` para configuracion avanzada,
y devuelve un snapshot con claves separadas `csv` y `analysis`.

La fachada soporta checkpoints. El collector poblacional persiste solo indices de
batches y best IDs; records y snapshots se reconstruyen desde los `EvaluatedBatch`
autoritativos de `OptimizationSession`. Figuras e imagenes nunca se serializan: los
plots intermedios/finales y ambos JSON se regeneran al restaurar.

Los collectors orientados a poblaciones pueden usar los modelos inmutables de
`genio.statistics` sin depender del CSV:

```python
record = PopulationRecord.from_proposal(proposal)
record = record.with_evaluation(evaluated_individual)
snapshot = PopulationSnapshot(batch_index, records)
```

`PopulationRecord` conserva la ocurrencia mediante `proposal_id`, genotipo, stages,
parametros, design, metadata algoritmica, estado, objetivos raw y score agregado.
`PopulationSnapshot` exige posiciones contiguas de un mismo batch y expone vistas de
registros evaluados, objetivos validos y genotipos disponibles.

`PopulationAnalysisCollector(output_dir)` construye estos modelos directamente desde
los hooks. Valida el orden `batch started -> proposals -> evaluated batch -> batch
completed`, correlaciona por `proposal_id` y conserva al finalizar los IDs devueltos por
`algorithm.best_individuals()`. Al completar la sesion escribe el resumen numerico,
manifest y graficos finales. `every_batches=N` habilita snapshots intermedios.

Los calculos reutilizables viven en `genio.statistics.population_analysis` y no
dependen del renderer. Incluyen frecuencias de stages/genes/parametros, entropia,
ratio de genotipos unicos, distancia Hamming, resumen de objetivos y scores, tasa de
fallos, one-hot categorico, PCA y matrices de los individuos seleccionados:

```python
entropy = gene_entropy(snapshot)
distance = mean_pairwise_hamming_distance(snapshot)
objectives = objective_summary_by_batch(collector.snapshots)
projection = project_population_pca(collector.records)
```

La PCA nunca usa `search_index` como coordenada: codifica cada valor de gen como una
categoria para no introducir distancias artificiales entre indices mixed-radix.

`PopulationPlotRenderer` convierte esos calculos en graficos headless mediante
Matplotlib Agg. En esta fase se invoca explicitamente sobre los snapshots capturados:

```python
renderer = PopulationPlotRenderer(PopulationPlotConfig(
    tracked_genes=(0, 2, 4),
    image_format="png",
    strict=False,
))
result = renderer.render(
    collector.snapshots,
    best_individual_ids=collector.best_individual_ids,
    target_dir="results/analysis/plots/final",
)
```

El resultado distingue `generated`, `skipped` y `warnings`. Se generan composicion,
diversidad, PCA, evolucion de objetivos/score, fallos, dispersion de dos objetivos y
genes seleccionados. Un grafico sin datos aplicables se omite; con `strict=False`, un
fallo de renderizado no oculta los demas graficos.

Salida automatica del collector:

```text
analysis/
├── analysis_summary.json
├── analysis_manifest.json
└── plots/
    ├── final/
    └── batch_000000/
```

`analysis_summary.json` contiene diversidad, fallos, objetivos, scores y distribucion
final de stages. El manifest usa rutas relativas y conserva graficos omitidos y warnings.
Los JSON se reemplazan atomicamente y rechazan `NaN`/`Infinity`.

## 11. Montar Una Sesion Completa

Una sesion conecta espacio de busqueda, algoritmo, workflow y la configuracion
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
    max_workers=4,
    work_dir=Path("tmp/genetic-work"),
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
    max_workers=4,
    work_dir=Path("tmp/nsga2-work"),
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

Implementar o modificar una `EvaluationTask` para devolver la nueva metrica en `EvaluationOutput.metrics`.

```text
EvaluationTask.run -> EvaluationOutput.metrics -> Result.metrics
```

### Necesito Un Nuevo Criterio De Optimizacion

Usar `MetricObjective` si basta una metrica. Crear una subclase de `Objective` si hay combinacion o penalizacion, y configurar la normalizacion separadamente.

```text
Result.metrics -> ObjectiveRuntime -> EvaluatedBatch -> SearchAlgorithm
```

### Necesito Un Nuevo Algoritmo

Crear una subclase de `SearchAlgorithm`, validar sus necesidades en `configure(context)` y consumir `EvaluatedBatch`. El `ObjectiveSet` se configura en `OptimizationSession`, no se inyecta en el algoritmo.

### Necesito Un Nuevo Paso De Evaluacion

Crear una subclase de `EvaluationTask` con `requires` si consume artefactos y
devolver salidas nombradas en `EvaluationOutput.artifacts`; configurar una instancia
generica de `EvaluationStep` con su clase y parametros.

```text
EvaluationStep.create_task -> EvaluationTask.run
```

Las extensiones instalables incluyen como referencia un workflow de tres fases:

```text
EvaluationStep(task_class=PythonImageFunctionalTask)
    -> EvaluationStep(task_class=HLSImagePipelineSynthesisTask)
    -> EvaluationStep(task_class=XHeepVerilatorSimulationTask)
```

El último step demuestra cómo consumir un artefacto RTL, materializar un checkout
externo aislado, preservar symlinks, ejecutar comandos dentro de Conda y convertir
líneas `GENIO_METRIC:nombre:valor` en métricas. Véase
[Evaluación de pipelines HLS en X-HEEP con SAFA](XHEEP_SAFA_EVALUATION.md).
El compositor selecciona exclusivamente mediante `application_name` los flujos
`genio_trans_mem_mem`, `genio_trans_mem_flash` o `genio_trans_flash_mem`.

### Necesito Mas Paralelismo

Configurar `max_workers` en `OptimizationSession`. La ejecucion remota por SSH
ya no esta soportada.

### Necesito Generar Codigo O Configuracion

Crear una subclase de `Composer` y usarla dentro de una `EvaluationTask`.

### Necesito Guardar Estadisticas

Crear una subclase de `StatisticsCollector`.

## 13. Reglas De Integracion

- Los objetivos proceden siempre de `Result.metrics`, pero el algoritmo los recibe ya transformados en `EvaluatedBatch`.
- Los nombres de metricas agregadas siguen la forma `step_id.metric_name`.
- Los nombres de artefactos acumulados siguen la forma `step_id.artifact_name`.
- Los artefactos no se devuelven en `Result`.
- `ObjectiveSet` es opcional en `OptimizationSession`; sus estrategias tienen defaults `None`.
- Los algoritmos declaran sus requisitos mediante `SearchContext` y deciden su propio `best_individuals()`.
- `ObjectiveSet` no almacena historia ni selecciona resultados globales.
- La ponderacion de objetivos pertenece a `WeightedMeanScalarizer`.
- Para comparar scores entre generaciones se necesita una escala estable, normalmente `MinMaxNormalizer(FIXED)` con bounds completos.
- `batch_index` en `Evaluation.metadata` permite reconstruir poblaciones o generaciones evaluadas.
- La infraestructura de procesos no debe contener logica de dominio.
- Las tasks son el lugar correcto para ejecutar herramientas, materializar archivos y producir artefactos.

### Checkpoints De Extensiones

Toda extension que participe en la firma de una sesion debe declarar de forma
determinista y serializable como JSON la configuracion que afecta a su comportamiento:

- `Objective`, `Normalizer` y `Scalarizer` deben implementar siempre `checkpoint_signature()`; sus bases rechazan la firma implicita.
- `SearchAlgorithm`, `StatisticsCollector` y `Composer` deben incluir todos sus parametros relevantes en `checkpoint_signature()`.
- `EvaluationStep` firma id, clase de task, config, timeout y requisitos de entrada; `EvaluationWorkflow` firma el orden efectivo.
- La configuracion de una `EvaluationTask` queda cubierta por el `EvaluationStep` que la construye.
- Un algoritmo con `supports_checkpointing=True` implementa `checkpoint_state()` y `restore_checkpoint_state(state, *, search_space, evaluated_batches)`.
- Un normalizer con estado implementa `NormalizationState.checkpoint_state()` y `restore_state(data)`.
- El estado mutable no debe duplicar la historia: los `EvaluatedBatch` de la sesion son la fuente autoritativa para restaurar algoritmos.
- `OptimizationSession` es el unico lector y escritor del formato de checkpoint; no existen schemas versionados ni formatos por algoritmo.

Cambiar una metrica, bound, peso, template, dataset, configuracion de ejecucion o parametro algorítmico
debe cambiar la firma y hacer incompatible la reanudacion.

## 14. Mapa Rapido De Clases Abstractas

| Clase | Se extiende para | Metodo clave |
| --- | --- | --- |
| `SearchAlgorithm` | Nuevas estrategias de busqueda | `configure`, `ask`, `tell`, `should_stop` |
| `Objective` | Nuevos criterios escalares o compuestos | `value`, `checkpoint_signature` |
| `Normalizer` | Nuevas transformaciones vectoriales con estado propio | `validate`, `fit`, `transform`, `restore_state` |
| `Scalarizer` | Nuevas agregaciones a score | `validate`, `scalarize` |
| `Artifact[T]` | Datos estructurados pequenos y archivos para steps posteriores | campos `data`, `files` |
| `EvaluationOutput` | Metricas y artefactos separados de cada task | campos `metrics`, `artifacts` |
| `EvaluationTask` | Nuevas unidades ejecutables | `run` |
| `EvaluationStep` | Configurar tareas en el workflow | `task_class`, `config`, `timeout_seconds` |
| `Composer` | Nuevas traducciones de individuos | `compose` |
| `StatisticsCollector` | Nuevos recolectores de eventos | hooks de sesion |

## 15. Checklist Para Una Extension De Dominio

1. Definir las metricas y, si hacen falta, los artefactos que produce cada fase.
2. Implementar `Artifact` solo para las salidas consumidas por otros steps.
3. Implementar una `EvaluationTask` por unidad ejecutable.
4. Configurar un `EvaluationStep` generico por task.
5. Construir un `EvaluationWorkflow` con dependencias declaradas en `EvaluationTask.requires`.
6. Configurar `max_workers`, `work_dir` y recursos con `execution_metadata` en la sesion.
7. Implementar o configurar un `SearchAlgorithm` con `configure`, `ask()` y `tell(batch)`.
8. Definir `ObjectiveSet` en la sesion si el algoritmo necesita comparar resultados.
9. Elegir explicitamente normalizer y scalarizer; usar bounds fijos si se comparan generaciones.
10. Implementar firmas completas de checkpoint para todas las extensiones configurables.
11. Crear un `StatisticsCollector` si se necesitan historicos o trazas.
12. Ejecutar la sesion con `OptimizationSession`.
