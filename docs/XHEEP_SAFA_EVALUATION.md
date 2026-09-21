# Evaluación De Pipelines HLS En X-HEEP Con SAFA

GENIO puede sintetizar un pipeline de imagen con Vitis HLS, integrarlo como acelerador
streaming en SAFA y medirlo mediante una simulación Verilator de GR-HEEP/X-HEEP.

El flujo completo es:

```text
evaluación funcional
    -> síntesis HLS con interfaz safa_fifo
    -> composición del overlay GR-HEEP
    -> generación de MCU
    -> compilación de Verilator
    -> compilación del firmware
    -> simulación y extracción de métricas
```

## Requisitos Del Host

El host de ejecución debe proporcionar:

- Un checkout existente de GR-HEEP, indicado mediante `gr_heep_path`.
- El entorno Conda `core-v-mini-mcu`.
- El toolchain RISC-V y FuseSoC disponibles dentro de ese entorno.
- Vitis y Vitis Libraries para el step HLS anterior.

El ejemplo `tests/run_insect_random_search.py` usa:

```python
GR_HEEP_PATH = Path("/home/joselu/Universidad/Doctorado/GEN-HEEP")
```

Cada comando X-HEEP se ejecuta como:

```bash
conda run --no-capture-output -n core-v-mini-mcu <comando>
```

El entorno y el ejecutable pueden cambiarse con `conda_environment` y `conda_tool`.

## Configuración Del Workflow

```python
xheep_step = XHeepVerilatorSimulationEvaluationStep(
    depends_on=(hls_step.id,),
    composer=GRHeepConfigurationComposer(
        ROOT / "search_space/stages/definitions",
        templates_path=ROOT / "gr_heep_templates",
        application_name="genio_trans_flash_mem",
    ),
    gr_heep_path=Path("/home/joselu/Universidad/Doctorado/GEN-HEEP"),
    metadata={"execution": {"timeout_seconds": 1800}},
)
```

El step requiere un `HLSRTLArtifact` con `metadata["interface"] == "safa_fifo"`.
El nombre del módulo superior y las dimensiones de entrada/salida se obtienen del
artefacto HLS.

El nombre de la aplicación se configura exclusivamente mediante
`GRHeepConfigurationComposer.application_name`. El step y la tarea de simulación lo
obtienen del compositor para generar el overlay y sustituir `{application_name}` en
los comandos, evitando configuraciones divergentes.

### Aplicaciones Y Flujos De Datos

Las tres plantillas seleccionables mantienen el mismo protocolo SAFA y de metricas,
pero cambian donde reside la entrada y donde termina la salida:

| `application_name` | Flujo esperado | Observaciones |
|---|---|---|
| `genio_trans_mem_mem` | SRAM -> DMA HW FIFO -> SAFA -> SRAM | La imagen embebida y el buffer de salida residen en memoria del sistema. |
| `genio_trans_mem_flash` | SRAM -> DMA HW FIFO -> SAFA -> SRAM de staging -> SPI flash | La salida se programa en W25Q128JW despues de terminar SAFA. `FLASH_OUTPUT_OFFSET` selecciona una region alineada a sector. |
| `genio_trans_flash_mem` | SPI flash -> FIFO RX SPI -> DMA HW FIFO -> SAFA -> SRAM | La imagen queda en la seccion flash-only y se lee por SPI estandar; la salida permanece en memoria. |

En `genio_trans_mem_flash` no se puede conectar directamente el stream de salida de
SAFA a la memoria SPI: W25Q128JW exige borrado de sectores, `WRITE ENABLE`, comandos
`PAGE PROGRAM`, limites de pagina y esperas de estado `BUSY`. El DMA HW FIFO no
implementa esa maquina de estados, por lo que el buffer SRAM de staging es parte del
flujo previsto. En `genio_trans_flash_mem`, en cambio, el FIFO RX del controlador SPI
si proporciona una fuente con backpressure utilizable directamente por el DMA.

Para seleccionar otro flujo solo cambia el compositor:

```python
composer = GRHeepConfigurationComposer(
    ROOT / "search_space/stages/definitions",
    templates_path=ROOT / "gr_heep_templates",
    application_name="genio_trans_mem_flash",
    application_defaults={"FLASH_OUTPUT_OFFSET": "0x00900000"},
)
```

`application_defaults` debe contener un offset de salida solo para la variante que
escribe flash. No se configura un segundo nombre de aplicacion en
`XHeepVerilatorSimulationEvaluationStep` ni en `XHeepVerilatorSimulationTask`.

La secuencia predeterminada es:

```text
make mcu-gen
make verilator-build
make app PROJECT=<application_name seleccionado>
make verilator-run
```

El checkout se copia de forma aislada para cada individuo. Los enlaces simbólicos se
preservan, especialmente:

```text
sw/build -> ../hw/vendor/x-heep/sw/build
```

Ese enlace es parte del contrato de software externo de X-HEEP: el firmware se compila
en el build vendorizado y los targets de simulación lo consumen mediante `sw/build`.

## Overlay Generado

`GRHeepConfigurationComposer` genera o instala:

- `config/mcu-gen-config.py` con CPU, bus, memoria y DMA.
- `hw/vendor/safa/rtl/safa_wrapper.sv`.
- El `.core` de FuseSoC y todos los ficheros RTL producidos por HLS.
- El driver SAFA y su mapa de registros.
- La aplicación seleccionada, por ejemplo `sw/applications/genio_trans_mem_mem`.
- `genio_app_config.h` con tamaños derivados del artefacto HLS.
- `main.h` con la imagen de entrada embebida.

Si se proporciona `input_image_path`, el composer selecciona el modo OpenCV desde
`artifact.metadata["input_type"]`: `XF_8UC1` carga grayscale y `XF_8UC3` carga BGR.
Después redimensiona y empaqueta los bytes en palabras little-endian de 32 bits,
validando canales, tipo `uint8` y `input_words`. Sin imagen se utiliza un patrón
sintético determinista; otros tipos XF se rechazan explícitamente.
La plantilla seleccionada decide si esas palabras se enlazan en SRAM o en una seccion
flash-only; esta eleccion no se codifica en el individuo ni en la task.

## Hiperparámetros Del Sistema

Las claves consumidas directamente por el composer son:

| Clave | Efecto |
|---|---|
| `cpu` | Núcleo X-HEEP, por ejemplo `cv32e20` o `cv32e40px`. |
| `bus_type` | Topología de bus soportada por X-HEEP. |
| `dma_fifo_depth` | Profundidad interna del DMA. |
| `accelerator_fifo_depth` | Profundidad de las FIFO de entrada y salida de SAFA. |
| `accelerator_fifo_almost_full_margin` | Margen de anticipación de `almost_full`. |
| `memory_total_kib` | Capacidad SRAM total. |
| `memory_bank_size_kib` | Tamaño uniforme de cada banco. |
| `memory_interleaved_ratio` | Porcentaje de capacidad destinado a bancos entrelazados. El número derivado de bancos debe ser potencia de dos. |
| `memory_placement` | Colocación de código y datos. |

El composer deriva:

```text
interleaved_kib = memory_total_kib * memory_interleaved_ratio / 100
continuous_kib  = memory_total_kib - interleaved_kib
continuous_banks  = continuous_kib / memory_bank_size_kib
interleaved_banks = interleaved_kib / memory_bank_size_kib
```

Las capacidades continua y entrelazada deben ser divisibles por el tamaño de banco.
El ratio debe estar entre 0 y 99 porque el diseño actual necesita al menos un banco
continuo. `memory_placement="input_output_interleaved"` requiere al menos un banco
entrelazado. X-HEEP exige además que el número de bancos entrelazados sea una potencia
de dos. Un producto cartesiano válido es `memory_total_kib=[128,256,512]`,
`memory_bank_size_kib=[16,32]` y `memory_interleaved_ratio=[0,50]`.

Los valores derivados se conservan en metadata como:

```text
MEMORY_TOTAL_KIB
MEMORY_CONTINUOUS_KIB
MEMORY_INTERLEAVED_KIB
RAM_BANKS
INTERLEAVED_BANK_COUNT
INTERLEAVED_BANK_SIZE
```

## Tráfico Concurrente

El firmware utiliza dos canales DMA:

- Canal 0: transferencia streaming HW FIFO entre memoria y SAFA.
- Canal 1: copia memoria-memoria concurrente de 1024 palabras.

El DMA competidor se lanza inmediatamente después del DMA de SAFA. Esto representa
doble buffering o carga del siguiente frame y genera contención sostenida sin depender
de que la CPU mononúcleo ejecute accesos durante el acelerador.

## Contadores SAFA

SAFA expone por MMIO:

| Métrica | Significado |
|---|---|
| `safa_active_cycles` | Ciclos durante los estados activos de la transacción. |
| `safa_input_stall_cycles` | Ciclos en que HLS solicita entrada y la FIFO está vacía. |
| `safa_output_stall_cycles` | Ciclos en que HLS produce salida y la FIFO está llena. |
| `safa_dma_push_stall_cycles` | Ciclos con `push` DMA bloqueado en la entrada. |
| `safa_dma_pop_stall_cycles` | Ciclos con `pop` DMA sin salida disponible. |
| `safa_input_words` | Palabras aceptadas por SAFA. |
| `safa_output_words` | Palabras extraídas de SAFA. |

El firmware también informa ciclos de aplicación y estado final. Las variantes con SPI
añaden metricas de bytes, offsets y ciclos de
lectura o escritura flash sin cambiar las etiquetas comunes de SAFA.

## Formato De Log Y Métricas

El firmware emite líneas estables:

```text
GENIO_PERF:application:43807
GENIO_METRIC:safa_active_cycles:43807
GENIO_METRIC:safa_input_stall_cycles:120
GENIO_STATUS:0
```

El evaluador convierte `GENIO_METRIC:nombre:valor` y `GENIO_PERF` en métricas del
artefacto `XHeepSimulationArtifact`. Al agregarse al resultado aparecen bajo el prefijo
del step:

```text
xheep_verilator_simulation.xheep_verilator.safa_active_cycles
xheep_verilator_simulation.xheep_verilator.application_cycles
```

La simulación se considera fallida si falta `GENIO_STATUS`, si su valor no es cero o
si X-HEEP informa un `Program Finished with value` distinto de cero.

## Uso Como Objetivos

Las metricas permanecen crudas en `Result.metrics`. Su direccion, normalizacion y
ponderacion se configuran en el `ObjectiveSet` de `OptimizationSession`, no en la task
X-HEEP. Por ejemplo, una escala estable para varias generaciones puede declarar:

```python
from genio import (
    MetricObjective,
    MinMaxNormalizer,
    NormalizationScope,
    ObjectiveSet,
    OptimizationDirection,
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
            "xheep_verilator_simulation.xheep_verilator.application_cycles",
            OptimizationDirection.MINIMIZE,
            name="cycles",
            normalization_bounds=(0.0, 10_000_000.0),
        ),
    ),
    normalizer=MinMaxNormalizer(NormalizationScope.FIXED),
    scalarizer=WeightedMeanScalarizer({"quality": 0.7, "cycles": 0.3}),
)
```

`ObjectiveRuntime` es interno y exclusivo de la sesion. Los valores objetivos se
registran mediante `StatisticsCollector.on_evaluated_batch`; cada algoritmo conserva
su propia semantica de `best_individuals()` y `ObjectiveSet` no analiza el historico.

## Diagnóstico

Cada comando conserva stdout y stderr en:

```text
<base_work_dir>/<individual>/xheep_verilator_simulation/logs/
```

También se generan:

- `xheep_overlay_metadata.json`
- `xheep_run_metadata.json`
- `xheep_simulation.json`

El checkout aislado queda en `xheep_verilator_simulation/xheep`. Para comprobar que se
ejecuta el firmware correcto, `sw/build` debe seguir siendo un enlace simbólico y el
log de UART debe contener etiquetas `GENIO_*`, no la salida de `hello_world`.
