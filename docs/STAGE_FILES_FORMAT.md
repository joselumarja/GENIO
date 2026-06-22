# Formato real de archivos de stages

Este documento describe el formato **actual** de los JSON de stages en este repositorio.

Hay dos familias de archivos:

1. `definitions`: contrato conceptual de una etapa.
2. `implementations`: llamada concreta en un backend funcional o HLS.

Los tests en `search_space/tests/*.json` configuran escenarios concretos de búsqueda/evaluación y se mencionan solo para explicar cómo se consumen los campos.

---

## 1. Estructura de directorios

```text
search_space/stages/
├── definitions/
│   ├── base/
│   └── custom/
└── implementations/
    ├── functional/
    │   ├── opencv/
    │   ├── numpy/
    │   ├── scipy/
    │   ├── scikit_image/
    │   ├── scikit_learn/
    │   └── custom/
    └── hls/
        └── vitis_vision/
```

Las rutas dentro de `definitions[*].implementations` son relativas a:

```text
search_space/stages/
```

---

# 2. Definitions

Ubicación:

```text
search_space/stages/definitions/base/*.json
search_space/stages/definitions/custom/*.json
```

Una definición declara:

```text
id
name
category
description
parameters
constraints
interface
implementations
composer_resolved
```

No existe ya el campo `auxiliary_inputs`. Las entradas laterales reales se representan como entradas posicionales en `interface.inputs`. Los recursos embebidos/cargados por wrapper se representan como `parameters` si proceden del archivo de valores, o como `composer_resolved` si los resuelve el orquestador.

## 2.1 Ejemplo real de definition

Ejemplo reducido basado en la estructura actual:

```json
{
  "id": "remap",
  "name": "Remap",
  "category": "geometry",
  "description": "Apply generic geometrical remapping.",
  "parameters": [
    {
      "name": "interpolation",
      "token": "@INTERPOLATION",
      "type": "enum",
      "description": "Interpolation policy."
    }
  ],
  "interface": {
    "inputs": [
      {
        "format": "image",
        "token": "@input_0"
      },
      {
        "format": "image",
        "token": "@input_1"
      },
      {
        "format": "image",
        "token": "@input_2"
      }
    ],
    "outputs": [
      {
        "format": "image",
        "token": "@output_0"
      }
    ]
  },
  "implementations": {
    "hls": {
      "vitis_vision": "implementations/hls/vitis_vision/remap.json"
    },
    "functional": {
      "opencv": "implementations/functional/opencv/remap.json"
    }
  },
  "composer_resolved": [
    {
      "name": "rows",
      "token": "@ROWS",
      "description": "Maximum/current input image height."
    }
  ]
}
```

El orden de campos no tiene semántica.

---

## 2.2 `id`

Identificador estable de la etapa.

```json
"id": "black_level_correction"
```

Se usa en:

- tests, mediante `stage`;
- implementaciones, mediante `implementation.stage`;
- referencias internas del catálogo.

Convención actual: `snake_case`.

---

## 2.3 `name`

Nombre legible para humanos.

```json
"name": "Black Level Correction"
```

No se usa para enlazar archivos. Para enlazar se usa `id`.

---

## 2.4 `category`

Categoría descriptiva.

Ejemplos existentes o habituales:

| Categoría | Uso |
|---|---|
| `arithmetic` / `image_arithmetic` | Operaciones aritméticas entre imágenes o acumuladores. |
| `color_conversion` | Conversiones de color. |
| `edges` / `gradients` | Bordes y gradientes. |
| `filtering` | Filtros locales. |
| `geometry` | Transformaciones geométricas. |
| `isp` | Bloques de image signal processing. |
| `morphology` | Erosión/dilatación. |
| `statistics` | Histogramas, medias, min/max. |
| `stereo` | Disparidad y estéreo. |
| `pipeline_control` | Etapas especiales como `nop`. |

La categoría es informativa; no reemplaza a `interface`.

---

## 2.5 `description`

Descripción conceptual breve.

```json
"description": "Apply black level correction."
```

Debe describir la etapa, no una firma concreta de OpenCV/Vitis.

---

## 2.6 `parameters`

Lista de valores configurables por tests/searcher/archivo de valores.

Estructura:

```json
"parameters": [
  {
    "name": "black_level",
    "token": "@black_level",
    "type": "number",
    "description": "Black level offset."
  }
]
```

Campos:

| Campo | Significado |
|---|---|
| `name` | Nombre usado por los tests en `candidate.parameters`. |
| `token` | Placeholder usado en `implementation.function`, `configuration` y `tokens`. |
| `type` | Tipo conceptual. |
| `description` | Descripción humana. |

Regla práctica:

```text
tests configuran por name
implementations consumen por token
```

Ejemplo:

```text
Test:                         black_level
Definition parameter token:    @black_level
Implementation function token: @black_level
```

`token` no siempre es `@` + `name`. Hay parámetros con tokens adaptados a la convención HLS, por ejemplo:

```json
{
  "name": "interpolation",
  "token": "@INTERPOLATION"
}
```

Tipos conceptuales frecuentes:

| Tipo | Uso |
|---|---|
| `integer` | Tamaños, índices, iteraciones, umbrales enteros. |
| `number` / `float` | Escalas, sigmas, ganancias, offsets. |
| `boolean` | Flags. |
| `enum` | Modos o políticas. |
| `array` / `matrix` / `object` | Recursos o valores estructurados cargados como parámetros de etapa. |

---

## 2.7 `constraints`

Lista de expresiones booleanas que forman parte del contrato general de validez de la etapa.

Ejemplo:

```json
"constraints": [
  "@lower_0 <= @upper_0",
  "@lower_1 <= @upper_1",
  "@lower_2 <= @upper_2"
]
```

Las constraints se declaran en la definición porque expresan restricciones propias de la etapa, no de un escenario concreto de búsqueda.

Reglas actuales:

- son una lista de strings;
- usan tokens, no nombres de parámetros;
- los tokens deben corresponder a `parameters` de la misma definición;
- no deben referirse a `@input_N`, `@output_N` ni a `composer_resolved` salvo que se formalice una necesidad concreta.

### 2.7.1 Cuándo se aplican

`SearchSpace` aplica estas constraints al generar las alternativas concretas de cada slot. Para cada candidate del archivo de test, se expanden las combinaciones de parámetros y solo se añaden a la lista de alternativas aquellas que cumplen todas las constraints declaradas en la definition de la stage.

Por ejemplo, si un test define:

```json
"kernel_rows": {
  "type": "choice",
  "values": [3, 5, 7]
},
"kernel_cols": {
  "type": "choice",
  "values": [3, 5, 7]
}
```

y la definition contiene:

```json
"constraints": [
  "@K_ROWS == @K_COLS"
]
```

entonces se descartan combinaciones como:

```text
kernel_rows = 3, kernel_cols = 5
kernel_rows = 5, kernel_cols = 7
```

y solo quedan alternativas válidas como:

```text
kernel_rows = 3, kernel_cols = 3
kernel_rows = 5, kernel_cols = 5
kernel_rows = 7, kernel_cols = 7
```

Esto significa que el espacio de búsqueda construido por `SearchSpace` contiene únicamente alternativas válidas a nivel local de stage.

### 2.7.2 Tokens permitidos

Las constraints se escriben usando los tokens definidos en `parameters`.

Ejemplo:

```json
"parameters": [
  {
    "name": "kernel_rows",
    "token": "@K_ROWS",
    "type": "integer"
  },
  {
    "name": "kernel_cols",
    "token": "@K_COLS",
    "type": "integer"
  }
],
"constraints": [
  "@K_ROWS == @K_COLS"
]
```

La constraint usa:

```text
@K_ROWS
@K_COLS
```

no:

```text
kernel_rows
kernel_cols
```

Durante la expansión del espacio de búsqueda, `SearchSpace` sustituye cada token por el valor concreto de la combinación que está evaluando.

### 2.7.3 Comparaciones soportadas

Se soportan comparaciones básicas:

| Operador | Significado | Ejemplo |
|---|---|---|
| `==` | Igualdad | `@K_ROWS == @K_COLS` |
| `!=` | Desigualdad | `@mode != "disabled"` |
| `<` | Menor que | `@min_value < @max_value` |
| `<=` | Menor o igual | `@lower_0 <= @upper_0` |
| `>` | Mayor que | `@iterations > 0` |
| `>=` | Mayor o igual | `@threshold >= 0` |

Ejemplos válidos:

```text
@K_ROWS == @K_COLS
@lower_0 <= @upper_0
@iterations > 0
@mode != "unsupported"
```

También se soportan comparaciones encadenadas:

```text
@min_value <= @value <= @max_value
```

Aunque los límites absolutos suelen expresarse en los archivos de test mediante rangos, las comparaciones encadenadas pueden ser útiles cuando el límite depende de otros parámetros variables.

### 2.7.4 Operadores booleanos

Se soportan combinaciones con:

```text
and
or
```

Ejemplos:

```text
@K_ROWS == @K_COLS and @iterations > 0
@mode == "nearest" or @mode == "linear"
```

Para expresiones complejas, se recomienda usar paréntesis para hacer explícita la intención:

```text
(@mode == "nearest" or @mode == "linear") and @scale > 0
```

### 2.7.5 Aritmética soportada

Se soportan operaciones aritméticas simples:

| Operador | Significado | Ejemplo |
|---|---|---|
| `+` | Suma | `@x + @width <= @max_width` |
| `-` | Resta | `abs(@a - @b) <= 2` |
| `*` | Multiplicación | `@rows * @cols <= 921600` |
| `/` | División | `@width / @scale >= 1` |
| `%` | Módulo | `@K_ROWS % 2 == 1` |

También se soportan números negativos mediante signo unario:

```text
@offset >= -10
```

Ejemplos válidos:

```text
@OUT_ROWS * @OUT_COLS <= 921600
@K_ROWS % 2 == 1
@x + @width <= @image_width
```

### 2.7.6 Pertenencia

Se soportan expresiones de pertenencia con:

```text
in
not in
```

Y literales de colección:

```text
listas
tuplas
sets
```

Ejemplos válidos:

```text
@mode in ["nearest", "linear", "area"]
@mode not in ("unsupported", "disabled")
@K_ROWS in {3, 5, 7}
```

Esto permite expresar constraints compactas para modos discretos o valores permitidos dependientes de la stage.

Ejemplo en JSON:

```json
"constraints": [
  "@INTERPOLATION in [\"nearest\", \"linear\", \"area\"]"
]
```

Si los valores permitidos son simples y no dependen de otros parámetros, normalmente es mejor expresarlos en el archivo de test con `type: "choice"`. La pertenencia es más útil cuando la restricción depende de la definition o combina varios parámetros.

### 2.7.7 Literales soportados

Se soportan literales simples:

```text
enteros
floats
strings
booleans
None
```

Ejemplos:

```text
@threshold >= 0
@sigma <= 2.5
@mode == "binary"
@enabled == True
```

También se soportan colecciones literales para pertenencia:

```text
["nearest", "linear"]
("nearest", "linear")
{"nearest", "linear"}
```

### 2.7.8 Funciones matemáticas seguras

Se permiten llamadas directas únicamente a funciones incluidas en una whitelist. No se permite ejecutar Python arbitrario.

Funciones matemáticas soportadas:

| Función | Significado | Ejemplo |
|---|---|---|
| `abs(x)` | Valor absoluto | `abs(@a - @b) <= 2` |
| `min(a, b, ...)` | Mínimo | `min(@rows, @cols) >= 128` |
| `max(a, b, ...)` | Máximo | `max(@rows, @cols) <= 1920` |
| `round(x)` | Redondeo Python estándar | `round(@scale) == 2` |

Ejemplos válidos:

```text
abs(@target - @actual) <= 4
min(@OUT_ROWS, @OUT_COLS) >= 128
max(@OUT_ROWS, @OUT_COLS) <= 1920
round(@scale) == @scale
```

### 2.7.9 Funciones específicas de dominio

Además de las funciones matemáticas básicas, se soportan funciones de dominio para hacer las constraints más expresivas.

| Función | Significado | Ejemplo |
|---|---|---|
| `is_odd(x)` | `True` si `x` es impar. | `is_odd(@K_ROWS)` |
| `is_even(x)` | `True` si `x` es par. | `is_even(@OUT_COLS)` |
| `is_power_of_two(x)` | `True` si `x` es potencia de dos positiva. | `is_power_of_two(@NPC)` |
| `divisible_by(x, divisor)` | `True` si `x` es divisible por `divisor`. Si `divisor` es `0`, devuelve `False`. | `divisible_by(@OUT_COLS, 8)` |
| `square(rows, cols)` | `True` si `rows == cols`. | `square(@K_ROWS, @K_COLS)` |

Ejemplos válidos:

```text
is_odd(@K_ROWS)
is_odd(@K_COLS)
square(@K_ROWS, @K_COLS)
divisible_by(@OUT_COLS, 8)
is_power_of_two(@NPC)
```

La constraint actual de `erode` puede escribirse de estas dos formas equivalentes:

```text
@K_ROWS == @K_COLS
square(@K_ROWS, @K_COLS)
```

Para kernels que deban ser cuadrados e impares, podría usarse:

```text
square(@K_ROWS, @K_COLS) and is_odd(@K_ROWS)
```

### 2.7.10 Sintaxis no soportada

El evaluador de constraints es intencionadamente limitado. No se soporta Python completo.

No están permitidos:

```text
imports
acceso a atributos
llamadas a métodos
indexación
lambdas
comprensiones
asignaciones
funciones no incluidas en la whitelist
argumentos por keyword
```

Ejemplos no válidos:

```text
__import__("os").system("...")
@mode.lower() == "linear"
@shape[0] == @shape[1]
[x for x in values]
lambda x: x > 0
sum([@a, @b]) == 3
within(@value, 0, 10)
```

`within(...)` no está soportada actualmente. Si se necesita, debería añadirse explícitamente a la whitelist y documentarse en esta sección.

### 2.7.11 Recomendaciones de uso

Usar `constraints` para reglas generales de validez de la stage:

```text
kernel cuadrado
límites inferiores no mayores que límites superiores
alineamientos requeridos
relaciones entre parámetros
modos incompatibles
```

No usar `constraints` para reducir manualmente un espacio que ya puede expresarse mejor en el archivo de test. Por ejemplo, si un parámetro solo puede tomar tres valores independientes, preferir:

```json
"type": "choice",
"values": [3, 5, 7]
```

en vez de:

```text
@K_ROWS in [3, 5, 7]
```

Usar `constraints` cuando la validez depende de la relación entre dos o más parámetros:

```text
@lower_0 <= @upper_0
square(@K_ROWS, @K_COLS)
divisible_by(@OUT_COLS, @TILE_COLS)
```

No hacer que las constraints dependan de detalles de implementación concreta si la regla no pertenece al contrato conceptual de la stage. Las restricciones específicas de backend deberían resolverse posteriormente en el composer o en validadores de backend.

Ejemplos actuales:

```text
in_range:
  @lower_0 <= @upper_0
  @lower_1 <= @upper_1
  @lower_2 <= @upper_2

erode:
  @K_ROWS == @K_COLS
```

---

## 2.8 `interface`

Declara entradas y salidas de datos reales de la etapa.

Estructura actual:

```json
"interface": {
  "inputs": [
    {
      "format": "image",
      "token": "@input_0"
    },
    {
      "format": "image",
      "token": "@input_1"
    }
  ],
  "outputs": [
    {
      "format": "image",
      "token": "@output_0"
    }
  ]
}
```

Los puertos usan solo información estructural. No se usan en puertos:

```text
name
role
semantic_name
description
compatibility
```

### Tokens de puertos

Los tokens se indexan por posición:

```text
inputs[0]  -> @input_0
inputs[1]  -> @input_1
outputs[0] -> @output_0
outputs[1] -> @output_1
```

El primer input suele ser el flujo principal. Inputs posteriores representan flujos laterales reales, por ejemplo máscara o mapas de remapeo, pero sin campo adicional que los etiquete.

Ejemplos:

| Stage | Inputs |
|---|---|
| `remap` | `@input_0` imagen, `@input_1` mapa X, `@input_2` mapa Y. |
| `paint_mask` | `@input_0` imagen, `@input_1` máscara. |
| `add` | `@input_0`, `@input_1`. |

### `format`

Formatos usados o esperados:

| Formato | Significado |
|---|---|
| `image` | Imagen/matriz de píxeles. |
| `array` | Vector, histograma, descriptor o buffer. |
| `matrix` | Matriz matemática. |
| `object` | Objeto/modelo. |

---

## 2.9 `implementations`

Mapa de implementaciones disponibles.

```json
"implementations": {
  "hls": {
    "vitis_vision": "implementations/hls/vitis_vision/black_level_correction.json"
  },
  "functional": {
    "numpy": "implementations/functional/numpy/black_level_correction.json"
  }
}
```

Primer nivel:

| Clave | Significado |
|---|---|
| `functional` | Implementación de referencia en software. |
| `hls` | Implementación orientada a Vitis/HLS. |

Segundo nivel habitual:

```text
opencv
numpy
scipy
scikit_image
scikit_learn
custom
vitis_vision
```

---

## 2.10 `composer_resolved`

Valores que no configura el archivo de valores/test, sino el orquestador/composer.

```json
"composer_resolved": [
  {
    "name": "rows",
    "token": "@ROWS",
    "description": "Maximum/current input image height."
  },
  {
    "name": "type",
    "token": "@TYPE",
    "description": "Vitis Vision data type inferred from the pipeline."
  }
]
```

Ejemplos frecuentes:

| Token | Significado |
|---|---|
| `@ROWS`, `@COLS` | Dimensiones máximas/efectivas. |
| `@TYPE`, `@OUT_TYPE`, `@DST_T` | Tipos Vitis Vision. |
| `@NPC`, `@OUT_NPC` | Píxeles por ciclo. |
| `@XFCVDEPTH_*` | Profundidades internas de `xf::cv::Mat`/streams. |
| `@USE_URAM`, `@USE_DSP` | Decisiones de implementación. |
| `@MUL_VALUE`, `@MAXCOLORS`, `@FILTER_TYPE` | Valores derivados o específicos de la primitiva/wrapper. |

Estos tokens pueden depender de:

- backend seleccionado;
- dimensiones y tipo de imagen;
- restricciones de Vitis Vision;
- parámetros configurables;
- decisiones de wrapper/composer.

---

## 2.11 `nop`

`nop` representa ausencia de operación.

Debe interpretarse como:

```text
no generar llamada funcional
no generar llamada HLS
pasar al siguiente slot
```

---

# 3. Implementations

Ubicación:

```text
search_space/stages/implementations/functional/<backend>/*.json
search_space/stages/implementations/hls/<backend>/*.json
```

Campos comunes:

```text
stage
function
configuration
tokens
source
```

Campos específicos:

```text
imports   usado en implementaciones funcionales
include   usado en implementaciones HLS
```

---

## 3.1 Ejemplo funcional real

Basado en `black_level_correction`:

```json
{
  "stage": "black_level_correction",
  "imports": [
    "import numpy as np"
  ],
  "function": [
    "maxval = np.iinfo(@input_0.dtype).max if np.issubdtype(@input_0.dtype, np.integer) else 1.0",
    "scale = maxval / max(maxval - @black_level, 1e-6)",
    "corrected = (@input_0.astype(np.float32) - @black_level) * scale",
    "@output_0 = np.clip(corrected, 0, maxval).astype(@input_0.dtype)"
  ],
  "configuration": [
    "@black_level"
  ],
  "tokens": {
    "@black_level": "Black level offset provided by the concrete test configuration.",
    "@input_0": "Input data from the previous pipeline step.",
    "@output_0": "Output generated by this stage."
  },
  "source": "numpy"
}
```

---

## 3.2 Ejemplo HLS real

Basado en `black_level_correction`:

```json
{
  "stage": "black_level_correction",
  "include": "imgproc/xf_black_level.hpp",
  "function": [
    "xf::cv::blackLevelCorrection<@TYPE, @ROWS, @COLS, @NPC, @MUL_VALUE_WIDTH, @FL_POS, @USE_DSP, @XFCVDEPTH_IN, @XFCVDEPTH_OUT>(@input_0, @output_0, @black_level, @MUL_VALUE);"
  ],
  "configuration": [
    "@black_level"
  ],
  "tokens": {
    "@TYPE": "Vitis Vision input/output pixel type selected by the composition layer.",
    "@ROWS": "Maximum image height used to specialize the hardware function.",
    "@COLS": "Maximum image width used to specialize the hardware function.",
    "@NPC": "Pixels processed per clock cycle selected by the composition layer.",
    "@input_0": "Input stream or matrix generated by the previous pipeline step.",
    "@output_0": "Output stream or matrix consumed by the next pipeline step.",
    "@black_level": "Black level offset provided by the concrete test configuration.",
    "@MUL_VALUE": "Q16.16 gain value computed from black_level and image bit depth."
  },
  "source": "vision/L1/include/imgproc/xf_black_level.hpp"
}
```

`source` es una ruta simplificada. Para localizar el archivo real se antepone:

```text
Vitis_Libraries/
```

Ejemplo:

```text
source: vision/L1/include/imgproc/xf_black_level.hpp
archivo real: Vitis_Libraries/vision/L1/include/imgproc/xf_black_level.hpp
```

---

## 3.3 `stage`

Debe coincidir con `definition.id`.

```json
"stage": "black_level_correction"
```

---

## 3.4 `imports`

Solo en implementaciones funcionales.

```json
"imports": [
  "import cv2 as cv"
]
```

---

## 3.5 `include`

Solo en implementaciones HLS.

```json
"include": "imgproc/xf_black_level.hpp"
```

Corresponde normalmente a un header bajo:

```text
Vitis_Libraries/vision/L1/include/
```

---

## 3.6 `function`

Lista ordenada de líneas de código o llamada HLS.

Reglas actuales:

- Todos los tokens usados en `function` deben estar en `tokens`.
- Todo token usado en `function` debe poder explicarse desde la definition como uno de:
  1. puerto en `interface.inputs`/`interface.outputs`;
  2. `parameters`;
  3. `composer_resolved`.
- Las variables locales no deben llevar `@`.

Ejemplo correcto:

```json
"function": [
  "ksize = 3",
  "@output_0 = cv.Sobel(@input_0, cv.CV_16S, 1, 0, ksize=ksize)"
]
```

`ksize` es local; `@input_0` y `@output_0` son puertos.

---

## 3.7 `configuration`

Lista de tokens configurables consumidos directamente por `function`.

```json
"configuration": [
  "@black_level"
]
```

Debe contener solo tokens procedentes de `definitions.parameters` y usados directamente en `function`.

No debe incluir:

```text
@input_N
@output_N
tokens composer_resolved
variables locales
tokens no usados en function
```

---

## 3.8 `tokens`

Diccionario documental de tokens usados por la implementación.

```json
"tokens": {
  "@input_0": "Input image from the previous pipeline step.",
  "@output_0": "Output generated by this stage.",
  "@black_level": "Black level offset."
}
```

Debe incluir todos los tokens usados en `function`.

`tokens` no define origen ni configurabilidad. El origen se determina cruzando con la definition:

| Origen | Ejemplo |
|---|---|
| Puertos | `@input_0`, `@input_1`, `@output_0` |
| Parámetros | `@black_level`, `@INTERPOLATION`, `@sigma` |
| Composer | `@ROWS`, `@COLS`, `@TYPE`, `@NPC`, `@MUL_VALUE` |

---

## 3.9 `source`

Indica origen de la implementación.

Funcional:

```json
"source": "opencv"
```

HLS:

```json
"source": "vision/L1/include/imgproc/xf_black_level.hpp"
```

Las rutas HLS no incluyen el prefijo `Vitis_Libraries/`.

---

# 4. Relación con tests

Los tests configuran por `parameter.name`, no por `parameter.token`.

Definition:

```json
{
  "name": "lower_0",
  "token": "@lower_0"
}
```

Test:

```json
"parameters": {
  "lower_0": {
    "type": "integer_range",
    "start": 0,
    "stop": 60,
    "step": 10
  }
}
```

Implementation:

```json
"function": [
  "... @lower_0 ..."
]
```

---

## 4.1 Tipos usados en tests

Los tests usan tipos de configuración de búsqueda, no los mismos tipos conceptuales de `definitions.parameters`.

Ejemplos:

```json
"lower_0": {
  "type": "integer_range",
  "start": 0,
  "stop": 60,
  "step": 10
}
```

```json
"threshold_type": {
  "type": "choice",
  "values": ["binary", "binary_inv"]
}
```

```json
"maxval": {
  "type": "constant",
  "value": 255
}
```

Cada parámetro configurable en tests debe declararse explícitamente con su propio espacio de valores. Las constraints generales pertenecen a las definiciones de stages.

---

## 4.2 Recursos de wrapper en tests

Algunos valores cargados para una etapa pueden aparecer en tests como `wrapper_inputs`.

Ejemplo:

```json
"wrapper_inputs": {
  "lut": {
    "type": "file_ref",
    "id": "default_lut"
  }
}
```

Estos no son `interface.inputs`: no representan streams conectados entre etapas, sino recursos que el wrapper/composer materializa para una llamada.

---

# 5. Validaciones recomendadas

1. `implementation.stage` debe existir como `definition.id`.
2. Toda ruta en `definition.implementations` debe existir.
3. Todo token usado en `implementation.function` debe aparecer en `implementation.tokens`.
4. Todo token usado en `implementation.function` debe pertenecer a una de estas fuentes de su definition:
   - `interface.inputs` / `interface.outputs`;
   - `parameters`;
   - `composer_resolved`.
5. `configuration` debe contener solo tokens de `parameters` usados directamente por `function`.
6. `configuration` no debe contener puertos ni tokens `composer_resolved`.
7. Los puertos de `interface` deben usar tokens indexados por posición.
8. Los tests deben configurar solo nombres existentes en `definitions.parameters`.
9. Los tests no deben usar configuración `derived`.
10. Las rutas HLS en `source` deben ser relativas a `Vitis_Libraries/`.

---

# 6. Resumen

`definitions` responden:

```text
¿Qué etapa existe, qué parámetros expone, qué puertos tiene y qué debe resolver el composer?
```

`implementations` responden:

```text
¿Cómo se invoca esa etapa en un backend concreto?
```

`tests` responden:

```text
¿Qué combinaciones concretas de stages, parámetros y restricciones se quieren explorar?
```
