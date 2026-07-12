# Formato de archivos de configuración de parámetros

Este documento describe el formato actual de los archivos JSON usados para definir espacios de búsqueda, escenarios de prueba y valores de parámetros.

Actualmente estos archivos están en:

```text
search_space/tests/*.json
```

Aunque la carpeta se llama `tests`, estos JSON no son tests unitarios de Python. Son configuraciones de pipelines candidatos: declaran qué etapas pueden aparecer en cada posición y qué valores puede tomar cada parámetro.

---

## 1. Estructura general

Un archivo de configuración tiene esta forma:

```json
{
  "id": "color_threshold_pipeline",
  "description": "Example search space for color conversion and range thresholding.",
  "resources": {},
  "design_spaces": {
    "hls": {
      "pipeline_ii": {
        "type": "choice",
        "values": [1, 2]
      }
    },
    "system": {
      "memory_size": {
        "type": "choice",
        "values": [32768, 65536]
      }
    }
  },
  "pipeline": [
    {
      "slot": 0,
      "candidates": [
        {
          "stage": "bgr_to_hsv"
        },
        {
          "stage": "rgb_to_hsv"
        }
      ]
    },
    {
      "slot": 1,
      "candidates": [
        {
          "stage": "in_range",
          "parameters": {}
        }
      ]
    }
  ]
}
```

Campos de primer nivel:

| Campo | Obligatorio | Significado |
|---|---:|---|
| `id` | Sí | Identificador del escenario de configuración. |
| `description` | No | Descripción humana del escenario. |
| `resources` | No | Recursos externos disponibles para wrappers o valores cargados. |
| `design_spaces` | No | Espacios de decisión globales separados por dominio, por ejemplo `hls` o `system`. |
| `pipeline` | Sí | Lista ordenada de slots del pipeline. |

---

## 2. Campo `id`

Identificador del escenario.

```json
"id": "resize_filter_pipeline"
```

Debe ser estable y descriptivo. Se usa para referenciar el escenario completo, no una etapa concreta.

---

## 3. Campo `description`

Texto descriptivo.

```json
"description": "Example search space for resizing followed by optional smoothing."
```

No tiene efecto en la generación de candidatos.

---

## 4. Campo `resources`

Declara recursos externos reutilizables por el escenario.

Ejemplo real:

```json
"resources": {
  "files": {
    "default_lut": "assets/luts/default_lut.npy"
  },
  "objects": {
    "svm_model": "models/default_svm.pkl"
  }
}
```

Subcampos actuales:

| Campo | Significado |
|---|---|
| `files` | Mapa de identificadores a rutas de archivo. |
| `objects` | Mapa de identificadores a objetos serializados o modelos. |

Los recursos no se conectan como `interface.inputs`. Se usan como valores cargados por wrappers o por el composer.

---

## 5. Campo `design_spaces`

`design_spaces` declara decisiones globales que forman parte del mismo espacio de búsqueda que el pipeline, pero no pertenecen a una stage concreta.

Ejemplo:

```json
"design_spaces": {
  "hls": {
    "pipeline_ii": {
      "type": "choice",
      "values": [1, 2, 4]
    },
    "array_partition": {
      "type": "choice",
      "values": ["none", "cyclic", "complete"]
    }
  },
  "system": {
    "memory_size": {
      "type": "choice",
      "values": [32768, 65536, 131072]
    },
    "mcu": {
      "type": "choice",
      "values": ["x-heep-small", "x-heep-big"]
    }
  }
}
```

Cada clave de primer nivel dentro de `design_spaces` es un dominio. Los dominios actuales recomendados son:

| Dominio | Uso |
|---|---|
| `hls` | Parámetros globales para síntesis HLS, como II objetivo, estrategia general de particionado o formato de salida. |
| `system` | Parámetros de una plataforma o sistema externo, por ejemplo memoria disponible, tipo de MCU o configuración X-HEEP. |

Los dominios tienen objetivos distintos. Un step de síntesis HLS debe consumir `Individual.design["hls"]`; un step de evaluación de sistema debe consumir `Individual.design["system"]`; un step funcional de imagen puede ignorar ambos.

Los parámetros dentro de cada dominio usan los mismos tipos de configuración finitos que `candidate.parameters`: `constant`, `choice`, `integer_range` y `number_range`. `file_ref` y `object_ref` están pensados para `wrapper_inputs`, no para decisiones globales de diseño.

El individuo generado mantiene estos valores separados:

```python
individual.design == {
    "hls": {
        "pipeline_ii": 1,
        "array_partition": "complete",
    },
    "system": {
        "memory_size": 65536,
        "mcu": "x-heep-small",
    },
}
```

El genotype y el `search_index` sí representan el espacio completo:

```text
genes de pipeline + genes de hls + genes de system + ...
```

---

## 6. Campo `pipeline`

Lista ordenada de slots.

```json
"pipeline": [
  {
    "slot": 0,
    "candidates": []
  },
  {
    "slot": 1,
    "candidates": []
  }
]
```

Cada slot representa una posición lógica en el pipeline. El buscador/orquestador selecciona un candidato de cada slot para construir una configuración concreta.

---

## 7. Slots

Un slot tiene esta forma:

```json
{
  "slot": 1,
  "candidates": [
    {
      "stage": "threshold"
    },
    {
      "stage": "otsu_threshold"
    }
  ]
}
```

Campos:

| Campo | Significado |
|---|---|
| `slot` | Índice entero de posición dentro del pipeline. |
| `candidates` | Lista de etapas alternativas para esa posición. |

La lista `pipeline` ya tiene orden, pero `slot` hace explícito el índice lógico.

---

## 8. Candidates

Un candidato representa una etapa concreta que puede ocupar un slot.

Forma mínima:

```json
{
  "stage": "nop"
}
```

Forma completa:

```json
{
  "stage": "in_range",
  "parameters": {},
  "wrapper_inputs": {}
}
```

Campos:

| Campo | Obligatorio | Significado |
|---|---:|---|
| `stage` | Sí | `id` de una definition existente. |
| `parameters` | No | Configuración de valores para `definitions.parameters`. |
| `wrapper_inputs` | No | Recursos o valores cargados por wrapper/composer. |

---

## 9. Campo `stage`

Referencia a una etapa definida en:

```text
search_space/stages/definitions/base/*.json
search_space/stages/definitions/custom/*.json
```

Ejemplo:

```json
{
  "stage": "gaussian_filter"
}
```

Debe coincidir con `definition.id`, no con `definition.name`.

---

## 10. Campo `parameters`

Mapa de nombre de parámetro a especificación de valores.

Ejemplo:

```json
"parameters": {
  "threshold": {
    "type": "integer_range",
    "start": 40,
    "stop": 220,
    "step": 20
  },
  "maxval": {
    "type": "constant",
    "value": 255
  }
}
```

Las claves de `parameters` usan `definition.parameters[*].name`, no tokens.

Ejemplo:

Definition:

```json
{
  "name": "filter_width",
  "token": "@FILTER_WIDTH"
}
```

Config:

```json
"parameters": {
  "filter_width": {
    "type": "choice",
    "values": [3, 5, 7]
  }
}
```

La implementación usará después el token correspondiente, por ejemplo `@FILTER_WIDTH`.

---

# 11. Tipos de configuración de parámetros

Los tipos usados en estos archivos no son los mismos que `definitions.parameters[*].type`.

En definitions:

```text
integer, number, enum, array, matrix, object...
```

En archivos de configuración:

```text
constant, choice, integer_range, number_range, file_ref, object_ref...
```

---

## 11.1 `constant`

Valor fijo.

```json
"maxval": {
  "type": "constant",
  "value": 255
}
```

Uso:

- cuando un parámetro no debe explorarse;
- para fijar valores auxiliares simples;
- para mantener constante una configuración en todos los candidatos generados.

El campo principal es:

| Campo | Significado |
|---|---|
| `value` | Valor concreto. Puede ser número, string, booleano, `null`, array u objeto JSON. |

---

## 11.2 `choice`

Conjunto finito de opciones.

```json
"interpolation": {
  "type": "choice",
  "values": ["nearest", "linear", "area"]
}
```

Uso:

- enumerar modos;
- elegir tamaños discretos;
- probar políticas de borde o interpolación.

Campo principal:

| Campo | Significado |
|---|---|
| `values` | Lista de valores posibles. |

---

## 11.3 `integer_range`

Rango discreto de enteros.

```json
"lower_0": {
  "type": "integer_range",
  "start": 0,
  "stop": 60,
  "step": 10
}
```

Campos:

| Campo | Significado |
|---|---|
| `start` | Primer valor del rango. |
| `stop` | Límite superior según la semántica que defina el generador. |
| `step` | Incremento. |

El proyecto debe fijar si `stop` es inclusivo o exclusivo en el generador definitivo. Los ejemplos actuales lo usan como especificación de espacio, no como ejecución directa de `range()` documentada formalmente.

---

## 11.4 `number_range`

Rango numérico para valores reales.

```json
"sigma": {
  "type": "number_range",
  "start": 0.5,
  "stop": 2.0,
  "step": 0.5
}
```

Uso:

- sigmas;
- escalas;
- pesos;
- offsets reales.

Campos:

| Campo | Significado |
|---|---|
| `start` | Valor inicial. |
| `stop` | Límite superior según semántica del generador. |
| `step` | Incremento. |

---

## 11.5 Parámetros compuestos

Si una etapa necesita un vector pequeño o una pareja de valores, cada componente se configura como un parámetro independiente.

Ejemplo para el rango de un histograma:

```json
"range_start": {
  "type": "constant",
  "value": 0
},
"range_end": {
  "type": "constant",
  "value": 256
}
```

La implementación puede reconstruir localmente la estructura que necesita la librería:

```python
ranges = [@range_start, @range_end]
```

Esta regla mantiene visible cada componente y permite que las restricciones generales de la etapa se declaren en su definition.

---

## 11.6 `file_ref`

Referencia a un recurso declarado en `resources.files`.

Ejemplo:

```json
"resources": {
  "files": {
    "default_lut": "assets/luts/default_lut.npy"
  }
}
```

Uso en wrapper:

```json
"wrapper_inputs": {
  "lut": {
    "type": "file_ref",
    "id": "default_lut"
  }
}
```

Campos:

| Campo | Significado |
|---|---|
| `id` | Clave existente en `resources.files`. |

---

## 11.7 `object_ref`

Referencia a un recurso declarado en `resources.objects`.

Ejemplo:

```json
"resources": {
  "objects": {
    "svm_model": "models/default_svm.pkl"
  }
}
```

Uso:

```json
"wrapper_inputs": {
  "model": {
    "type": "object_ref",
    "id": "svm_model"
  }
}
```

Campos:

| Campo | Significado |
|---|---|
| `id` | Clave existente en `resources.objects`. |

---

# 12. Campo `wrapper_inputs`

`wrapper_inputs` declara valores que no son parámetros simples de la etapa ni flujos conectados como `interface.inputs`, sino recursos que un wrapper/composer debe materializar.

Ejemplo real:

```json
{
  "stage": "lut",
  "wrapper_inputs": {
    "lut": {
      "type": "file_ref",
      "id": "default_lut"
    }
  }
}
```

Otro ejemplo:

```json
{
  "stage": "svm",
  "wrapper_inputs": {
    "model": {
      "type": "object_ref",
      "id": "svm_model"
    }
  }
}
```

Diferencia con `parameters`:

```text
parameters
  Valores configurables declarados en definitions.parameters.

wrapper_inputs
  Recursos externos o artefactos que el wrapper necesita cargar o convertir.
```

Diferencia con `interface.inputs`:

```text
interface.inputs
  Streams/datos conectados en el pipeline.

wrapper_inputs
  Recursos no conectados como flujo de datos principal/lateral.
```

---

# 13. Ejemplo completo

```json
{
  "id": "color_threshold_pipeline",
  "description": "Example search space for color conversion and range thresholding.",
  "design_spaces": {
    "hls": {
      "pipeline_ii": {
        "type": "choice",
        "values": [1, 2]
      }
    },
    "system": {
      "memory_size": {
        "type": "choice",
        "values": [32768, 65536]
      }
    }
  },
  "pipeline": [
    {
      "slot": 0,
      "candidates": [
        {
          "stage": "bgr_to_hsv"
        },
        {
          "stage": "rgb_to_hsv"
        }
      ]
    },
    {
      "slot": 1,
      "candidates": [
        {
          "stage": "in_range",
          "parameters": {
            "lower_0": {
              "type": "integer_range",
              "start": 0,
              "stop": 60,
              "step": 10
            },
            "upper_0": {
              "type": "integer_range",
              "start": 20,
              "stop": 80,
              "step": 10
            }
          }
        }
      ]
    }
  ]
}
```

---

# 14. Validaciones recomendadas

Para cada archivo de configuración:

1. `id` debe existir y ser único.
2. `pipeline` debe ser una lista.
3. Cada slot debe tener `slot` y `candidates`.
4. Cada candidate debe tener `stage`.
5. `stage` debe existir como `definition.id`.
6. Cada clave de `candidate.parameters` debe existir como `definition.parameters[*].name`.
7. `design_spaces`, si existe, debe ser un mapa de dominios a mapas de parámetros.
8. Cada dominio de `design_spaces` debe tener un nombre estable, por ejemplo `hls` o `system`.
9. Cada parámetro de `design_spaces` debe usar un tipo de configuración soportado para espacios finitos.
10. Los `file_ref` deben apuntar a claves existentes en `resources.files`.
11. Los `object_ref` deben apuntar a claves existentes en `resources.objects`.
12. `wrapper_inputs` no debe confundirse con `interface.inputs`.
13. Si un candidato es `nop`, no necesita `parameters` ni `wrapper_inputs`.

---

# 15. Resumen

Estos archivos responden a:

```text
¿Qué pipeline o familia de pipelines se quiere explorar?
¿Qué etapas puede ocupar cada slot?
¿Qué valores puede tomar cada parámetro?
¿Qué decisiones globales de HLS o sistema forman parte del diseño?
¿Qué recursos externos necesita el wrapper/composer?
```

La separación con respecto a los archivos de stages es:

```text
definitions
  Declaran qué parámetros existen.

implementations
  Declaran cómo se usan los tokens en cada backend.

configuration/test files
  Declaran qué valores concretos o espacios de valores se exploran.
```
