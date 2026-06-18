# Auditoría de stages: funcionalidad y correspondencia HLS/Python

Informe estático generado desde `search_space/stages`.

## `absolute_difference`

**Funcionalidad:** Compute absolute per-pixel difference between two images.

**Categoría:** `arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::absdiff<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `absdiff` alrededor de línea 619:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void absdiff(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src1,
             xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& _src2,
             xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    uint16_t image_width = _src1.cols >> XF_BITSHIFT(NPC);
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16SC1) || (SRC_T == XF_16SC3)) &&
           "Image type must be XF_8UC1 or XF_8UC3, XF_16SC1, XF_16SC3");
    assert(((_src1.rows <= ROWS) && (_src1.cols <= COLS) && (_src2.rows <= ROWS) && (_src2.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.absdiff(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `accumulate`

**Funcionalidad:** Accumulate one image into an accumulator image.

**Categoría:** `image_arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_accumulate_image.hpp`
- Source: `vision/L1/include/imgproc/xf_accumulate_image.hpp`
- Function:
```cpp
xf::cv::accumulate<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_accumulate_image.hpp`, firma detectada para `accumulate` alrededor de línea 98:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void accumulate(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
                xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst) {
#ifndef __SYNTHESIS__
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((src2.rows <= ROWS) && (src2.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((dst.rows <= ROWS) && (dst.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((src1.rows == src2.rows) && (src1.cols == src2.cols)) && "Both input images should have same size");
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1, XF_NPPC2, XF_NPPC4,XF_NPPC8 ");
#endif
    uint16_t lwidth = src2.cols >> XF_BITSHIFT(NPC);

// clang-format off
    #pragma HLS INLINE OFF
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv; import numpy as np`
- Function:
```python
@output_0 = @input_1.copy()
cv.accumulate(@input_0, @output_0)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `accumulate_squared`

**Funcionalidad:** Accumulate squared image values.

**Categoría:** `image_arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_accumulate_squared.hpp`
- Source: `vision/L1/include/imgproc/xf_accumulate_squared.hpp`
- Function:
```cpp
xf::cv::accumulateSquare<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_accumulate_squared.hpp`, firma detectada para `accumulateSquare` alrededor de línea 97:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void accumulateSquare(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                      xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) &&
           "Input TYPE must be XF_8UC1 for 1-channel and XF_8UC3 for 3-channel image");
    assert(((DST_T == XF_16UC1) || (DST_T == XF_16UC3)) &&
           "Output TYPE must be XF_16UC1 for 1-channel and XF_16UC3 for 3-channel image");
    assert(((src1.rows == src2.rows) && (src1.cols == src2.cols)) && "Both input images should have same size");
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1, XF_NPPC2, XF_NPPC4,XF_NPPC8 ");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv; import numpy as np`
- Function:
```python
@output_0 = @input_1.copy()
cv.accumulateSquare(@input_0, @output_0)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `accumulate_weighted`

**Funcionalidad:** Compute weighted accumulation of an image.

**Categoría:** `image_arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `alpha` / `@alpha` (`number`): First scale or weight factor.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_accumulate_weighted.hpp`
- Source: `vision/L1/include/imgproc/xf_accumulate_weighted.hpp`
- Function:
```cpp
xf::cv::accumulateWeighted<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @alpha);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @alpha
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_accumulate_weighted.hpp`, firma detectada para `accumulateWeighted` alrededor de línea 99:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void accumulateWeighted(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
                        xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst,
                        float alpha) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) &&
           "Input TYPE must be XF_8UC1 for 1-channel and XF_8UC3 for 3-channel image");
    assert(((DST_T == XF_16UC1) || (DST_T == XF_16UC3)) &&
           "Output TYPE must be XF_16UC1 for 1-channel and XF_16UC3 for 3-channel image");
    assert(((src1.rows == src2.rows) && (src1.cols == src2.cols)) && "Both input images should have same size");
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1, XF_NPPC2, XF_NPPC4,XF_NPPC8 ");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv; import numpy as np`
- Function:
```python
@output_0 = @input_1.copy()
cv.accumulateWeighted(@input_0, @output_0, @alpha)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @alpha
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `add`

**Funcionalidad:** Compute per-pixel image addition.

**Categoría:** `arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::add<@POLICY_TYPE, @TYPE, @ROWS, @COLS, @NPC, @XFCVDEPTH_IN_1, @XFCVDEPTH_IN_2, @XFCVDEPTH_OUT_1>(@input_0, @input_1, @output_0);
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @POLICY_TYPE, @ROWS, @TYPE, @XFCVDEPTH_IN_1, @XFCVDEPTH_IN_2, @XFCVDEPTH_OUT_1
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `add` alrededor de línea 866:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void add(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src1,
         xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& _src2,
         xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    uint16_t image_width = _src1.cols >> XF_BITSHIFT(NPC);
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16SC1) || (SRC_T == XF_16SC3)) &&
           "Image type must be XF_8UC1 or XF_8UC3, XF_16SC1, XF_16SC3");
    assert((POLICY_TYPE == XF_CONVERT_POLICY_SATURATE || POLICY_TYPE == XF_CONVERT_POLICY_TRUNCATE) &&
           "_policytype must be 'XF_CONVERT_POLICY_SATURATE' or 'XF_CONVERT_POLICY_TRUNCATE'");
    assert((_src1.rows <= ROWS) && "ROWS and COLS should be greater than input image");
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.add(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `add_weighted`

**Funcionalidad:** Compute weighted sum of two images.

**Categoría:** `image_arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `alpha` / `@alpha` (`number`): First scale or weight factor.
- `beta` / `@beta` (`number`): Second scale or weight factor.
- `gamma` / `@gamma` (`number`): Bias term.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_add_weighted.hpp`
- Source: `vision/L1/include/imgproc/xf_add_weighted.hpp`
- Function:
```cpp
xf::cv::addWeighted<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @input_1, @output_0, @alpha, @beta, @gamma);
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @alpha, @beta, @gamma
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_add_weighted.hpp`, firma detectada para `addWeighted` alrededor de línea 112:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void addWeighted(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                 float alpha,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
                 float beta,
                 float gama,
                 xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) &&
           "Input TYPE must be XF_8UC1 for 1-channel, XF_8UC3 for 3-channel");
    assert(((DST_T == XF_8UC1) || (DST_T == XF_8UC3)) &&
           "Output TYPE must be XF_8UC1 for 1-channel,XF_8UC3 for 3-channel ");
    assert(((src1.rows == src2.rows) && (src1.cols == src2.cols)) && "Both input images should have same size");
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4,XF_NPPC8 ");
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.addWeighted(@input_0, @alpha, @input_1, @beta, @gamma)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @alpha, @beta, @gamma
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `aec`

**Funcionalidad:** Apply auto exposure correction.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_aec.hpp`
- Source: `vision/L1/include/imgproc/xf_aec.hpp`
- Function:
```cpp
xf::cv::autoexposurecorrection<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_aec.hpp`, firma detectada para `autoexposurecorrection` alrededor de línea 107:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int USE_URAM = 0,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void autoexposurecorrection(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& src,
                            xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& dst,
                            uint32_t hist_array1[1][256],
                            uint32_t hist_array2[1][256]) {
#pragma HLS INLINE OFF

    int rows = src.rows;
    int cols = src.cols;

    uint16_t cols_shifted = cols >> (XF_BITSHIFT(NPC));
    uint16_t rows_shifted = rows;

    xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN> bgr2hsv(rows, cols);

    xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN> hsvimg1(rows, cols);
    xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN> hsvimg2(rows, cols);
```

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- No hay implementación funcional declarada.

## `agc`

**Funcionalidad:** Apply auto gain control.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_agc.hpp`
- Source: `vision/L1/include/imgproc/xf_agc.hpp`
- Function:
```cpp
xf::cv::auto_gain_control<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `auto_gain_control` en `Vitis_Libraries/vision/L1/include/imgproc/xf_agc.hpp`.

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.
- No hay implementación funcional declarada.

## `auto_white_balance`

**Funcionalidad:** Apply automatic white balance.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_autowhitebalance.hpp`
- Source: `vision/L1/include/imgproc/xf_autowhitebalance.hpp`
- Function:
```cpp
xf::cv::AWB<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `AWB` en `Vitis_Libraries/vision/L1/include/imgproc/xf_autowhitebalance.hpp`.

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
means = np.mean(@input_0, axis=(0, 1), keepdims=True)
gray = np.mean(means)
@output_0 = np.clip(@input_0.astype(np.float32) * (gray / np.maximum(means, 1e-6)), 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `awb_normalization`

**Funcionalidad:** Normalize image for AWB pipeline.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_awb_norm.hpp`
- Source: `vision/L1/include/imgproc/xf_awb_norm.hpp`
- Function:
```cpp
xf::cv::AWBNormalization<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `AWBNormalization` en `Vitis_Libraries/vision/L1/include/imgproc/xf_awb_norm.hpp`.

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(@input_0.astype(np.float32) / np.maximum(np.max(@input_0), 1e-6) * @maxval, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `bad_pixel_correction`

**Funcionalidad:** Correct defective pixels.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `kernel_size` / `@kernel_size` (`integer`): Kernel size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_bpc.hpp`
- Source: `vision/L1/include/imgproc/xf_bpc.hpp`
- Function:
```cpp
xf::cv::BPC<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `BPC` en `Vitis_Libraries/vision/L1/include/imgproc/xf_bpc.hpp`.

**Implementación funcional/Python:**
- Backend: `scipy`
- Source: `scipy`
- Imports: `from scipy import ndimage`
- Function:
```python
@output_0 = ndimage.median_filter(@input_0, size=kernel_size)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `bfmatcher`

**Funcionalidad:** Brute-force descriptor matching.

**Categoría:** `features`

- Inputs: @input_0:array, @input_1:array
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_bfmatcher.hpp`
- Source: `vision/L1/include/imgproc/xf_bfmatcher.hpp`
- Function:
```cpp
xf::cv::bfMatcher<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_bfmatcher.hpp`, firma detectada para `bfMatcher` alrededor de línea 95:
```cpp
            dist_min_2 = local_dist;
        }
    }
};

_GENERIC_BF_TPLT
void bfMatcher(DESC_TYPE desc_list_q[MAX_KEYPOINTS],
               DESC_TYPE desc_list_t[MAX_KEYPOINTS],
               MATCH_TYPE match_list[MAX_KEYPOINTS],
               ap_uint<32> num_keypoints_q,
               ap_uint<32> num_keypoints_t,
               float ratio_thresh) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on

#ifndef __SYNTHESIS__
    assert((num_keypoints_q <= MAX_KEYPOINTS) &&
           "Number of keypoints in the descriptor query set must be less than the MAX_KEYPOINTS parameter");
    assert((num_keypoints_t <= MAX_KEYPOINTS) &&
           "Number of keypoints in the descriptor training set must be less than the MAX_KEYPOINTS parameter");
#endif
```

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `scikit-image`
- Imports: `from skimage.feature import match_descriptors`
- Function:
```python
@output_0 = match_descriptors(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bgr_to_gray`

**Funcionalidad:** Convert BGR image to grayscale.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cvt_color.hpp`
- Source: `vision/L1/include/imgproc/xf_cvt_color.hpp`
- Function:
```cpp
xf::cv::bgr2gray<XF_8UC3, XF_8UC1, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color.hpp`, firma detectada para `bgr2gray` alrededor de línea 5610:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void bgr2gray(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC3) && " BGR image Type must be XF_8UC3");
    assert((DST_T == XF_8UC1) && " GRAY image Type must be XF_8UC1");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && " BGR image rows and cols should be less than ROWS, COLS");
    assert(((_dst.cols == _src.cols) && (_dst.rows == _src.rows)) && "BGR and GRAY plane dimensions mismatch");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && " 1,8 pixel parallelism is supported  ");
#endif
    xfbgr2gray<SRC_T, DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_T, NPC),
               XF_WORDWIDTH(DST_T, NPC), (ROWS * (COLS >> (XF_NPIXPERCYCLE(NPC))))>(_src, _dst, _src.rows, _src.cols);
}

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_BGR2GRAY)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bgr_to_hsv`

**Funcionalidad:** Convert BGR image to HSV.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_bgr2hsv.hpp`
- Source: `vision/L1/include/imgproc/xf_bgr2hsv.hpp`
- Function:
```cpp
xf::cv::bgr2hsv<XF_8UC3, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_bgr2hsv.hpp`, firma detectada para `bgr2hsv` alrededor de línea 78:
```cpp
template <int SRC_T,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void bgr2hsv(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
             xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    int hdiv[256] = {
        0,    122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192,
        7680, 7228,   6827,  6467,  6144,  5851,  5585,  5343,  5120,  4915,  4726,  4551,  4389,  4237, 4096, 3964,
        3840, 3724,   3614,  3511,  3413,  3321,  3234,  3151,  3072,  2997,  2926,  2858,  2793,  2731, 2671, 2614,
        2560, 2508,   2458,  2409,  2363,  2318,  2276,  2234,  2194,  2156,  2119,  2083,  2048,  2014, 1982, 1950,
        1920, 1890,   1862,  1834,  1807,  1781,  1755,  1731,  1707,  1683,  1661,  1638,  1617,  1596, 1575, 1555,
        1536, 1517,   1499,  1480,  1463,  1446,  1429,  1412,  1396,  1381,  1365,  1350,  1336,  1321, 1307, 1293,
        1280, 1267,   1254,  1241,  1229,  1217,  1205,  1193,  1182,  1170,  1159,  1148,  1138,  1127, 1117, 1107,
        1097, 1087,   1078,  1069,  1059,  1050,  1041,  1033,  1024,  1016,  1007,  999,   991,   983,  975,  968,
        960,  953,    945,   938,   931,   924,   917,   910,   904,   897,   890,   884,   878,   871,  865,  859,
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_BGR2HSV)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bgr_to_rgb`

**Funcionalidad:** Convert BGR image to RGB.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cvt_color.hpp`
- Source: `vision/L1/include/imgproc/xf_cvt_color.hpp`
- Function:
```cpp
xf::cv::bgr2rgb<XF_8UC3, XF_8UC3, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color.hpp`, firma detectada para `bgr2rgb` alrededor de línea 8302:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void bgr2rgb(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
             xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC3) && " BGR image Type must be XF_8UC3");
    assert((DST_T == XF_8UC3) && " RGB image Type must be XF_8UC3");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && " BGR image rows and cols should be less than ROWS, COLS");
    assert(((_dst.cols == _src.cols) && (_dst.rows == _src.rows)) && "BGR and RGB plane dimensions mismatch");

    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           " 1,2,4,8 pixel parallelism is supported  ");
#endif
    xfbgr2rgb<SRC_T, DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_T, NPC),
              XF_WORDWIDTH(DST_T, NPC), (ROWS * (COLS >> (XF_NPIXPERCYCLE(NPC)))), XF_NPIXPERCYCLE(NPC)>(
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_BGR2RGB)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bilateral_filter`

**Funcionalidad:** Apply edge-preserving bilateral filtering.

**Categoría:** `filtering`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `diameter` / `@diameter` (`integer`): Bilateral filter window diameter.
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.
- `sigma_color` / `@sigma_color` (`number`): Bilateral filter sigmaColor.
- `sigma_space` / `@sigma_space` (`number`): Bilateral filter sigmaSpace.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_bilateral_filter.hpp`
- Source: `vision/L1/include/imgproc/xf_bilateral_filter.hpp`
- Function:
```cpp
xf::cv::bilateralFilter<@diameter, @BORDER_TYPE, @TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @sigma_color, @sigma_space);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @diameter, @sigma_color, @sigma_space
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_bilateral_filter.hpp`, firma detectada para `bilateralFilter` alrededor de línea 616:
```cpp
          int TYPE,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void bilateralFilter(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
                     xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat,
                     float sigma_color,
                     float sigma_space) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xFbilateralFilterKernel<TYPE, ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, XFCVDEPTH_IN_1, XFCVDEPTH_OUT_1,
                            (TYPE << (XF_BITSHIFT(NPC))), WINDOW_SIZE>(_src_mat, _dst_mat, BORDER_TYPE, _src_mat.rows,
                                                                       _src_mat.cols, sigma_color, sigma_space);
}
} // namespace cv
} // namespace xf
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.bilateralFilter(@input_0, @diameter, @sigma_color, @sigma_space, borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @diameter, @sigma_color, @sigma_space
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bitwise_and`

**Funcionalidad:** Compute per-pixel bitwise AND.

**Categoría:** `arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::bitwise_and<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `bitwise_and` alrededor de línea 647:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void bitwise_and(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src1,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& _src2,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst) {
// clang-format off
    #pragma HLS inline off
// clang-format on
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16SC1) || (SRC_T == XF_16SC3)) &&
           "Image type must be XF_8UC1 or XF_8UC3, XF_16SC1, XF_16SC3");

    assert(((_src1.rows <= ROWS) && (_src1.cols <= COLS) && (_src2.rows <= ROWS) && (_src2.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
#endif
    uint16_t image_width = _src1.cols >> XF_BITSHIFT(NPC);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.bitwise_and(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bitwise_not`

**Funcionalidad:** Compute per-pixel bitwise NOT.

**Categoría:** `arithmetic`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::bitwise_not<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `bitwise_not` alrededor de línea 703:
```cpp
template <int SRC_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void bitwise_not(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst) {
    //	assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) &&
    //			"NPC must be XF_NPPC1 or XF_NPPC8 ");
    //	assert(((SRC_T == XF_8UC1) ) &&
    //			"Image type must be XF_8UC1 ");
    //	assert(((src.rows <= ROWS ) && (src.cols <= COLS) ) && "ROWS and COLS should be greater than input image");

    uint16_t image_width = src.cols >> XF_BITSHIFT(NPC);
// clang-format off
    #pragma HLS inline off
    // clang-format on

    xFBitwiseNOTKernel<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC, XFCVDEPTH_IN_1,
                       XFCVDEPTH_OUT_1, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(
        src, dst, src.rows, image_width);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.bitwise_not(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bitwise_or`

**Funcionalidad:** Compute per-pixel bitwise OR.

**Categoría:** `arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::bitwise_or<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `bitwise_or` alrededor de línea 676:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void bitwise_or(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src1,
                xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& _src2,
                xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16SC1) || (SRC_T == XF_16SC3)) &&
           "Image type must be XF_8UC1 or XF_8UC3, XF_16SC1, XF_16SC3");
    assert(((_src1.rows <= ROWS) && (_src1.cols <= COLS) && (_src2.rows <= ROWS) && (_src2.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
#endif
    uint16_t image_width = _src1.cols >> XF_BITSHIFT(NPC);

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.bitwise_or(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bitwise_xor`

**Funcionalidad:** Compute per-pixel bitwise XOR.

**Categoría:** `arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::bitwise_xor<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `bitwise_xor` alrededor de línea 728:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void bitwise_xor(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst) {
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16SC1) || (SRC_T == XF_16SC3)) &&
           "Image type must be XF_8UC1 or XF_8UC3, XF_16SC1, XF_16SC3");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS) && (src2.rows <= ROWS) && (src2.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
#endif
// clang-format off
    #pragma HLS inline off
    // clang-format on

    uint16_t image_width = src1.cols >> XF_BITSHIFT(NPC);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.bitwise_xor(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `black_level_correction`

**Funcionalidad:** Apply black level correction.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `black_level` / `@black_level` (`number`): Black level offset.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_black_level.hpp`
- Source: `vision/L1/include/imgproc/xf_black_level.hpp`
- Function:
```cpp
xf::cv::blackLevelCorrection<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_black_level.hpp`, firma detectada para `blackLevelCorrection` alrededor de línea 76:
```cpp
          int NPPC = XF_NPPC1,
          int MUL_VALUE_WIDTH = 16,
          int FL_POS = 15,
          int USE_DSP = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void blackLevelCorrection(xf::cv::Mat<SRC_T, _MAX_ROWS, _MAX_COLS, NPPC, XFCVDEPTH_IN>& _Src,
                          xf::cv::Mat<SRC_T, _MAX_ROWS, _MAX_COLS, NPPC, XFCVDEPTH_OUT>& _Dst,
                          unsigned short black_level,
                          int mul_value // ap_uint<MUL_VALUE_WIDTH> mul_value
                          ) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    // max/(max-black)

    const uint32_t _TC = _MAX_ROWS * (_MAX_COLS >> XF_BITSHIFT(NPPC));

    const int STEP = XF_DTPIXELDEPTH(SRC_T, NPPC);

    uint32_t LoopCount = _Src.rows * (_Src.cols >> XF_BITSHIFT(NPPC));
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(@input_0.astype(np.int32) - @black_level, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @black_level, @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `bounding_box`

**Funcionalidad:** Compute bounding boxes for detected regions.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_boundingbox.hpp`
- Source: `vision/L1/include/imgproc/xf_boundingbox.hpp`
- Function:
```cpp
xf::cv::boundingbox<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_boundingbox.hpp`, firma detectada para `boundingbox` alrededor de línea 157:
```cpp
            _src_mat.write(i * width + (c_new[b] - 1), color_box[b]);
        }
    }
}

template <int SRC_T, int ROWS, int COLS, int MAX_BOXES = 1, int NPC = 1, int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void boundingbox(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
                 xf::cv::Rect_<int>* roi,
                 xf::cv::Scalar<4, unsigned char>* color,
                 int num_box) {
    unsigned short width = _src_mat.cols;
    unsigned short height = _src_mat.rows;
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) || (SRC_T == XF_8UC4) || (SRC_T == XF_16UC1) ||
           (SRC_T == XF_16UC4) && "Type must be XF_8UC1 or XF_8UC4");
    assert((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) ||
           (NPC == XF_NPPC8) && "NPC must be 1, Multipixel parallelism is not supported");

    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");

    for (int i = 0; i < num_box; i++) {
        assert(((roi[i].height <= height) && (roi[i].width <= width)) &&
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.boundingRect(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `box_filter`

**Funcionalidad:** Apply normalized box filtering.

**Categoría:** `filtering`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `kernel_rows` / `@K_ROWS` (`integer`): Kernel height.
- `kernel_cols` / `@K_COLS` (`integer`): Kernel width.
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_box_filter.hpp`
- Source: `vision/L1/include/imgproc/xf_box_filter.hpp`
- Function:
```cpp
xf::cv::boxFilter<@BORDER_TYPE, @TYPE, @ROWS, @COLS, @NPC, @K_ROWS, @K_COLS>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @K_COLS, @K_ROWS
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_box_filter.hpp`, firma detectada para `boxFilter` alrededor de línea 1261:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM = false,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void boxFilter(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((FILTER_TYPE == XF_FILTER_3X3) || (FILTER_TYPE == XF_FILTER_5X5) || (FILTER_TYPE == XF_FILTER_7X7)) &&
           ("Filter width should be 3 or 5 or 7."));
    assert(BORDER_TYPE == XF_BORDER_CONSTANT && "Only XF_BORDER_CONSTANT is supported");

    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    uint16_t img_width = _src_mat.cols >> XF_BITSHIFT(NPC);
    uint16_t img_height = _src_mat.rows;

    if (FILTER_TYPE == XF_FILTER_3X3) {
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.boxFilter(@input_0, -1, (@K_COLS, @K_ROWS), borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @K_COLS, @K_ROWS
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `canny`

**Funcionalidad:** Apply Canny edge detection.

**Categoría:** `edges`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `low_threshold` / `@low_threshold` (`integer`): Lower hysteresis threshold.
- `high_threshold` / `@high_threshold` (`integer`): Upper hysteresis threshold.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_canny.hpp`
- Source: `vision/L1/include/imgproc/xf_canny.hpp`
- Function:
```cpp
xf::cv::Canny<@FILTER_TYPE, @NORM_TYPE, @TYPE, @OUT_TYPE, @ROWS, @COLS, @NPC, @OUT_NPC>(@input_0, @output_0, @low_threshold, @high_threshold);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @high_threshold, @low_threshold
- Tokens resueltos por composer: @COLS, @FILTER_TYPE, @NORM_TYPE, @NPC, @OUT_NPC, @OUT_TYPE, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_canny.hpp`, firma detectada para `Canny` alrededor de línea 288:
```cpp
          int COLS,
          int NPC,
          int NPC1,
          bool USE_URAM = false,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void Canny(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC1, XFCVDEPTH_OUT_1>& _dst_mat,
           unsigned char _lowthreshold,
           unsigned char _highthreshold) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((NORM_TYPE == XF_L1NORM) || (NORM_TYPE == XF_L2NORM)) &&
           "The _norm_type must be 'XF_L1NORM' or'XF_L2NORM'");
#endif

    if (NORM_TYPE == 1) {
        xFCannyKernel<SRC_T, DST_T, NORM_TYPE, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC1), NPC, NPC1,
                      XFCVDEPTH_IN_1, XFCVDEPTH_OUT_1, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC1),
                      (COLS >> XF_BITSHIFT(NPC)), 2, ((COLS >> XF_BITSHIFT(NPC)) * 3), FILTER_TYPE, USE_URAM>(
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.Canny(@input_0, @low_threshold, @high_threshold)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @high_threshold, @low_threshold
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `canny_sobel`

**Funcionalidad:** Compute Sobel stage used by Canny edge detection.

**Categoría:** `edges`

- Inputs: @input_0:image
- Outputs: @output_0:image, @output_1:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_canny_sobel.hpp`
- Source: `vision/L1/include/imgproc/xf_canny_sobel.hpp`
- Function:
```cpp
xf::cv::CannySobel<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `CannySobel` en `Vitis_Libraries/vision/L1/include/imgproc/xf_canny_sobel.hpp`.

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.
- No hay implementación funcional declarada.

## `channel_combine`

**Funcionalidad:** Combine single-channel images into a multi-channel image.

**Categoría:** `channels`

- Inputs: @input_0:image, @input_1:image, @input_2:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_channel_combine.hpp`
- Source: `vision/L1/include/imgproc/xf_channel_combine.hpp`
- Function:
```cpp
xf::cv::merge<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_channel_combine.hpp`, firma detectada para `merge` alrededor de línea 223:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void merge(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src1,
           xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& _src2,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst) {
#ifndef __SYNTHESIS__
    assert(((_src1.rows <= ROWS) && (_src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_src2.rows <= ROWS) && (_src2.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_dst.rows <= ROWS) && (_dst.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((SRC_T == XF_8UC1) && (DST_T == XF_8UC2) &&
           "Source image should be of 1 channel and destination image of 2 "
           "channels");
//    assert(((NPC == XF_NPPC1)) && "NPC must be XF_NPPC1");
#endif

// clang-format off
    #pragma HLS inline off
    // clang-format on
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.merge([@input_0, @input_1, @input_2])
```
- Tokens de puertos: @input_0, @input_1, @input_2, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `channel_extract`

**Funcionalidad:** Extract one channel from a multi-channel image.

**Categoría:** `channels`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `channel` / `@channel` (`integer`): Channel index.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_channel_extract.hpp`
- Source: `vision/L1/include/imgproc/xf_channel_extract.hpp`
- Function:
```cpp
xf::cv::extractChannel<@TYPE, @OUT_TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @channel);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @channel
- Tokens resueltos por composer: @COLS, @NPC, @OUT_TYPE, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_channel_extract.hpp`, firma detectada para `extractChannel` alrededor de línea 114:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void extractChannel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
                    xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat,
                    uint16_t _channel) {
#ifndef __SYNTHESIS__
    assert(((_channel == XF_EXTRACT_CH_0) || (_channel == XF_EXTRACT_CH_1) || (_channel == XF_EXTRACT_CH_2) ||
            (_channel == XF_EXTRACT_CH_3) || (_channel == XF_EXTRACT_CH_R) || (_channel == XF_EXTRACT_CH_G) ||
            (_channel == XF_EXTRACT_CH_B) || (_channel == XF_EXTRACT_CH_A) || (_channel == XF_EXTRACT_CH_Y) ||
            (_channel == XF_EXTRACT_CH_U) || (_channel == XF_EXTRACT_CH_V)) &&
           "Invalid Channel Value. See xf_channel_extract_e enumerated type");
    assert(!(((_channel == XF_EXTRACT_CH_A) || (_channel == XF_EXTRACT_CH_3)) &&
             (SRC_T == XF_8UC3 || SRC_T == XF_16UC3)) &&
           "Invalid Channel Value & Input Type combination");
    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_dst_mat.rows <= ROWS) && (_dst_mat.cols <= COLS)) && "ROWS and COLS should be greater than output image");
    assert((((SRC_T == XF_8UC4 || SRC_T == XF_8UC3) && (DST_T == XF_8UC1)) ||
            ((SRC_T == XF_16UC3 || SRC_T == XF_16UC4) && (DST_T == XF_16UC1)) ||
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.extractChannel(@input_0, @channel)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @channel
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `clahe`

**Funcionalidad:** Apply contrast limited adaptive histogram equalization.

**Categoría:** `enhancement`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `clip_limit` / `@clip_limit` (`number`): CLAHE contrast clipping limit.
- `tile_grid_size` / `@tile_grid_size` (`array`): CLAHE tile grid size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_clahe.hpp`
- Source: `vision/L1/include/imgproc/xf_clahe.hpp`
- Function:
```cpp
xf::cv::clahe<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `clahe` en `Vitis_Libraries/vision/L1/include/imgproc/xf_clahe.hpp`.

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
clahe = cv.createCLAHE(clipLimit=@clip_limit, tileGridSize=@tile_grid_size)
@output_0 = clahe.apply(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @clip_limit, @tile_grid_size
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `color_convert_extended`

**Funcionalidad:** Extended color conversion operation.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cvt_color_1.hpp`
- Source: `vision/L1/include/imgproc/xf_cvt_color_1.hpp`
- Function:
```cpp
xf::cv::cvtColor<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `cvtColor` en `Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color_1.hpp`.

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.
- No hay implementación funcional declarada.

## `color_correction_matrix`

**Funcionalidad:** Apply color correction matrix.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `bias` / `@bias` (`array`): Bias vector.
- `matrix` / `@matrix` (`matrix`): Transformation/calibration matrix supplied as stage value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_colorcorrectionmatrix.hpp`
- Source: `vision/L1/include/imgproc/xf_colorcorrectionmatrix.hpp`
- Function:
```cpp
xf::cv::colorcorrectionmatrix<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_colorcorrectionmatrix.hpp`, firma detectada para `colorcorrectionmatrix` alrededor de línea 173:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void colorcorrectionmatrix(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
                           xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat,
                           signed int ccm_matrix[3][3],
                           signed int offsetarray[3]) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;
    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");

// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    xfccmkernel<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XFCVDEPTH_IN_1, XFCVDEPTH_OUT_1, XF_WORDWIDTH(SRC_T, NPC),
                XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(_src_mat, _dst_mat, ccm_matrix, offsetarray,
                                                                      height, width);
}
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(@input_0 @ @matrix + @bias, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @bias, @matrix, @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `color_thresholding`

**Funcionalidad:** Apply multi-channel color thresholding.

**Categoría:** `thresholding`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `low_0` / `@low_0` (`integer`): Lower threshold for channel 0.
- `low_1` / `@low_1` (`integer`): Lower threshold for channel 1.
- `low_2` / `@low_2` (`integer`): Lower threshold for channel 2.
- `high_0` / `@high_0` (`integer`): Upper threshold for channel 0.
- `high_1` / `@high_1` (`integer`): Upper threshold for channel 1.
- `high_2` / `@high_2` (`integer`): Upper threshold for channel 2.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_colorthresholding.hpp`
- Source: `vision/L1/include/imgproc/xf_colorthresholding.hpp`
- Function:
```cpp
unsigned char low_thresh[3] = {@low_0, @low_1, @low_2};
unsigned char high_thresh[3] = {@high_0, @high_1, @high_2};
xf::cv::colorthresholding<XF_8UC3, XF_8UC1, @MAXCOLORS, @ROWS, @COLS, @NPC>(@input_0, @output_0, low_thresh, high_thresh);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @high_0, @high_1, @high_2, @low_0, @low_1, @low_2
- Tokens resueltos por composer: @COLS, @MAXCOLORS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_colorthresholding.hpp`, firma detectada para `colorthresholding` alrededor de línea 133:
```cpp
          int MAXCOLORS,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void colorthresholding(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
                       xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat,
                       unsigned char low_thresh[MAXCOLORS * 3],
                       unsigned char high_thresh[MAXCOLORS * 3]) {
// clang-format off
  #pragma HLS INLINE OFF
  #pragma HLS DATAFLOW
    // clang-format on

    unsigned char low_th[MAXCOLORS][3], high_th[MAXCOLORS][3];
// clang-format off
  #pragma HLS ARRAY_PARTITION variable = low_th dim = 1 complete
  #pragma HLS ARRAY_PARTITION variable = high_th dim = 1 complete
    // clang-format on
    uint16_t j = 0;
    for (uint16_t i = 0; i < (MAXCOLORS); i++) {
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv; import numpy as np`
- Function:
```python
low_thresh = np.array([@low_0, @low_1, @low_2])
high_thresh = np.array([@high_0, @high_1, @high_2])
@output_0 = cv.inRange(@input_0, low_thresh, high_thresh)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @high_0, @high_1, @high_2, @low_0, @low_1, @low_2
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `connected_components`

**Funcionalidad:** Label connected components in a binary image.

**Categoría:** `segmentation`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cca_custom.hpp`
- Source: `vision/L1/include/imgproc/xf_cca_custom.hpp`
- Function:
```cpp
xf::cv::connectedComponents<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `connectedComponents` en `Vitis_Libraries/vision/L1/include/imgproc/xf_cca_custom.hpp`.

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
num_labels, @output_0 = cv.connectedComponents(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `convert_bitdepth`

**Funcionalidad:** Convert image pixel bit depth.

**Categoría:** `type_conversion`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `dtype` / `@dtype` (`dtype`): Output data type.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_convert_bitdepth.hpp`
- Source: `vision/L1/include/core/xf_convert_bitdepth.hpp`
- Function:
```cpp
xf::cv::convertTo<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_convert_bitdepth.hpp`, firma detectada para `convertTo` alrededor de línea 127:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void convertTo(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat,
               ap_uint<4> _convert_type,
               int _shift) {
    // assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1 or XF_NPPC8 ");
    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_dst_mat.rows <= ROWS) && (_dst_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");

    assert((((_convert_type == XF_CONVERT_16U_TO_8U) || (_convert_type == XF_CONVERT_16S_TO_8U) ||
             (_convert_type == XF_CONVERT_32S_TO_8U) || (_convert_type == XF_CONVERT_32S_TO_16S) ||
             (_convert_type == XF_CONVERT_32S_TO_16U) || (_convert_type == XF_CONVERT_8U_TO_16U) ||
             (_convert_type == XF_CONVERT_8U_TO_16S) || (_convert_type == XF_CONVERT_8U_TO_32S) ||
             (_convert_type == XF_CONVERT_16U_TO_32S) || (_convert_type == XF_CONVERT_16S_TO_32S)) &&
            " conversion type is not valid "));

// clang-format off
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = @input_0.astype(@dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @dtype
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `convert_scale_abs`

**Funcionalidad:** Scale, take absolute value and convert pixel type.

**Categoría:** `type_conversion`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `alpha` / `@alpha` (`number`): First scale or weight factor.
- `beta` / `@beta` (`number`): Second scale or weight factor.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_convertscaleabs.hpp`
- Source: `vision/L1/include/imgproc/xf_convertscaleabs.hpp`
- Function:
```cpp
xf::cv::convertScaleAbs<@TYPE, @OUT_TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @alpha, @beta);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @alpha, @beta
- Tokens resueltos por composer: @COLS, @NPC, @OUT_TYPE, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_convertscaleabs.hpp`, firma detectada para `convertScaleAbs` alrededor de línea 100:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void convertScaleAbs(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst,
                     float scale,
                     float shift) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1)) && "Input TYPE must be XF_8UC1 for 1-channel ");

    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8) || (NPC == XF_NPPC4) || (NPC == XF_NPPC2)) &&
           "NPC must be XF_NPPC1,XF_NPPC4,XF_NPPC2,XF_NPPC8 ");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

    convertScaleAbsKernel<SRC_T, DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1, XFCVDEPTH_OUT_1, XF_CHANNELS(SRC_T, NPC),
                          XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.convertScaleAbs(@input_0, alpha=@alpha, beta=@beta)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @alpha, @beta
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `convert_to`

**Funcionalidad:** Convert image pixel type with optional scaling.

**Categoría:** `type_conversion`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `alpha` / `@alpha` (`number`): First scale or weight factor.
- `beta` / `@beta` (`number`): Second scale or weight factor.
- `dtype` / `@dtype` (`dtype`): Output data type.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_convertTo.hpp`
- Source: `vision/L1/include/imgproc/xf_convertTo.hpp`
- Function:
```cpp
xf::cv::convertTo<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `convertTo` en `Vitis_Libraries/vision/L1/include/imgproc/xf_convertTo.hpp`.

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = (@input_0 * @alpha + @beta).astype(@dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @alpha, @beta, @dtype
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `corner_img_to_list`

**Funcionalidad:** Convert corner image representation to corner list.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_corner_img_to_list.hpp`
- Source: `vision/L1/include/imgproc/xf_corner_img_to_list.hpp`
- Function:
```cpp
xf::cv::cornersImgToList<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_corner_img_to_list.hpp`, firma detectada para `cornersImgToList` alrededor de línea 30:
```cpp
template <unsigned int MAXCORNERSNO,
          unsigned int TYPE,
          unsigned int ROWS,
          unsigned int COLS,
          unsigned int NPC,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void cornersImgToList(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
                      unsigned int list[MAXCORNERSNO],
                      unsigned int* ncorners) {
    int cornerCount = 0;
    for (unsigned short i = 0; i < _src.rows; i++) {
        for (unsigned short j = 0; j < _src.cols; j++) {
// clang-format off
            #pragma HLS PIPELINE
            // clang-format on
            ap_uint<8> tempValue = _src.read(i * _src.cols + j);
            if (tempValue == 255 && cornerCount < MAXCORNERSNO) // value is 255 if there's a corner
            {
                ap_uint<32> point;
                point.range(31, 16) = i;
                point.range(15, 0) = j;

```

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- No hay implementación funcional declarada.

## `corner_update`

**Funcionalidad:** Update tracked corner locations.

**Categoría:** `features`

- Inputs: @input_0:array
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_corner_update.hpp`
- Source: `vision/L1/include/imgproc/xf_corner_update.hpp`
- Function:
```cpp
xf::cv::cornerUpdate<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_corner_update.hpp`, firma detectada para `cornerUpdate` alrededor de línea 31:
```cpp
template <unsigned int MAXCORNERSNO,
          unsigned int TYPE,
          unsigned int ROWS,
          unsigned int COLS,
          unsigned int NPC,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void cornerUpdate(unsigned long* list_fix,
                  unsigned int* list,
                  uint32_t nCorners,
                  xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& flow_vectors,
                  ap_uint<1> harris_flag) {
    unsigned int* flowvectorsDataPtr = (unsigned int*)flow_vectors.data;
    unsigned int list_flag_tmp;
    unsigned int row_num_fix = 0;
    unsigned int col_num_fix = 0;
    unsigned short row_num;
    unsigned short col_num;

    // reading the packed flow vectors
    unsigned int flowuv_tl;
    unsigned int flowuv_tr;
    unsigned int flowuv_bl;
```

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- No hay implementación funcional declarada.

## `crop`

**Funcionalidad:** Crop a rectangular image region.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `x` / `@x` (`integer`): Left coordinate of the crop region.
- `y` / `@y` (`integer`): Top coordinate of the crop region.
- `out_rows` / `@OUT_ROWS` (`integer`): Output image height.
- `out_cols` / `@OUT_COLS` (`integer`): Output image width.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_crop.hpp`
- Source: `vision/L1/include/imgproc/xf_crop.hpp`
- Function:
```cpp
xf::cv::Rect_<unsigned int> roi(@x, @y, @OUT_COLS, @OUT_ROWS);
xf::cv::crop<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, roi);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @OUT_COLS, @OUT_ROWS, @x, @y
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `Rect_` en `Vitis_Libraries/vision/L1/include/imgproc/xf_crop.hpp`.
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_crop.hpp`, firma detectada para `crop` alrededor de línea 301:
```cpp
          int ROWS,
          int COLS,
          int ARCH_TYPE = 0,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void crop(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
          xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat,
          xf::cv::Rect_<unsigned int>& roi) {
    unsigned short width = _src_mat.cols;
    unsigned short height = _src_mat.rows;
#ifndef __SYNTHESIS__
    // assert((SRC_T == XF_8UC1) ||(SRC_T == XF_8UC3)  && "Type must be XF_8UC1 or XF_8UC3");
    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((roi.height <= height) && (roi.width <= width)) &&
           "ROI dimensions should be smaller or equal to the input image");
    assert(((roi.height > 0) && (roi.width > 0)) && "ROI dimensions should be greater than 0");
    assert(((roi.height + roi.y <= height) && (roi.width + roi.x <= width)) && "ROI area exceeds the input image area");
#endif
    // Width in terms of NPPC
    width = _src_mat.cols >> XF_BITSHIFT(NPC);

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Function:
```python
@output_0 = @input_0[@y:@y + @OUT_ROWS, @x:@x + @OUT_COLS]
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @OUT_COLS, @OUT_ROWS, @x, @y
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `custom_bgr_to_y8`

**Funcionalidad:** Convert BGR image to 8-bit luma representation.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_custom_bgr2y8.hpp`
- Source: `vision/L1/include/imgproc/xf_custom_bgr2y8.hpp`
- Function:
```cpp
xf::cv::bgr2y8<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `bgr2y8` en `Vitis_Libraries/vision/L1/include/imgproc/xf_custom_bgr2y8.hpp`.

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(0.114 * @input_0[..., 0] + 0.587 * @input_0[..., 1] + 0.299 * @input_0[..., 2], 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `custom_convolution`

**Funcionalidad:** Apply custom convolution filter.

**Categoría:** `filtering`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_custom_convolution.hpp`
- Source: `vision/L1/include/imgproc/xf_custom_convolution.hpp`
- Function:
```cpp
xf::cv::filter2D<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_custom_convolution.hpp`, firma detectada para `filter2D` alrededor de línea 816:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void filter2D(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst_mat,
              short int filter[FILTER_HEIGHT * FILTER_WIDTH],
              unsigned char _shift) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    unsigned short img_width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short img_height = _src_mat.rows;

    short int lfilter[FILTER_HEIGHT][FILTER_WIDTH];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=lfilter complete dim=0
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.filter2D(@input_0, -1, kernel, borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `de_gamma`

**Funcionalidad:** Apply inverse gamma correction.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `gamma` / `@gamma` (`number`): Gamma value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_degamma.hpp`
- Source: `vision/L1/include/imgproc/xf_degamma.hpp`
- Function:
```cpp
xf::cv::degamma<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_degamma.hpp`, firma detectada para `degamma` alrededor de línea 162:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          int N,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void degamma(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& src,
             xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& dst,
             uint32_t params[3][N][3],
             unsigned short bayerp) {
#ifndef __SYNTHESIS__
    assert(((bayerp == XF_BAYER_BG) || (bayerp == XF_BAYER_GB) || (bayerp == XF_BAYER_GR) || (bayerp == XF_BAYER_RG)) &&
           ("Unsupported Bayer pattern. Use anyone among: "
            "XF_BAYER_BG;XF_BAYER_GB;XF_BAYER_GR;XF_BAYER_RG"));
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_14UC1) || (SRC_T == XF_16UC1)) &&
           "Input TYPE must be XF_8UC1 or XF_14UC1 or XF_16UC1");
    assert(((DST_T == XF_8UC1) || (DST_T == XF_14UC1) || (DST_T == XF_16UC1)) &&
           "OUTPUT TYPE must be XF_8UC1 or XF_14UC1 or XF_16UC1");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1, XF_NPPC2 ");
    assert((src.rows <= ROWS) && (src.cols <= COLS) && "ROWS and COLS should be greater than input image size ");
#endif
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
normalized = @input_0.astype(np.float32) / @maxval
@output_0 = np.clip(np.power(normalized, 1.0 / @gamma) * @maxval, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @gamma, @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `delay`

**Funcionalidad:** Delay image stream for pipeline alignment.

**Categoría:** `pipeline_control`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_delay.hpp`
- Source: `vision/L1/include/imgproc/xf_delay.hpp`
- Function:
```cpp
xf::cv::delayMat<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_delay.hpp`, firma detectada para `delayMat` alrededor de línea 35:
```cpp
          int SRC_T,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void delayMat(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src,
              xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    // clang-format off
  //  #pragma HLS dataflow
    // clang-format on

    hls::stream<XF_TNAME(SRC_T, NPC)> src;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst;

/********************************************************/

Read_yuyv_Loop:
```

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- No hay implementación funcional declarada.

## `demosaicing`

**Funcionalidad:** Convert Bayer image to color image.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `color_conversion` / `@COLOR_CONVERSION` (`enum`): Color conversion code.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_demosaicing.hpp`
- Source: `vision/L1/include/imgproc/xf_demosaicing.hpp`
- Function:
```cpp
xf::cv::demosaicing<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_demosaicing.hpp`, firma detectada para `demosaicing` alrededor de línea 164:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void demosaicing(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src_mat,
                 xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst_mat,
                 unsigned short& bformat) {
#ifndef __SYNTHESIS__
    assert(((bformat == XF_BAYER_BG) || (bformat == XF_BAYER_GB) || (bformat == XF_BAYER_GR) ||
            (bformat == XF_BAYER_RG)) &&
           ("Unsupported Bayer pattern. Use anyone among: "
            "XF_BAYER_BG;XF_BAYER_GB;XF_BAYER_GR;XF_BAYER_RG"));
    assert(((src_mat.rows <= ROWS) && (src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == 1) || (NPC == 2) || (NPC == 4)) && "Only 1, 2 and 4 pixel-parallelism are supported");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_10UC1) || (SRC_T == XF_12UC1) || (SRC_T == XF_14UC1) ||
            (SRC_T == XF_16UC1)) &&
           "Only 8, 10, 12 and 16 bit, single channel images are supported");
    assert(((DST_T == XF_8UC3) || (DST_T == XF_10UC3) || (DST_T == XF_12UC3) || (DST_T == XF_14UC3) ||
            (DST_T == XF_16UC3) || (DST_T == XF_8UC4) || (DST_T == XF_10UC4) || (DST_T == XF_12UC4) ||
            (DST_T == XF_16UC4)) &&
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, @COLOR_CONVERSION)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @COLOR_CONVERSION
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `demosaicing_runtime`

**Funcionalidad:** Runtime-configurable Bayer demosaicing.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `color_conversion` / `@COLOR_CONVERSION` (`enum`): Color conversion code.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_demosaicing_rt.hpp`
- Source: `vision/L1/include/imgproc/xf_demosaicing_rt.hpp`
- Function:
```cpp
xf::cv::demosaicing<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_demosaicing_rt.hpp`, firma detectada para `demosaicing` alrededor de línea 228:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void demosaicing(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src_mat,
                 xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst_mat,
                 unsigned short& bformat) {
#ifndef __SYNTHESIS__
    //    assert(((BFORMAT == XF_BAYER_BG) || (BFORMAT == XF_BAYER_GB) || (BFORMAT == XF_BAYER_GR) ||
    //           (BFORMAT == XF_BAYER_RG)) &&
    //           ("Unsupported Bayer pattern. Use anyone among: "
    //            "XF_BAYER_BG;XF_BAYER_GB;XF_BAYER_GR;XF_BAYER_RG"));
    assert(((src_mat.rows <= ROWS) && (src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == 1) || (NPC == 2) || (NPC == 4)) && "Only 1, 2 and 4 pixel-parallelism are supported");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_10UC1) || (SRC_T == XF_12UC1) || (SRC_T == XF_16UC1)) &&
           "Only 8, 10, 12 and 16 bit, single channel images are supported");
    assert(((DST_T == XF_8UC3) || (DST_T == XF_10UC3) || (DST_T == XF_12UC3) || (DST_T == XF_16UC3) ||
            (DST_T == XF_8UC4) || (DST_T == XF_10UC4) || (DST_T == XF_12UC4) || (DST_T == XF_16UC4)) &&
           "Only 8, 10, 12 and 16 bit, 3 and 4 channel images are supported");
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, @COLOR_CONVERSION)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @COLOR_CONVERSION
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `depth_3d`

**Funcionalidad:** Compute depth information from stereo/disparity data.

**Categoría:** `stereo`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_3ddepth.hpp`
- Source: `vision/L1/include/imgproc/xf_3ddepth.hpp`
- Function:
```cpp
xf::cv::compute3Ddepth<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `compute3Ddepth` en `Vitis_Libraries/vision/L1/include/imgproc/xf_3ddepth.hpp`.

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.
- No hay implementación funcional declarada.

## `dilate`

**Funcionalidad:** Apply morphological dilation.

**Categoría:** `morphology`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `kernel_shape` / `@K_SHAPE` (`enum`): Structuring element shape.
- `kernel_rows` / `@K_ROWS` (`integer`): Kernel height.
- `kernel_cols` / `@K_COLS` (`integer`): Kernel width.
- `iterations` / `@ITERATIONS` (`integer`): Number of morphology iterations.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_dilation.hpp`
- Source: `vision/L1/include/imgproc/xf_dilation.hpp`
- Function:
```cpp
xf::cv::dilate<@BORDER_TYPE, @TYPE, @ROWS, @COLS, @K_SHAPE, @K_ROWS, @K_COLS, @ITERATIONS, @NPC>(@input_0, @output_0, @kernel);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @ITERATIONS, @K_COLS, @K_ROWS, @K_SHAPE
- Tokens resueltos por composer: @BORDER_TYPE, @COLS, @NPC, @ROWS, @TYPE, @kernel
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_dilation.hpp`, firma detectada para `dilate` alrededor de línea 356:
```cpp
          int K_ROWS,
          int K_COLS,
          int ITERATIONS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void dilate(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src,
            xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst,
            unsigned char _kernel[K_ROWS * K_COLS]) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    unsigned short imgheight = _src.rows;
    unsigned short imgwidth = _src.cols;
#ifndef __SYNTHESIS__
    assert(BORDER_TYPE == XF_BORDER_CONSTANT && "Only XF_BORDER_CONSTANT is supported");
    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    //    FILE *fpx = fopen("xf_in.txt","w");
    //
    //           for(int i=0; i< 128;++i){
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
kernel = cv.getStructuringElement(@K_SHAPE, (@K_COLS, @K_ROWS))
@output_0 = cv.dilate(@input_0, kernel, iterations=@ITERATIONS)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @ITERATIONS, @K_COLS, @K_ROWS, @K_SHAPE
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `distance_transform`

**Funcionalidad:** Compute distance transform.

**Categoría:** `segmentation`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `dist_type` / `@dist_type` (`enum`): Distance transform type.
- `mask_size` / `@mask_size` (`integer`): Distance transform mask size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_distancetransform.hpp`
- Source: `vision/L1/include/imgproc/xf_distancetransform.hpp`
- Function:
```cpp
xf::cv::distanceTransform<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_distancetransform.hpp`, firma detectada para `distanceTransform` alrededor de línea 436:
```cpp
    };
};

// ======================================================================================

template <int IN_PTR, int FW_PTR, int ROWS, int COLS, int USE_URAM>
void distanceTransform(ap_uint<IN_PTR>* _src, float* _dst, ap_uint<FW_PTR>* _fw_pass, int rows, int cols) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    assert(((rows <= ROWS) && (cols <= COLS)) &&
           "ROWS and COLS must be greater or equal torows and cols respectively.");
    assert((IN_PTR == 8) &&
           "The input must be a grayscale image, encoded with "
           "binary values (0 or 255), which means the pointer "
           "width must be '8'.");
    assert((FW_PTR == 32) && "FW_PTR, is the forwards-pass datawidth, which must be '32'.");

    xf::cv::dt_kernel_fw_pass<IN_PTR, FW_PTR, ROWS, COLS, USE_URAM> dt_fw(rows, cols);
    xf::cv::dt_kernel_bk_pass<FW_PTR, ROWS, COLS, USE_URAM> dt_bk(rows, cols);

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.distanceTransform(@input_0, @dist_type, @mask_size)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @dist_type, @mask_size
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `duplicate_image`

**Funcionalidad:** Duplicate one image stream into two outputs.

**Categoría:** `channels`

- Inputs: @input_0:image
- Outputs: @output_0:image, @output_1:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_duplicateimage.hpp`
- Source: `vision/L1/include/imgproc/xf_duplicateimage.hpp`
- Function:
```cpp
xf::cv::duplicateMat<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_duplicateimage.hpp`, firma detectada para `duplicateMat` alrededor de línea 118:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_2 = _XFCVDEPTH_DEFAULT>
void duplicateMat(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst1,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_2>& _dst2) {
// clang-format off
#pragma HLS inline off
    // clang-format on

    xFDuplicate<ROWS, COLS, SRC_T, XF_DEPTH(SRC_T, NPC), NPC, XFCVDEPTH_IN_1, XFCVDEPTH_OUT_1, XFCVDEPTH_OUT_2,
                XF_WORDWIDTH(SRC_T, NPC)>(_src, _dst1, _dst2, _src.rows, _src.cols);
}

template <int SRC_T,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Function:
```python
@output_0 = @input_0.copy()
@output_1 = @input_0.copy()
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `edge_tracing`

**Funcionalidad:** Trace connected edge components.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_edge_tracing.hpp`
- Source: `vision/L1/include/imgproc/xf_edge_tracing.hpp`
- Function:
```cpp
xf::cv::EdgeTracing<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_edge_tracing.hpp`, firma detectada para `EdgeTracing` alrededor de línea 458:
```cpp
            if (j < (width_8 / 8)) _dst.write((ii * (width_8 / 8) + j), oRegF[0]);
        }
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC_SRC, int NPC_DST, bool USE_URAM = false, int depthm = -1>
void EdgeTracing(xf::cv::Mat<SRC_T, ROWS, COLS, NPC_SRC, depthm>& _src,
                 xf::cv::Mat<DST_T, ROWS, COLS, NPC_DST, depthm>& _dst) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    xfEdgeTracing<SRC_T, DST_T, NPC_SRC, NPC_DST, ROWS, COLS, USE_URAM, depthm>(_dst, _src, _src.rows, _src.cols,
                                                                                _dst.cols);
}

} // namespace cv
} // namespace xf
#endif
```

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- No hay implementación funcional declarada.

## `erode`

**Funcionalidad:** Apply morphological erosion.

**Categoría:** `morphology`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `kernel_shape` / `@K_SHAPE` (`enum`): Structuring element shape.
- `kernel_rows` / `@K_ROWS` (`integer`): Kernel height.
- `kernel_cols` / `@K_COLS` (`integer`): Kernel width.
- `iterations` / `@ITERATIONS` (`integer`): Number of morphology iterations.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_erosion.hpp`
- Source: `vision/L1/include/imgproc/xf_erosion.hpp`
- Function:
```cpp
xf::cv::erode<@BORDER_TYPE, @TYPE, @ROWS, @COLS, @K_SHAPE, @K_ROWS, @K_COLS, @ITERATIONS, @NPC>(@input_0, @output_0, @kernel);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @ITERATIONS, @K_COLS, @K_ROWS, @K_SHAPE
- Tokens resueltos por composer: @BORDER_TYPE, @COLS, @NPC, @ROWS, @TYPE, @kernel
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_erosion.hpp`, firma detectada para `erode` alrededor de línea 359:
```cpp
          int K_ROWS,
          int K_COLS,
          int ITERATIONS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void erode(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src,
           xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst,
           unsigned char _kernel[K_ROWS * K_COLS]) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    unsigned short imgheight = _src.rows;
    unsigned short imgwidth = _src.cols;
#ifndef __SYNTHESIS__
    assert(BORDER_TYPE == XF_BORDER_CONSTANT && "Only XF_BORDER_CONSTANT is supported");
    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    if (K_SHAPE == XF_SHAPE_RECT) // iterations >1 is not supported for ELLIPSE and  CROSS
    {
#define NEW_K_ROWS (K_ROWS + ((ITERATIONS - 1) * (K_ROWS - 1)))
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
kernel = cv.getStructuringElement(@K_SHAPE, (@K_COLS, @K_ROWS))
@output_0 = cv.erode(@input_0, kernel, iterations=@ITERATIONS)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @ITERATIONS, @K_COLS, @K_ROWS, @K_SHAPE
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `extract_eframes`

**Funcionalidad:** Extract exposure frames from sensor stream.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image, @output_1:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_extract_eframes.hpp`
- Source: `vision/L1/include/imgproc/xf_extract_eframes.hpp`
- Function:
```cpp
xf::cv::extractExposureFrames<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_extract_eframes.hpp`, firma detectada para `extractExposureFrames` alrededor de línea 225:
```cpp
          int MAX_COLS,
          int NPPC = XF_NPPC1,
          int USE_URAM = 0,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_LEF = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_SEF = _XFCVDEPTH_DEFAULT>
void extractExposureFrames(xf::cv::Mat<SRC_T, MAX_ROWS * 2, MAX_COLS + N_COLS, NPPC, XFCVDEPTH_IN_1>& _hdrSrc,
                           xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC, XFCVDEPTH_LEF>& _lefSrc,
                           xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC, XFCVDEPTH_SEF>& _sefSrc) {
// clang-format off
        #pragma HLS INLINE OFF
    // clang-format on

    xf::cv::ExposureFramesExtract<SRC_T, N_ROWS, N_COLS, MAX_ROWS, MAX_COLS, NPPC, XFCVDEPTH_IN_1, XFCVDEPTH_LEF,
                                  XFCVDEPTH_SEF, USE_URAM>
        extractor;

    extractor.extract(_hdrSrc, _lefSrc, _sefSrc);

    return;
}
}
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = @input_0[..., 0::2]
@output_1 = @input_0[..., 1::2]
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `find_contours`

**Funcionalidad:** Find contours in a binary image.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- `retrieval_mode` / `@RETRIEVAL_MODE` (`enum`): Contour retrieval mode.
- `approximation_method` / `@APPROXIMATION_METHOD` (`enum`): Contour approximation method.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_findcontours.hpp`
- Source: `vision/L1/include/imgproc/xf_findcontours.hpp`
- Function:
```cpp
xf::cv::findContours<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `findContours` en `Vitis_Libraries/vision/L1/include/imgproc/xf_findcontours.hpp`.

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0, hierarchy = cv.findContours(@input_0, @RETRIEVAL_MODE, @APPROXIMATION_METHOD)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @APPROXIMATION_METHOD, @RETRIEVAL_MODE
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `flip`

**Funcionalidad:** Flip image around horizontal, vertical or both axes.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `flip_code` / `@flip_code` (`enum`): Flip direction code.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_flip.hpp`
- Source: `vision/L1/include/imgproc/xf_flip.hpp`
- Function:
```cpp
xf::cv::flip<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @flip_code);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @flip_code
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_flip.hpp`, firma detectada para `flip` alrededor de línea 151:
```cpp
	MMIterOut<PTR_WIDTH, TYPE, 1, COLS, NPC, 1, -1>::xfMat2Array(DstRow, DstPtr+OffsetDst, 1, Cols, -1);
		
	return;
}

template <int PTR_WIDTH, int TYPE, int ROWS, int COLS, int NPC>
void flip(ap_uint<PTR_WIDTH>* SrcPtr,
                      ap_uint<PTR_WIDTH>* DstPtr,
                      int Rows,
                      int Cols,
					  int Direction){
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on		
						  
#ifndef __SYNTHESIS__
    assert(((TYPE == XF_8UC1) || (TYPE == XF_8UC3)) &&
           "Input TYPE must be XF_8UC1 for 1-channel, XF_8UC3 for 3-channel");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC4) || (NPC == XF_NPPC2) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC4 ");
    assert((Rows <= ROWS) && (Cols <= COLS) && "COLS should be greater than input image size ");
#endif
	const int NPC_COLS = COLS >> XF_BITSHIFT(NPC);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.flip(@input_0, @flip_code)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @flip_code
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `gain_control`

**Funcionalidad:** Apply gain control.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `gain` / `@gain` (`number`): Gain value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_gaincontrol.hpp`
- Source: `vision/L1/include/imgproc/xf_gaincontrol.hpp`
- Function:
```cpp
xf::cv::gaincontrol<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_gaincontrol.hpp`, firma detectada para `gaincontrol` alrededor de línea 195:
```cpp
template <int SRC_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void gaincontrol(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst,
                 unsigned short rgain,
                 unsigned short bgain,
                 unsigned short ggain,
                 unsigned short bayer_p) {
#pragma HLS INLINE OFF
#ifndef __SYNTHESIS__
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

    gaincontrolkernel<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1, XFCVDEPTH_OUT_1, XF_CHANNELS(SRC_T, NPC),
                      XF_DEPTH(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                      (COLS >> XF_BITSHIFT(NPC))>(src1, dst, src1.rows, width, rgain, bgain, ggain, bayer_p);
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(@input_0.astype(np.float32) * @gain, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @gain, @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `gain_control_multi`

**Funcionalidad:** Apply multi-channel gain control.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `gains` / `@gains` (`array`): Per-channel gain values.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_gaincontrol_multi.hpp`
- Source: `vision/L1/include/imgproc/xf_gaincontrol_multi.hpp`
- Function:
```cpp
xf::cv::gaincontrol<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_gaincontrol_multi.hpp`, firma detectada para `gaincontrol` alrededor de línea 189:
```cpp
template <int SRC_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void gaincontrol(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst,
                 unsigned short rgain,
                 unsigned short bgain,
                 unsigned short ggain,
                 unsigned short bayer_p) {
#pragma HLS INLINE OFF
#ifndef __SYNTHESIS__
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

    gaincontrolkernel<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1, XFCVDEPTH_OUT_1, XF_CHANNELS(SRC_T, NPC),
                      XF_DEPTH(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                      (COLS >> XF_BITSHIFT(NPC))>(src1, dst, src1.rows, width, rgain, bgain, ggain, bayer_p);
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(@input_0.astype(np.float32) * @gains, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @gains, @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `gamma_correction`

**Funcionalidad:** Apply gamma correction.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `gamma` / `@gamma` (`number`): Gamma value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_gammacorrection.hpp`
- Source: `vision/L1/include/imgproc/xf_gammacorrection.hpp`
- Function:
```cpp
xf::cv::gammacorrection<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_gammacorrection.hpp`, firma detectada para `gammacorrection` alrededor de línea 107:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void gammacorrection(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst,
                     unsigned char lut_table[256 * XF_CHANNELS(SRC_T, NPC)]) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    unsigned short height = src.rows;
    unsigned short width = src.cols >> XF_BITSHIFT(NPC);

    xFGAMMAKernel<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC, XFCVDEPTH_IN_1,
                  XFCVDEPTH_OUT_1, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(
        src, dst, lut_table, height, width);
}

template <int SRC_T,
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
normalized = @input_0.astype(np.float32) / @maxval
@output_0 = np.clip(np.power(normalized, @gamma) * @maxval, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @gamma, @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `gaussian_filter`

**Funcionalidad:** Apply Gaussian blur filtering.

**Categoría:** `filtering`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `filter_width` / `@FILTER_WIDTH` (`integer`): Filter or kernel width.
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.
- `sigma` / `@sigma` (`number`): Gaussian sigma.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_gaussian_filter.hpp`
- Source: `vision/L1/include/imgproc/xf_gaussian_filter.hpp`
- Function:
```cpp
xf::cv::GaussianBlur<@FILTER_WIDTH, @BORDER_TYPE, @TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @sigma);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @FILTER_WIDTH, @sigma
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_gaussian_filter.hpp`, firma detectada para `GaussianBlur` alrededor de línea 1295:
```cpp
          int SRC_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void GaussianBlur(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst,
                  float sigma) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    int imgwidth = _src.cols >> XF_BITSHIFT(NPC);

    if (FILTER_SIZE == XF_FILTER_3X3) {
        unsigned char weights[3];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
        // clang-format on
        weightsghcalculation3x3(sigma, weights);
        xfGaussianFilter3x3<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC, XFCVDEPTH_IN_1,
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
ksize = (@FILTER_WIDTH, @FILTER_WIDTH)
@output_0 = cv.GaussianBlur(@input_0, ksize, sigmaX=@sigma, sigmaY=@sigma, borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @FILTER_WIDTH, @sigma
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `global_tone_mapping`

**Funcionalidad:** Apply global tone mapping.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_gtm.hpp`
- Source: `vision/L1/include/imgproc/xf_gtm.hpp`
- Function:
```cpp
xf::cv::gtm<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_gtm.hpp`, firma detectada para `gtm` alrededor de línea 426:
```cpp
          int SIN_CHANNEL_OUT_TYPE,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void gtm(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& src,
         xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& dst,
         ap_ufixed<16, 4>& mean1,
         ap_ufixed<16, 4>& mean2,
         ap_ufixed<16, 4>& L_max1,
         ap_ufixed<16, 4>& L_max2,
         ap_ufixed<16, 4>& L_min1,
         ap_ufixed<16, 4>& L_min2,
         unsigned int c1,
         unsigned int c2) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_16UC3) || (SRC_T == XF_14UC3)) && "Input TYPE must be XF_16UC3 or XF_14UC3");
    assert(((SIN_CHANNEL_IN_TYPE == XF_16UC1) || (SIN_CHANNEL_IN_TYPE == XF_14UC1)) &&
           "Input Single Channel TYPE must be XF_16UC1 or XF_14UC1");
    assert(((DST_T == XF_8UC3) || (SIN_CHANNEL_OUT_TYPE == XF_8UC1)) && "OUTPUT TYPE must be XF_8UC3");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2)) && "NPC must be XF_NPPC1, XF_NPPC2 ");
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(@input_0.astype(np.float32) / np.maximum(np.max(@input_0), 1e-6) * @maxval, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `gray_to_bgr`

**Funcionalidad:** Convert grayscale image to BGR.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cvt_color.hpp`
- Source: `vision/L1/include/imgproc/xf_cvt_color.hpp`
- Function:
```cpp
xf::cv::gray2bgr<XF_8UC1, XF_8UC3, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color.hpp`, firma detectada para `gray2bgr` alrededor de línea 5765:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void gray2bgr(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) && " GRAY image Type must be XF_8UC1");
    assert((DST_T == XF_8UC3) && " BGR image Type must be XF_8UC3");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && " GRAY image rows and cols should be less than ROWS, COLS");
    assert(((_dst.cols == _src.cols) && (_dst.rows == _src.rows)) && "BGR and GRAY plane dimensions mismatch");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && " 1,8 pixel parallelism is supported  ");
#endif
    xfgray2bgr<SRC_T, DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_T, NPC),
               XF_WORDWIDTH(DST_T, NPC), (ROWS * (COLS >> (XF_NPIXPERCYCLE(NPC))))>(_src, _dst, _src.rows, _src.cols);
}
/////////////////////////////////	RGB2XYZ
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_GRAY2BGR)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `gray_to_rgb`

**Funcionalidad:** Convert grayscale image to RGB.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cvt_color.hpp`
- Source: `vision/L1/include/imgproc/xf_cvt_color.hpp`
- Function:
```cpp
xf::cv::gray2rgb<XF_8UC1, XF_8UC3, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color.hpp`, firma detectada para `gray2rgb` alrededor de línea 5688:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void gray2rgb(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) && " GRAY image Type must be XF_8UC1");
    assert((DST_T == XF_8UC3) && " RGB image Type must be XF_8UC3");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && " GRAY image rows and cols should be less than ROWS, COLS");
    assert(((_dst.cols == _src.cols) && (_dst.rows == _src.rows)) && "RGB and GRAY plane dimensions mismatch");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && " 1,8 pixel parallelism is supported  ");
#endif
    xfgray2rgb<SRC_T, DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_T, NPC),
               XF_WORDWIDTH(DST_T, NPC), (ROWS * (COLS >> (XF_NPIXPERCYCLE(NPC))))>(_src, _dst, _src.rows, _src.cols);
}
//////////////////////////////////////	GRAY2BGR
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_GRAY2RGB)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `hdr_decompand`

**Funcionalidad:** Apply HDR decompanding.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `gain` / `@gain` (`number`): Gain value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hdrdecompand.hpp`
- Source: `vision/L1/include/imgproc/xf_hdrdecompand.hpp`
- Function:
```cpp
xf::cv::HdrDecompand<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `HdrDecompand` en `Vitis_Libraries/vision/L1/include/imgproc/xf_hdrdecompand.hpp`.

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.clip(@input_0.astype(np.float32) * @gain, 0, @maxval).astype(@input_0.dtype)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @gain, @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `hdr_merge`

**Funcionalidad:** Merge short and long exposure images.

**Categoría:** `isp`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hdrmerge.hpp`
- Source: `vision/L1/include/imgproc/xf_hdrmerge.hpp`
- Function:
```cpp
xf::cv::Hdrmerge<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `Hdrmerge` en `Vitis_Libraries/vision/L1/include/imgproc/xf_hdrmerge.hpp`.

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.maximum(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `hist_equalize`

**Funcionalidad:** Apply histogram equalization.

**Categoría:** `statistics`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hist_equalize.hpp`
- Source: `vision/L1/include/imgproc/xf_hist_equalize.hpp`
- Function:
```cpp
xf::cv::equalizeHist<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @input_duplicate, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE, @input_duplicate
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_hist_equalize.hpp`, firma detectada para `equalizeHist` alrededor de línea 468:
```cpp
          int COLS,
          int NPC = 1,
          int USE_URAM = 0,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void equalizeHist(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src1,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    uint16_t img_height = _src1.rows;
    uint16_t img_width = _src1.cols;
#ifndef __SYNTHESIS__
    assert(((img_height <= ROWS) && (img_width <= COLS)) && "ROWS and COLS should be greater than input image");

    assert((SRC_T == XF_8UC1) && "Type must be of XF_8UC1");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && " NPC must be XF_NPPC1, XF_NPPC8");
#endif
    uint32_t histogram[1][256];
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.equalizeHist(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `histogram`

**Funcionalidad:** Compute image histogram.

**Categoría:** `statistics`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:array

**Parámetros declarados:**
- `channel` / `@channel` (`integer`): Channel index.
- `bins` / `@bins` (`integer`): Number of histogram bins.
- `ranges` / `@ranges` (`array`): Histogram value ranges.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_histogram.hpp`
- Source: `vision/L1/include/imgproc/xf_histogram.hpp`
- Function:
```cpp
xf::cv::calcHist<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_histogram.hpp`, firma detectada para `calcHist` alrededor de línea 462:
```cpp
        }
        hist[i] += value;
    }
}

template <int SRC_T, int ROWS, int COLS, int NPC = 1, int USE_URAM = 0, int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void calcHist(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src, uint32_t* histogram) {
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    uint32_t hist_array[XF_CHANNELS(SRC_T, NPC)][256] = {0};
    uint16_t width = _src.cols >> (XF_BITSHIFT(NPC));
    uint16_t height = _src.rows;

    xFHistogramKernel<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, USE_URAM, XFCVDEPTH_IN, XF_WORDWIDTH(SRC_T, NPC),
                      ((COLS >> (XF_BITSHIFT(NPC))) >> 1), XF_CHANNELS(SRC_T, NPC)>(_src, hist_array, height, width);

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.calcHist([@input_0], [@channel], @input_1, [@bins], @ranges)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @bins, @channel, @ranges
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `hog_compute_hist`

**Funcionalidad:** Compute HOG cell histograms.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- `nbins` / `@nbins` (`integer`): Number of bins.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hog_descriptor_compute_hist.hpp`
- Source: `vision/L1/include/imgproc/xf_hog_descriptor_compute_hist.hpp`
- Function:
```cpp
xf::cv::HOGComputeHist<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `HOGComputeHist` en `Vitis_Libraries/vision/L1/include/imgproc/xf_hog_descriptor_compute_hist.hpp`.

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.histogram(@input_0, bins=@nbins)[0]
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @nbins
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `hog_descriptor`

**Funcionalidad:** Compute histogram of oriented gradients descriptors.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- `kernel_rows` / `@K_ROWS` (`integer`): Kernel height.
- `kernel_cols` / `@K_COLS` (`integer`): Kernel width.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hog_descriptor.hpp`
- Source: `vision/L1/include/imgproc/xf_hog_descriptor.hpp`
- Function:
```cpp
xf::cv::HOGDescriptor<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_hog_descriptor.hpp`, firma detectada para `HOGDescriptor` alrededor de línea 50:
```cpp
          int ROWS,
          int COLS,
          int NPC = XF_NPPC1,
          bool USE_URAM = false,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_DESC = _XFCVDEPTH_DEFAULT>
void HOGDescriptor(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _in_mat,
                   xf::cv::Mat<DST_T, 1, DESC_SIZE, NPC, XFCVDEPTH_DESC>& _desc_mat) {
    hls::stream<XF_TNAME(SRC_T, NPC)> in_strm;
    hls::stream<XF_CTUNAME(SRC_T, NPC)> in[IMG_COLOR];
    hls::stream<XF_SNAME(XF_576UW)> _block_strm;
    hls::stream<XF_TNAME(DST_T, NPC)> desc_strm;

    int dsize = _desc_mat.size;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    const int IN_TC = (ROWS * COLS >> XF_BITSHIFT(NPC));
    for (int i = 0; i < _in_mat.size; i++) {
// clang-format off
```

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `scikit-image`
- Imports: `from skimage.feature import hog`
- Function:
```python
@output_0 = hog(@input_0, pixels_per_cell=(@K_ROWS, @K_COLS))
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @K_COLS, @K_ROWS
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `hog_gradients`

**Funcionalidad:** Compute gradients for HOG descriptor pipeline.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:image, @output_1:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hog_descriptor_gradients.hpp`
- Source: `vision/L1/include/imgproc/xf_hog_descriptor_gradients.hpp`
- Function:
```cpp
xf::cv::HOGGradients<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `HOGGradients` en `Vitis_Libraries/vision/L1/include/imgproc/xf_hog_descriptor_gradients.hpp`.

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.gradient(@input_0.astype(np.float32))
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `hog_hist_norm`

**Funcionalidad:** Normalize HOG histograms.

**Categoría:** `features`

- Inputs: @input_0:array
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hog_descriptor_hist_norm.hpp`
- Source: `vision/L1/include/imgproc/xf_hog_descriptor_hist_norm.hpp`
- Function:
```cpp
xf::cv::HOGHistNorm<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `HOGHistNorm` en `Vitis_Libraries/vision/L1/include/imgproc/xf_hog_descriptor_hist_norm.hpp`.

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = @input_0 / np.maximum(np.linalg.norm(@input_0), 1e-6)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `hog_norm`

**Funcionalidad:** Normalize HOG descriptor blocks.

**Categoría:** `features`

- Inputs: @input_0:array
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hog_descriptor_norm.hpp`
- Source: `vision/L1/include/imgproc/xf_hog_descriptor_norm.hpp`
- Function:
```cpp
xf::cv::HOGNorm<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `HOGNorm` en `Vitis_Libraries/vision/L1/include/imgproc/xf_hog_descriptor_norm.hpp`.

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = @input_0 / np.maximum(np.linalg.norm(@input_0), 1e-6)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `hog_pm`

**Funcionalidad:** Compute phase and magnitude for HOG descriptor pipeline.

**Categoría:** `features`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image, @output_1:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_hog_descriptor_pm.hpp`
- Source: `vision/L1/include/imgproc/xf_hog_descriptor_pm.hpp`
- Function:
```cpp
xf::cv::HOGPhaseMagnitude<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `HOGPhaseMagnitude` en `Vitis_Libraries/vision/L1/include/imgproc/xf_hog_descriptor_pm.hpp`.

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
gy, gx = np.gradient(@input_0.astype(np.float32))
@output_0 = np.hypot(gx, gy)
@output_1 = np.arctan2(gy, gx)
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `hough_lines`

**Funcionalidad:** Detect lines using Hough transform.

**Categoría:** `features`

- Inputs: @input_0:image
- Outputs: @output_0:array, @output_1:array

**Parámetros declarados:**
- `threshold` / `@threshold` (`number`): Threshold value.
- `rho` / `@RHO` (`number`): Distance resolution for Hough transform.
- `theta` / `@THETA` (`number`): Angle resolution for Hough transform.
- `linesmax` / `@linesmax` (`integer`): Maximum number of lines to return.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_houghlines.hpp`
- Source: `vision/L1/include/imgproc/xf_houghlines.hpp`
- Function:
```cpp
xf::cv::HoughLines<@RHO, @THETA, @MAXLINES, @DIAG, @MINTHETA, @MAXTHETA, @TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @output_1, @threshold, @linesmax);
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: @RHO, @THETA, @linesmax, @threshold
- Tokens resueltos por composer: @COLS, @DIAG, @MAXLINES, @MAXTHETA, @MINTHETA, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_houghlines.hpp`, firma detectada para `HoughLines` alrededor de línea 719:
```cpp
          int MAXTHETA,
          int SRC_T,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT>
void HoughLines(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src_mat,
                float outputrho[MAXLINES],
                float outputtheta[MAXLINES],
                short threshold,
                short linesmax) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((NPC == XF_NPPC1) && "NPC must be XF_NPPC1");

    assert((((MAXTHETA - MINTHETA) > 0)) && "MINTHETA must be less than MAXTHETA");
    assert(((MINTHETA >= 0) && (MINTHETA < 180)) && "MINTHETA must be between 0 to 180");
    assert(((MAXTHETA > 0) && (MAXTHETA <= 180)) && "MAXTHETA must be between 0 to 180");
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.HoughLines(@input_0, @RHO, @THETA, @threshold)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @RHO, @THETA, @threshold
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `in_range`

**Funcionalidad:** Check whether image pixels lie between lower and upper bounds.

**Categoría:** `thresholding`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `lower_0` / `@lower_0` (`integer`): Lower bound for channel 0.
- `lower_1` / `@lower_1` (`integer`): Lower bound for channel 1.
- `lower_2` / `@lower_2` (`integer`): Lower bound for channel 2.
- `upper_0` / `@upper_0` (`integer`): Upper bound for channel 0.
- `upper_1` / `@upper_1` (`integer`): Upper bound for channel 1.
- `upper_2` / `@upper_2` (`integer`): Upper bound for channel 2.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_inrange.hpp`
- Source: `vision/L1/include/imgproc/xf_inrange.hpp`
- Function:
```cpp
unsigned char lower_bound[3] = {@lower_0, @lower_1, @lower_2};
unsigned char upper_bound[3] = {@upper_0, @upper_1, @upper_2};
xf::cv::inRange<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, lower_bound, upper_bound);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @lower_0, @lower_1, @lower_2, @upper_0, @upper_1, @upper_2
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_inrange.hpp`, firma detectada para `inRange` alrededor de línea 137:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void inRange(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& src,
             unsigned char lower_thresh[XF_CHANNELS(SRC_T, NPC)],
             unsigned char upper_thresh[XF_CHANNELS(SRC_T, NPC)],
             xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& dst) {
    unsigned short width = src.cols >> XF_BITSHIFT(NPC);
    unsigned short height = src.rows;

#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16UC1) || (SRC_T == XF_16UC3)) &&
           "Type must be XF_8UC1 or XF_8UC3");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");

    assert(((lower_thresh[0] >= 0) && (lower_thresh[0] <= 255)) && "lower_thresh must be with the range of 0 to 255");

    assert(((upper_thresh[0] >= 0) && (upper_thresh[0] <= 255)) && "lower_thresh must be with the range of 0 to 255");
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv; import numpy as np`
- Function:
```python
lower_bound = np.array([@lower_0, @lower_1, @lower_2])
upper_bound = np.array([@upper_0, @upper_1, @upper_2])
@output_0 = cv.inRange(@input_0, lower_bound, upper_bound)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @lower_0, @lower_1, @lower_2, @upper_0, @upper_1, @upper_2
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `insert_border`

**Funcionalidad:** Insert border padding around image.

**Categoría:** `dnn`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.
- `top` / `@top` (`integer`): Top border size.
- `bottom` / `@bottom` (`integer`): Bottom border size.
- `left` / `@left` (`integer`): Left border size.
- `right` / `@right` (`integer`): Right border size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `dnn/xf_insertBorder.hpp`
- Source: `vision/L1/include/dnn/xf_insertBorder.hpp`
- Function:
```cpp
xf::cv::insertBorder<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/dnn/xf_insertBorder.hpp`, firma detectada para `insertBorder` alrededor de línea 53:
```cpp
          int SRC_COLS,
          int DST_ROWS,
          int DST_COLS,
          int NPC,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void insertBorder(xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC, XFCVDEPTH_IN>& _src,
                  xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC, XFCVDEPTH_OUT>& _dst,
                  int insert_pad_val) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    enum { DEPTH = TYPE, PLANES = XF_CHANNELS(TYPE, NPC) };

    unsigned short in_height = _src.rows;
    unsigned short in_width = _src.cols;
    unsigned short out_height = _dst.rows;
    unsigned short out_width = _dst.cols;

    unsigned short dx = (out_width - in_width) >> 1;
    unsigned short dy = (out_height - in_height) >> 1;
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.copyMakeBorder(@input_0, @top, @bottom, @left, @right, @BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @bottom, @left, @right, @top
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `integral_image`

**Funcionalidad:** Compute integral image.

**Categoría:** `statistics`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_integral_image.hpp`
- Source: `vision/L1/include/imgproc/xf_integral_image.hpp`
- Function:
```cpp
xf::cv::integral<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_integral_image.hpp`, firma detectada para `integral` alrededor de línea 95:
```cpp
          int DST_TYPE,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void integral(xf::cv::Mat<SRC_TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
              xf::cv::Mat<DST_TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((_src_mat.rows <= ROWS) && (_dst_mat.cols <= COLS)) &&
           "ROWS and COLS should be greater than or equal to input image size");
#endif
    XFIntegralImageKernel<SRC_TYPE, DST_TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_TYPE, NPC),
                          XF_WORDWIDTH(DST_TYPE, NPC), (COLS >> XF_BITSHIFT(NPC))>(_src_mat, _dst_mat, _src_mat.rows,
                                                                                   _src_mat.cols);
}
} // namespace cv
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.integral(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `isp_stats`

**Funcionalidad:** Compute ISP statistics.

**Categoría:** `statistics`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_ispstats.hpp`
- Source: `vision/L1/include/imgproc/xf_ispstats.hpp`
- Function:
```cpp
xf::cv::IspStats<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `IspStats` en `Vitis_Libraries/vision/L1/include/imgproc/xf_ispstats.hpp`.

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.
- No hay implementación funcional declarada.

## `lens_shading_correction`

**Funcionalidad:** Apply lens shading correction.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_lensshadingcorrection.hpp`
- Source: `vision/L1/include/imgproc/xf_lensshadingcorrection.hpp`
- Function:
```cpp
xf::cv::Lscdistancebased<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_lensshadingcorrection.hpp`, firma detectada para `Lscdistancebased` alrededor de línea 82:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void Lscdistancebased(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& src,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& dst) {
    int rows = src.rows;
    int cols = src.cols >> XF_BITSHIFT(NPC);

    assert(((rows <= ROWS) && (cols <= COLS)) && "ROWS and COLS should be greater than input image");

    short center_pixel_pos_x = (src.cols >> 1);
    short center_pixel_pos_y = (rows >> 1);
    short y_distance = rows - center_pixel_pos_y;
    short x_distance = src.cols - center_pixel_pos_x;
    float y_2 = y_distance * y_distance;
    float x_2 = x_distance * x_distance;

    float max_distance = std::sqrt(y_2 + x_2);
    // ap_fixed<48,24> max_distance_inv = (1/max_distance);
```

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- No hay implementación funcional declarada.

## `letterbox`

**Funcionalidad:** Resize and pad image preserving aspect ratio.

**Categoría:** `dnn`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `out_rows` / `@OUT_ROWS` (`integer`): Output image height.
- `out_cols` / `@OUT_COLS` (`integer`): Output image width.
- `interpolation` / `@INTERPOLATION` (`enum`): Interpolation policy.
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.
- `top` / `@top` (`integer`): Top border size.
- `bottom` / `@bottom` (`integer`): Bottom border size.
- `left` / `@left` (`integer`): Left border size.
- `right` / `@right` (`integer`): Right border size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `dnn/xf_letterbox.hpp`
- Source: `vision/L1/include/dnn/xf_letterbox.hpp`
- Function:
```cpp
xf::cv::letterbox<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/dnn/xf_letterbox.hpp`, firma detectada para `letterbox` alrededor de línea 271:
```cpp
          int DST_COLS,
          int NPC,
          int MAX_DOWN_SCALE,
          int INSERT_VAL,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void letterbox(xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC, XFCVDEPTH_IN>& _src,
               xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC, XFCVDEPTH_OUT>& _dst,
               int rows_out_resize,
               int cols_out_resize) {
// clang-format off
#pragma inline off
    // clang-format on
    xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC, XFCVDEPTH_OUT> out_mat_resize(rows_out_resize, cols_out_resize);
// clang-format off
#pragma HLS DATAFLOW
    // clang-format on
    xf::cv::resize<INTERPOLATION_TYPE, TYPE, SRC_ROWS, SRC_COLS, DST_ROWS, DST_COLS, NPC, 1, XFCVDEPTH_IN,
                   XFCVDEPTH_OUT, MAX_DOWN_SCALE>(_src, out_mat_resize);
    xf::cv::insertBorder<TYPE, DST_ROWS, DST_COLS, DST_ROWS, DST_COLS, NPC, 128, XFCVDEPTH_IN, XFCVDEPTH_OUT>(
        out_mat_resize, _dst);
}
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
resized = cv.resize(@input_0, (@OUT_COLS, @OUT_ROWS), interpolation=@INTERPOLATION)
@output_0 = cv.copyMakeBorder(resized, @top, @bottom, @left, @right, @BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @INTERPOLATION, @OUT_COLS, @OUT_ROWS, @bottom, @left, @right, @top
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `local_tone_mapping`

**Funcionalidad:** Apply local tone mapping.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `clip_limit` / `@clip_limit` (`number`): CLAHE contrast clipping limit.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_ltm.hpp`
- Source: `vision/L1/include/imgproc/xf_ltm.hpp`
- Function:
```cpp
xf::cv::LTM<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `LTM` en `Vitis_Libraries/vision/L1/include/imgproc/xf_ltm.hpp`.

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `scikit-image`
- Imports: `from skimage import exposure`
- Function:
```python
@output_0 = exposure.equalize_adapthist(@input_0, clip_limit=@clip_limit)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @clip_limit
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `lut`

**Funcionalidad:** Apply look-up table transform.

**Categoría:** `enhancement`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `lut` / `@lut` (`array`): Look-up table supplied as stage value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_lut.hpp`
- Source: `vision/L1/include/imgproc/xf_lut.hpp`
- Function:
```cpp
xf::cv::LUT<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_lut.hpp`, firma detectada para `LUT` alrededor de línea 125:
```cpp
template <int SRC_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void LUT(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
         xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst,
         unsigned char* _lut) {
// clang-format off
    #pragma HLS INLINE OFF
	unsigned char height=_src.rows;
	unsigned char width=_src.cols;
	
	 #ifndef __SYNTHESIS__
    	assert((SRC_T == XF_8UC1 ) ||(SRC_T == XF_8UC3 ) && "input type must be XF_8UC1 or XF_8UC3");
    	assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1 or XF_NPPC8");
    	assert(((height <= ROWS ) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
	#endif
	
    xFLUTKernel<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_T, NPC),
               XF_WORDWIDTH(SRC_T, NPC),(COLS >> XF_BITSHIFT(NPC))>(_src, _dst, _lut, _src.rows, _src.cols);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.LUT(@input_0, @lut)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @lut
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `magnitude`

**Funcionalidad:** Compute magnitude from two gradient images.

**Categoría:** `gradients`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_magnitude.hpp`
- Source: `vision/L1/include/core/xf_magnitude.hpp`
- Function:
```cpp
xf::cv::magnitude<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_magnitude.hpp`, firma detectada para `magnitude` alrededor de línea 129:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_X = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_Y = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void magnitude(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_X>& _src_matx,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_Y>& _src_maty,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat) {
#ifndef __SYNTHESIS__
    assert(((_src_matx.rows <= ROWS) && (_src_matx.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
    assert(((_src_maty.rows <= ROWS) && (_src_maty.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
    assert(((_src_matx.rows == _src_maty.rows) && (_src_matx.cols == _src_maty.cols)) &&
           "Both input images should have same size");
    assert(((_src_matx.rows == _dst_mat.rows) && (_src_matx.cols == _dst_mat.cols)) &&
           "Input and output image should be of same size");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
#endif

// clang-format off
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv+numpy`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.magnitude(@input_0.astype(np.float32), @input_1.astype(np.float32))
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `mean_shift`

**Funcionalidad:** Apply mean-shift filtering/tracking.

**Categoría:** `segmentation`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `n_segments` / `@n_segments` (`integer`): Number of segments.
- `compactness` / `@compactness` (`number`): Compactness factor.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_mean_shift.hpp`
- Source: `vision/L1/include/imgproc/xf_mean_shift.hpp`
- Function:
```cpp
xf::cv::MeanShift<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_mean_shift.hpp`, firma detectada para `MeanShift` alrededor de línea 36:
```cpp
          int OBJ_COLS,
          int SRC_T,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void MeanShift(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _in_mat,
               uint16_t* x1,
               uint16_t* y1,
               uint16_t* obj_height,
               uint16_t* obj_width,
               uint16_t* dx,
               uint16_t* dy,
               uint16_t* status,
               uint8_t frame_status,
               uint8_t no_objects,
               uint8_t no_iters) {
    // local arrays for memcopy
    uint16_t img_height[1], img_width[1], objects[1], frame[1];
    uint16_t tlx[MAXOBJ], tly[MAXOBJ], _obj_height[MAXOBJ], _obj_width[MAXOBJ], dispx[MAXOBJ], dispy[MAXOBJ];
    uint16_t track_status[MAXOBJ];

```

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `scikit-image`
- Imports: `from skimage.segmentation import slic`
- Function:
```python
@output_0 = slic(@input_0, n_segments=@n_segments, compactness=@compactness)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @compactness, @n_segments
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `mean_stddev`

**Funcionalidad:** Compute image mean and standard deviation.

**Categoría:** `statistics`

- Inputs: @input_0:image
- Outputs: @output_0:scalar, @output_1:scalar

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_mean_stddev.hpp`
- Source: `vision/L1/include/core/xf_mean_stddev.hpp`
- Function:
```cpp
xf::cv::meanStdDev<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_mean_stddev.hpp`, firma detectada para `meanStdDev` alrededor de línea 143:
```cpp
        _mean[i] = mean_acc.range(i * 16 + 15, i * 16);
        _dst_stddev[i] = stddev_acc.range(i * 16 + 15, i * 16);
    }
}

template <int SRC_T, int ROWS, int COLS, int NPC = 1, int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void meanStdDev(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
                unsigned short _mean[XF_CHANNELS(SRC_T, NPC)],
                unsigned short _stddev[XF_CHANNELS(SRC_T, NPC)]) {
// clang-format off
    #pragma HLS inline off
// clang-format on
//#pragma HLS dataflow

#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1 || SRC_T == XF_8UC3 || SRC_T == XF_8UC4) &&
           "Input image type should be XF_8UC1, XF_8UC3 or XF_8UC4");

    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");

    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && "ROWS and COLS should be greater than input image");
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
mean, stddev = cv.meanStdDev(@input_0)
@output_0 = mean
@output_1 = stddev
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `median_blur`

**Funcionalidad:** Apply median blur filtering.

**Categoría:** `filtering`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `filter_width` / `@FILTER_WIDTH` (`integer`): Filter or kernel width.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_median_blur.hpp`
- Source: `vision/L1/include/imgproc/xf_median_blur.hpp`
- Function:
```cpp
xf::cv::medianBlur<@FILTER_WIDTH, XF_BORDER_REPLICATE, @TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @FILTER_WIDTH
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_median_blur.hpp`, firma detectada para `medianBlur` alrededor de línea 490:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int USE_URAM = 0,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void medianBlur(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
                xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    unsigned short imgheight = _src.rows;
    unsigned short imgwidth = _src.cols;

    assert(BORDER_TYPE == XF_BORDER_REPLICATE && "Only XF_BORDER_REPLICATE is supported");

    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");

    xFMedianNxN<ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, USE_URAM, XFCVDEPTH_IN, XFCVDEPTH_OUT, 0,
                (COLS >> XF_BITSHIFT(NPC)) + (FILTER_SIZE >> 1), FILTER_SIZE, FILTER_SIZE * FILTER_SIZE>(
        _src, _dst, FILTER_SIZE, imgheight, imgwidth);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.medianBlur(@input_0, @FILTER_WIDTH)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @FILTER_WIDTH
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `min_max_loc`

**Funcionalidad:** Compute minimum and maximum values and locations.

**Categoría:** `statistics`

- Inputs: @input_0:image
- Outputs: @output_0:scalar

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_min_max_loc.hpp`
- Source: `vision/L1/include/core/xf_min_max_loc.hpp`
- Function:
```cpp
xf::cv::minMaxLoc<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_min_max_loc.hpp`, firma detectada para `minMaxLoc` alrededor de línea 217:
```cpp

    _minval1 = _minval;
    _maxval1 = _maxval;
}

template <int SRC_T, int ROWS, int COLS, int NPC = 0, int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void minMaxLoc(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
               int32_t* min_value,
               int32_t* max_value,
               uint16_t* _minlocx,
               uint16_t* _minlocy,
               uint16_t* _maxlocx,
               uint16_t* _maxlocy) {
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
// clang-format off
    #pragma HLS inline off
    // clang-format on

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(@input_0)
@output_0 = (min_val, max_val, min_loc, max_loc)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `mode_filter`

**Funcionalidad:** Apply mode filtering.

**Categoría:** `filtering`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `kernel_size` / `@kernel_size` (`integer`): Kernel size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_modefilter.hpp`
- Source: `vision/L1/include/imgproc/xf_modefilter.hpp`
- Function:
```cpp
xf::cv::modefilter<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_modefilter.hpp`, firma detectada para `modefilter` alrededor de línea 487:
```cpp
          int TYPE,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void modefilter(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
                xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    unsigned short imgheight = _src.rows;
    unsigned short imgwidth = _src.cols;

    assert(BORDER_TYPE == XF_BORDER_REPLICATE && "Only XF_BORDER_REPLICATE is supported");

    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");

    xFModeNxN<ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, 0,
              (COLS >> XF_BITSHIFT(NPC)) + (FILTER_SIZE >> 1), FILTER_SIZE, FILTER_SIZE * FILTER_SIZE>(
        _src, _dst, FILTER_SIZE, imgheight, imgwidth);
```

**Implementación funcional/Python:**
- Backend: `scipy`
- Source: `scipy`
- Imports: `from scipy import ndimage; import numpy as np`
- Function:
```python
@output_0 = ndimage.generic_filter(@input_0, lambda values: np.bincount(values.astype(np.int64)).argmax(), size=kernel_size)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `multiply`

**Funcionalidad:** Compute per-pixel image multiplication.

**Categoría:** `arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::multiply<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `multiply` alrededor de línea 758:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void multiply(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& src1,
              xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& src2,
              xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& dst,
              float scale) {
// clang-format off
    #pragma HLS inline off
// clang-format on
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16SC1) || (SRC_T == XF_16SC3)) &&
           "Image type must be XF_8UC1 or XF_8UC3, XF_16SC1, XF_16SC3");
    assert((POLICY_TYPE == XF_CONVERT_POLICY_SATURATE || POLICY_TYPE == XF_CONVERT_POLICY_TRUNCATE) &&
           "_policytype must be 'XF_CONVERT_POLICY_SATURATE' or 'XF_CONVERT_POLICY_TRUNCATE'");
    assert(((scale >= 0) && (scale <= 1)) && "_scale_val must be within the range of 0 to 1");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS) && (src2.rows <= ROWS) && (src2.cols <= COLS)) &&
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.multiply(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `nop`

**Funcionalidad:** Forward input data unchanged without applying any operation.

**Categoría:** `pipeline_control`

- Inputs: @input_0:any
- Outputs: @output_0:same_as_input

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:** no declarada.

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- No hay implementación HLS declarada.
- No hay implementación funcional declarada.

## `otsu_threshold`

**Funcionalidad:** Compute threshold with Otsu method.

**Categoría:** `thresholding`

- Inputs: @input_0:image
- Outputs: @output_0:scalar

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_otsuthreshold.hpp`
- Source: `vision/L1/include/imgproc/xf_otsuthreshold.hpp`
- Function:
```cpp
xf::cv::OtsuThreshold<@TYPE, @ROWS, @COLS, @NPC>(@input_0, threshold);
```
- Tokens de puertos: @input_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_otsuthreshold.hpp`, firma detectada para `OtsuThreshold` alrededor de línea 137:
```cpp

/*********************************************************************
 * Otsuthreshold : Computes the otsu threshold for the input image
 *********************************************************************/

template <int SRC_T, int ROWS, int COLS, int NPC = 1, int USE_URAM = 0, int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void OtsuThreshold(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat, uint8_t& _thresh) {
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif

    uint32_t hist[XF_CHANNELS(SRC_T, NPC)][256];
    uint8_t thresh;

// clang-format off
    #pragma HLS INLINE off
    #pragma HLS interface ap_fifo port=hist
    // clang-format on

    uint16_t width = _src_mat.cols >> (XF_BITSHIFT(NPC));
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
threshold, _ = cv.threshold(@input_0, 0, @maxval, cv.THRESH_BINARY + cv.THRESH_OTSU)
@output_0 = threshold
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @maxval
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `paint_mask`

**Funcionalidad:** Overlay or paint a mask on an image.

**Categoría:** `enhancement`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `color` / `@color` (`array`): Color value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_paintmask.hpp`
- Source: `vision/L1/include/imgproc/xf_paintmask.hpp`
- Function:
```cpp
xf::cv::paintmask<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_paintmask.hpp`, firma detectada para `paintmask` alrededor de línea 115:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_MASK_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void paintmask(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
               xf::cv::Mat<MASK_T, ROWS, COLS, NPC, XFCVDEPTH_MASK_IN>& in_mask,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat,
               unsigned char _color[XF_CHANNELS(SRC_T, NPC)]) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;
    xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), unsigned char> color;
    for (int i = 0; i < XF_CHANNELS(SRC_T, NPC); i++) {
        color.val[i] = _color[i];
    }
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) && "Type must be XF_8UC1");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = @input_0.copy()
@output_0[@input_1 > 0] = @color
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @color
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `phase`

**Funcionalidad:** Compute gradient phase from two gradient images.

**Categoría:** `gradients`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_phase.hpp`
- Source: `vision/L1/include/core/xf_phase.hpp`
- Function:
```cpp
xf::cv::phase<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_phase.hpp`, firma detectada para `phase` alrededor de línea 155:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_X = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_Y = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void phase(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_X>& _src_matx,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN_Y>& _src_maty,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat) {
#ifndef __SYNTHESIS__
    assert(((_src_matx.rows <= ROWS) && (_src_matx.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
    assert(((_src_maty.rows <= ROWS) && (_src_maty.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
    assert(((_src_matx.rows == _src_maty.rows) && (_src_matx.cols == _src_maty.cols)) &&
           "Both input images should have same size");
    assert(((_src_matx.rows == _dst_mat.rows) && (_src_matx.cols == _dst_mat.cols)) &&
           "Input and output image should be of same size");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
#endif

    uint16_t imgwidth = _src_matx.cols >> XF_BITSHIFT(NPC);
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv+numpy`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.phase(@input_0.astype(np.float32), @input_1.astype(np.float32))
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `preprocess`

**Funcionalidad:** Preprocess image data for DNN inference.

**Categoría:** `dnn`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- `out_rows` / `@OUT_ROWS` (`integer`): Output image height.
- `out_cols` / `@OUT_COLS` (`integer`): Output image width.
- `interpolation` / `@INTERPOLATION` (`enum`): Interpolation policy.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `dnn/xf_preprocess.hpp`
- Source: `vision/L1/include/dnn/xf_preprocess.hpp`
- Function:
```cpp
xf::cv::preProcess<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/dnn/xf_preprocess.hpp`, firma detectada para `preProcess` alrededor de línea 105:
```cpp
          int WIDTH_B,
          int IBITS_B,
          int WIDTH_OUT,
          int IBITS_OUT,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void preProcess(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC, XFCVDEPTH_IN>& in_mat,
                xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC, XFCVDEPTH_OUT>& out_mat,
                float params[2 * XF_CHANNELS(IN_TYPE, NPC)]) {
#pragma HLS INLINE OFF

    ap_ufixed<WIDTH_A, IBITS_A, AP_RND> alpha_reg[XF_CHANNELS(IN_TYPE, NPC)];
    ap_fixed<WIDTH_B, IBITS_B, AP_RND> beta_reg[XF_CHANNELS(IN_TYPE, NPC)];

// clang-format off
#pragma HLS ARRAY_PARTITION variable=alpha_reg dim=0 complete
#pragma HLS ARRAY_PARTITION variable=beta_reg dim=0 complete

    // clang-format on
    int channels = XF_CHANNELS(IN_TYPE, NPC);
    for (int i = 0; i < 2 * XF_CHANNELS(IN_TYPE, NPC); i++) {
// clang-format off
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv; import numpy as np`
- Function:
```python
resized = cv.resize(@input_0, (@OUT_COLS, @OUT_ROWS), interpolation=@INTERPOLATION)
@output_0 = resized.astype(np.float32)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @INTERPOLATION, @OUT_COLS, @OUT_ROWS
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `pyr_down`

**Funcionalidad:** Downsample image using Gaussian pyramid.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_pyr_down.hpp`
- Source: `vision/L1/include/imgproc/xf_pyr_down.hpp`
- Function:
```cpp
xf::cv::pyrDown<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_pyr_down.hpp`, firma detectada para `pyrDown` alrededor de línea 89:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM = false,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void pyrDown(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
             xf::cv::Mat<TYPE, (ROWS / 2), (COLS / 2), NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    unsigned short input_height = _src.rows;
    unsigned short input_width = _src.cols;
    xFpyrDownKernel<ROWS, COLS, TYPE, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_CHANNELS(TYPE, NPC), USE_URAM>(
        _src, _dst, input_height, input_width);
    return;
}
} // namespace cv
} // namespace xf
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.pyrDown(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `pyr_down_gaussian_blur`

**Funcionalidad:** Gaussian blur stage for pyramid downsampling.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `filter_width` / `@FILTER_WIDTH` (`integer`): Filter or kernel width.
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.
- `sigma_x` / `@sigma_x` (`number`): Gaussian sigma in X direction.
- `sigma_y` / `@sigma_y` (`number`): Gaussian sigma in Y direction.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_pyr_down_gaussian_blur.hpp`
- Source: `vision/L1/include/imgproc/xf_pyr_down_gaussian_blur.hpp`
- Function:
```cpp
xf::cv::pyrDownGaussianBlur<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `pyrDownGaussianBlur` en `Vitis_Libraries/vision/L1/include/imgproc/xf_pyr_down_gaussian_blur.hpp`.

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.GaussianBlur(@input_0, (@FILTER_WIDTH, @FILTER_WIDTH), sigmaX=@sigma_x, sigmaY=@sigma_y, borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @FILTER_WIDTH, @sigma_x, @sigma_y
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `pyr_up`

**Funcionalidad:** Upsample image using Gaussian pyramid.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_pyr_up.hpp`
- Source: `vision/L1/include/imgproc/xf_pyr_up.hpp`
- Function:
```cpp
xf::cv::pyrUp<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_pyr_up.hpp`, firma detectada para `pyrUp` alrededor de línea 96:
```cpp
template <int TYPE,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void pyrUp(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
           xf::cv::Mat<TYPE, 2 * ROWS, 2 * COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    unsigned short input_height = _src.rows;
    unsigned short input_width = _src.cols;

    xFpyrUpKernel<ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, TYPE, XF_CHANNELS(TYPE, NPC)>(_src, _dst, input_height,
                                                                                              input_width);

    return;
}
} // namespace cv
} // namespace xf
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.pyrUp(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `pyr_up_gaussian_blur`

**Funcionalidad:** Gaussian blur stage for pyramid upsampling.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `filter_width` / `@FILTER_WIDTH` (`integer`): Filter or kernel width.
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.
- `sigma_x` / `@sigma_x` (`number`): Gaussian sigma in X direction.
- `sigma_y` / `@sigma_y` (`number`): Gaussian sigma in Y direction.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_pyr_up_gaussian_blur.hpp`
- Source: `vision/L1/include/imgproc/xf_pyr_up_gaussian_blur.hpp`
- Function:
```cpp
xf::cv::pyrUpGaussianBlur<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `pyrUpGaussianBlur` en `Vitis_Libraries/vision/L1/include/imgproc/xf_pyr_up_gaussian_blur.hpp`.

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.GaussianBlur(@input_0, (@FILTER_WIDTH, @FILTER_WIDTH), sigmaX=@sigma_x, sigmaY=@sigma_y, borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @BORDER_TYPE, @FILTER_WIDTH, @sigma_x, @sigma_y
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `quantization_dithering`

**Funcionalidad:** Apply quantization dithering.

**Categoría:** `enhancement`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `maxval` / `@maxval` (`number`): Maximum output value.
- `quantization_levels` / `@quantization_levels` (`integer`): Number of quantization levels.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_quantizationdithering.hpp`
- Source: `vision/L1/include/imgproc/xf_quantizationdithering.hpp`
- Function:
```cpp
xf::cv::xf_QuatizationDithering<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_quantizationdithering.hpp`, firma detectada para `xf_QuatizationDithering` alrededor de línea 55:
```cpp
          int SCALE_FACTOR,
          int MAX_REPRESENTED_VALUE,
          int NPC,
          int USE_URAM = 0,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void xf_QuatizationDithering(xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& stream_in,
                             xf::cv::Mat<OUT_TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT>& stream_out) {
    enum {
        PLANES = XF_CHANNELS(IN_TYPE, NPC),

        PIXELWIDTH_IN = XF_PIXELWIDTH(IN_TYPE, NPC),
        BITDEPTH_IN = PIXELWIDTH_IN / PLANES,

        PIXELWIDTH_OUT = XF_PIXELWIDTH(OUT_TYPE, NPC),
        BITDEPTH_OUT = PIXELWIDTH_OUT / PLANES,

        QUANTIZATION_INTERVAL = MAX_REPRESENTED_VALUE / SCALE_FACTOR,

        LOG2_SCALE_FACTOR = XF_LOG2(SCALE_FACTOR),
        LOG2_QUANTIZATION_INTERVAL = XF_LOG2(QUANTIZATION_INTERVAL),
        LOG2_MAX_REPRESENTED_VALUE = XF_LOG2(MAX_REPRESENTED_VALUE),
```

**Implementación funcional/Python:**
- Backend: `scikit_image`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = np.round(@input_0.astype(np.float32) * (@quantization_levels - 1) / @maxval) * (@maxval / (@quantization_levels - 1))
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @maxval, @quantization_levels
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `reduce`

**Funcionalidad:** Reduce image along rows or columns.

**Categoría:** `statistics`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- `op` / `@op` (`enum`): Reduction operation.
- `dim` / `@dim` (`integer`): Reduction dimension.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_reduce.hpp`
- Source: `vision/L1/include/imgproc/xf_reduce.hpp`
- Function:
```cpp
xf::cv::reduce<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_reduce.hpp`, firma detectada para `reduce` alrededor de línea 171:
```cpp
          int COLS,
          int ONE_D_HEIGHT,
          int ONE_D_WIDTH,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void reduce(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
            xf::cv::Mat<DST_T, ONE_D_HEIGHT, ONE_D_WIDTH, 1, XFCVDEPTH_OUT>& _dst_mat,
            unsigned char dim) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;

#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8");
    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
#endif

// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xFreduceKernel<SRC_T, DST_T, ROWS, COLS, ONE_D_HEIGHT, ONE_D_WIDTH, XF_DEPTH(SRC_T, NPC), NPC, XFCVDEPTH_IN,
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.reduce(@input_0, @dim, @op)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @dim, @op
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `remap`

**Funcionalidad:** Apply generic geometrical remapping.

**Categoría:** `geometry`

- Inputs: @input_0:image, @input_1:image, @input_2:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `interpolation` / `@INTERPOLATION` (`enum`): Interpolation policy.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_remap.hpp`
- Source: `vision/L1/include/imgproc/xf_remap.hpp`
- Function:
```cpp
xf::cv::remap<@INTERPOLATION, @WIN_ROWS, @TYPE, @OUT_TYPE, @MAP_TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @input_1, @input_2);
```
- Tokens de puertos: @input_0, @input_1, @input_2, @output_0
- Tokens de parámetros: @INTERPOLATION
- Tokens resueltos por composer: @COLS, @MAP_TYPE, @NPC, @OUT_TYPE, @ROWS, @TYPE, @WIN_ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_remap.hpp`, firma detectada para `remap` alrededor de línea 522:
```cpp
          int NPC,
          bool USE_URAM = false,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_Remapped = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_MAPX = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_MAPY = _XFCVDEPTH_DEFAULT>
void remap(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_Remapped>& _remapped_mat,
           xf::cv::Mat<MAP_T, ROWS, COLS, NPC, XFCVDEPTH_MAPX>& _mapx_mat,
           xf::cv::Mat<MAP_T, ROWS, COLS, NPC, XFCVDEPTH_MAPY>& _mapy_mat) {
// clang-format off
     #pragma HLS inline off

// clang-format on

#ifndef __SYNTHESIS__
    assert((MAP_T == XF_32SC1) && "The MAP_T must be XF_32SC1");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) && "The SRC_T must be XF_8UC1 or XF_8UC3");
    assert(((DST_T == XF_8UC1) || (SRC_T == XF_8UC3)) && "The DST_T must be XF_8UC1 or XF_8UC3");
    assert((SRC_T == DST_T) && "Source Mat type and Destination Mat type must be the same");
    assert((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) ||
           (NPC == XF_NPPC8) && "The NPC must be XF_NPPC1 or XF_NPPC2 or XF_NPPC4 or XF_NPPC8");
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.remap(@input_0, @input_1, @input_2, interpolation=@INTERPOLATION)
```
- Tokens de puertos: @input_0, @input_1, @input_2, @output_0
- Tokens de parámetros: @INTERPOLATION
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `reproject_3d`

**Funcionalidad:** Reproject disparity image to 3D coordinates.

**Categoría:** `stereo`

- Inputs: @input_0:image
- Outputs: @output_0:array

**Parámetros declarados:**
- `matrix` / `@matrix` (`matrix`): Transformation/calibration matrix supplied as stage value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_reproject3D.hpp`
- Source: `vision/L1/include/imgproc/xf_reproject3D.hpp`
- Function:
```cpp
xf::cv::reprojectImageTo3D<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `reprojectImageTo3D` en `Vitis_Libraries/vision/L1/include/imgproc/xf_reproject3D.hpp`.

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.reprojectImageTo3D(@input_0, @matrix)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @matrix
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `resize`

**Funcionalidad:** Resize image to another resolution.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `out_rows` / `@OUT_ROWS` (`integer`): Output image height.
- `out_cols` / `@OUT_COLS` (`integer`): Output image width.
- `interpolation` / `@INTERPOLATION` (`enum`): Interpolation policy.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_resize.hpp`
- Source: `vision/L1/include/imgproc/xf_resize.hpp`
- Function:
```cpp
xf::cv::resize<@INTERPOLATION, @TYPE, @ROWS, @COLS, @OUT_ROWS, @OUT_COLS, @NPC, @USE_URAM, @MAXDOWNSCALE>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @INTERPOLATION, @OUT_COLS, @OUT_ROWS
- Tokens resueltos por composer: @COLS, @MAXDOWNSCALE, @NPC, @ROWS, @TYPE, @USE_URAM
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_resize.hpp`, firma detectada para `resize` alrededor de línea 39:
```cpp
          int DST_COLS,
          int NPC,
          bool USE_URAM = false,
          int MAX_DOWN_SCALE,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void resize(xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC, XFCVDEPTH_IN>& _src,
            xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    assert(((INTERPOLATION_TYPE == XF_INTERPOLATION_NN) || (INTERPOLATION_TYPE == XF_INTERPOLATION_BILINEAR) ||
            (INTERPOLATION_TYPE == XF_INTERPOLATION_AREA)) &&
           "Incorrect parameters interpolation type");
    assert(((_src.rows <= SRC_ROWS) && (_src.cols <= SRC_COLS)) &&
           "SRC_ROWS and SRC_COLS should be greater than input image");
    assert(((_dst.rows <= DST_ROWS) && (_dst.cols <= DST_COLS)) &&
           "DST_ROWS and DST_COLS should be greater than output image");

    if (INTERPOLATION_TYPE == XF_INTERPOLATION_AREA) {
        assert((((_src.rows < _dst.rows) && (_src.cols < _dst.cols)) ||
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.resize(@input_0, (@OUT_COLS, @OUT_ROWS), interpolation=@INTERPOLATION)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @INTERPOLATION, @OUT_COLS, @OUT_ROWS
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `rgb_to_bgr`

**Funcionalidad:** Convert RGB image to BGR.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cvt_color.hpp`
- Source: `vision/L1/include/imgproc/xf_cvt_color.hpp`
- Function:
```cpp
xf::cv::rgb2bgr<XF_8UC3, XF_8UC3, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color.hpp`, firma detectada para `rgb2bgr` alrededor de línea 7484:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void rgb2bgr(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
             xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC3) && " RGB image Type must be XF_8UC3");
    assert((DST_T == XF_8UC3) && " BGR image Type must be XF_8UC3");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && " RGB image rows and cols should be less than ROWS, COLS");
    assert(((_dst.cols == _src.cols) && (_dst.rows == _src.rows)) && "BGR and RGB plane dimensions mismatch");

    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           " 1,2,4,8 pixel parallelism is supported  ");
#endif
    xfrgb2bgr<SRC_T, DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_T, NPC),
              XF_WORDWIDTH(DST_T, NPC), ((COLS >> (XF_NPIXPERCYCLE(NPC)))), XF_NPIXPERCYCLE(NPC)>(_src, _dst, _src.rows,
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_RGB2BGR)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `rgb_to_gray`

**Funcionalidad:** Convert RGB image to grayscale.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_cvt_color.hpp`
- Source: `vision/L1/include/imgproc/xf_cvt_color.hpp`
- Function:
```cpp
xf::cv::rgb2gray<XF_8UC3, XF_8UC1, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color.hpp`, firma detectada para `rgb2gray` alrededor de línea 5534:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void rgb2gray(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC3) && " RGB image Type must be XF_8UC3");
    assert((DST_T == XF_8UC1) && " GRAY image Type must be XF_8UC1");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && " RGB image rows and cols should be less than ROWS, COLS");
    assert(((_dst.cols == _src.cols) && (_dst.rows == _src.rows)) && "RGB and GRAY plane dimensions mismatch");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && " 1,8 pixel parallelism is supported  ");
#endif
    xfrgb2gray<SRC_T, DST_T, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT, XF_WORDWIDTH(SRC_T, NPC),
               XF_WORDWIDTH(DST_T, NPC), (ROWS * (COLS >> (XF_NPIXPERCYCLE(NPC))))>(_src, _dst, _src.rows, _src.cols);
}

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_RGB2GRAY)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `rgb_to_hsv`

**Funcionalidad:** Convert RGB image to HSV.

**Categoría:** `color`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_rgb2hsv.hpp`
- Source: `vision/L1/include/imgproc/xf_rgb2hsv.hpp`
- Function:
```cpp
xf::cv::rgb2hsv<XF_8UC3, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_rgb2hsv.hpp`, firma detectada para `rgb2hsv` alrededor de línea 79:
```cpp
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void rgb2hsv(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
             xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    //#pragma HLS DATAFLOW

    int hdiv[256] = {
        0,    122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192,
        7680, 7228,   6827,  6467,  6144,  5851,  5585,  5343,  5120,  4915,  4726,  4551,  4389,  4237, 4096, 3964,
        3840, 3724,   3614,  3511,  3413,  3321,  3234,  3151,  3072,  2997,  2926,  2858,  2793,  2731, 2671, 2614,
        2560, 2508,   2458,  2409,  2363,  2318,  2276,  2234,  2194,  2156,  2119,  2083,  2048,  2014, 1982, 1950,
        1920, 1890,   1862,  1834,  1807,  1781,  1755,  1731,  1707,  1683,  1661,  1638,  1617,  1596, 1575, 1555,
        1536, 1517,   1499,  1480,  1463,  1446,  1429,  1412,  1396,  1381,  1365,  1350,  1336,  1321, 1307, 1293,
        1280, 1267,   1254,  1241,  1229,  1217,  1205,  1193,  1182,  1170,  1159,  1148,  1138,  1127, 1117, 1107,
        1097, 1087,   1078,  1069,  1059,  1050,  1041,  1033,  1024,  1016,  1007,  999,   991,   983,  975,  968,
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.cvtColor(@input_0, cv.COLOR_RGB2HSV)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `rgbir`

**Funcionalidad:** Process RGB-IR sensor data.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_rgbir.hpp`
- Source: `vision/L1/include/imgproc/xf_rgbir.hpp`
- Function:
```cpp
xf::cv::rgbir2bayer<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_rgbir.hpp`, firma detectada para `rgbir2bayer` alrededor de línea 786:
```cpp
          int BORDER_T = XF_BORDER_CONSTANT,
          int USE_URAM = 0,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_0 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_2 = _XFCVDEPTH_DEFAULT>
void rgbir2bayer(xf::cv::Mat<TYPE, ROWS, COLS, NPPC, XFCVDEPTH_IN>& _src,
                 char R_IR_C1_wgts[FSIZE1 * FSIZE1],
                 char R_IR_C2_wgts[FSIZE1 * FSIZE1],
                 char B_at_R_wgts[FSIZE1 * FSIZE1],
                 char IR_at_R_wgts[FSIZE2 * FSIZE2],
                 char IR_at_B_wgts[FSIZE2 * FSIZE2],
                 char sub_wgts[4],
                 xf::cv::Mat<TYPE, ROWS, COLS, NPPC, XFCVDEPTH_OUT_0>& _dst_rggb,
                 xf::cv::Mat<TYPE, ROWS, COLS, NPPC, XFCVDEPTH_OUT_1>& _dst_ir) {
#ifndef __SYNTHESIS__
    assert(((BFORMAT == XF_BAYER_BG) || (BFORMAT == XF_BAYER_GB) || (BFORMAT == XF_BAYER_GR) ||
            (BFORMAT == XF_BAYER_RG)) &&
           ("Unsupported Bayer pattern. Use anyone among: "
            "XF_BAYER_BG;XF_BAYER_GB;XF_BAYER_GR;XF_BAYER_RG"));
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPPC == 1) || (NPPC == 2) || (NPPC == 4)) && "Only 1, 2 and 4 pixel-parallelism are supported");
```

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = @input_0.copy()
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `rgbir_bilinear`

**Funcionalidad:** Process RGB-IR data with bilinear interpolation.

**Categoría:** `isp`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_rgbir_bilinear.hpp`
- Source: `vision/L1/include/imgproc/xf_rgbir_bilinear.hpp`
- Function:
```cpp
xf::cv::rgbir2bayer<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `rgbir2bayer` en `Vitis_Libraries/vision/L1/include/imgproc/xf_rgbir_bilinear.hpp`.

**Implementación funcional/Python:**
- Backend: `numpy`
- Source: `numpy`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = @input_0.copy()
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.

## `rotate`

**Funcionalidad:** Rotate image by supported angles.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `rotate_code` / `@rotate_code` (`enum`): Rotation code.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_rotate.hpp`
- Source: `vision/L1/include/imgproc/xf_rotate.hpp`
- Function:
```cpp
xf::cv::rotate<@INPUT_PTR_WIDTH, @OUTPUT_PTR_WIDTH, @TYPE, @TILE_SIZE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @ROWS, @COLS, @rotate_code);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @rotate_code
- Tokens resueltos por composer: @COLS, @INPUT_PTR_WIDTH, @NPC, @OUTPUT_PTR_WIDTH, @ROWS, @TILE_SIZE, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_rotate.hpp`, firma detectada para `rotate` alrededor de línea 219:
```cpp
    }

    return;
}

template <int INPUT_PTR_WIDTH, int OUTPUT_PTR_WIDTH, int TYPE, int TILE_SZ, int ROWS, int COLS, int NPC>
void rotate(ap_uint<INPUT_PTR_WIDTH>* src_ptr, ap_uint<OUTPUT_PTR_WIDTH>* dst_ptr, int rows, int cols, int direction) {
// clang-format off
		#pragma HLS INLINE OFF
// clang-format on

#ifndef __SYNTHESIS__
    assert(((TYPE == XF_8UC1) || (TYPE == XF_8UC3)) &&
           "Input and Output TYPE must be XF_8UC1 for 1-channel, XF_8UC3 for 3-channel");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2)) && "NPC must be XF_NPPC1, XF_NPPC2 ");
    assert((rows <= ROWS) && (cols <= COLS) && "ROWS and COLS should be greater or equal to input image size ");
#endif

    const uint16_t PXL_WIDTH = XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC));

    const uint16_t ROWS_TC = (ROWS + TILE_SZ - 1) / TILE_SZ;
    const uint16_t COLS_TC = (COLS + TILE_SZ - 1) / TILE_SZ;
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.rotate(@input_0, @rotate_code)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @rotate_code
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `scharr`

**Funcionalidad:** Compute Scharr image gradients.

**Categoría:** `gradients`

- Inputs: @input_0:image
- Outputs: @output_0:image, @output_1:image

**Parámetros declarados:**
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_scharr.hpp`
- Source: `vision/L1/include/imgproc/xf_scharr.hpp`
- Function:
```cpp
xf::cv::Scharr<@BORDER_TYPE, @TYPE, @OUT_TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @output_1);
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: @BORDER_TYPE
- Tokens resueltos por composer: @COLS, @NPC, @OUT_TYPE, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_scharr.hpp`, firma detectada para `Scharr` alrededor de línea 509:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_X = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_Y = _XFCVDEPTH_DEFAULT>
void Scharr(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
            xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_X>& _dst_matx,
            xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_Y>& _dst_maty) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    uint16_t img_height = _src_mat.rows;
    uint16_t img_width = (_src_mat.cols >> XF_BITSHIFT(NPC));
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1 or XF_NPPC8");

    assert((BORDER_TYPE == XF_BORDER_CONSTANT) && "Border type must be XF_BORDER_CONSTANT ");

    assert(((img_height <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.Scharr(@input_0, cv.CV_16S, 1, 0, borderType=@BORDER_TYPE)
@output_1 = cv.Scharr(@input_0, cv.CV_16S, 0, 1, borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: @BORDER_TYPE
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `sgbm`

**Funcionalidad:** Compute disparity using semi-global block matching.

**Categoría:** `stereo`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `num_disparities` / `@num_disparities` (`integer`): Stereo matcher disparity range.
- `block_size` / `@block_size` (`integer`): Stereo matcher block size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_sgbm.hpp`
- Source: `vision/L1/include/imgproc/xf_sgbm.hpp`
- Function:
```cpp
xf::cv::SemiGlobalBM<@TYPE, @ROWS, @COLS, @NPC, @num_disparities, @block_size>(@input_0, @input_1, @output_0);
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @block_size, @num_disparities
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_sgbm.hpp`, firma detectada para `SemiGlobalBM` alrededor de línea 903:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN_L = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_R = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void SemiGlobalBM(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_L>& _src_mat_l,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_R>& _src_mat_r,
                  xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat,
                  uint8_t p1,
                  uint8_t p2) {
#ifndef _SYNTHESIS_
    assert((SRC_T == XF_8UC1) && " WORDWIDTH_SRC must be XF_8UC1 ");
    assert((DST_T == XF_8UC1) && " WORDWIDTH_DST must be XF_8UC1 ");
    assert((NPC == XF_NPPC1) && " NPC must be XF_NPPC1 ");
    assert((WINDOW_SIZE == 5) && " WSIZE must be set to '5' ");
    assert(((NDISP > 1) && (NDISP <= 256)) && " NDISP must be greater than '1' and less than or equal to '256' ");
    assert((NDISP >= PU) && " NDISP must not be lesser than PU (parallel units)");
    assert((((NDISP / PU) * PU) == NDISP) && " NDISP/PU must be a non-fractional number ");
    assert(((R == 2) || (R == 3) || (R == 4)) && "Number of directions R must be '2', '3' or '4' ");
    assert((p1 < p2) && "p1 must be always less than p2");
    assert((p2 <= 100) && "Maximum value of p2 must be 100 ");
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
matcher = cv.StereoSGBM_create(numDisparities=@num_disparities, blockSize=@block_size)
@output_0 = matcher.compute(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @block_size, @num_disparities
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `sobel`

**Funcionalidad:** Compute Sobel image gradients.

**Categoría:** `gradients`

- Inputs: @input_0:image
- Outputs: @output_0:image, @output_1:image

**Parámetros declarados:**
- `filter_width` / `@FILTER_WIDTH` (`integer`): Filter or kernel width.
- `border_type` / `@BORDER_TYPE` (`enum`): Border handling policy.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_sobel.hpp`
- Source: `vision/L1/include/imgproc/xf_sobel.hpp`
- Function:
```cpp
xf::cv::Sobel<@BORDER_TYPE, @FILTER_WIDTH, @TYPE, @OUT_TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @output_1);
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: @BORDER_TYPE, @FILTER_WIDTH
- Tokens resueltos por composer: @COLS, @NPC, @OUT_TYPE, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_sobel.hpp`, firma detectada para `Sobel` alrededor de línea 1873:
```cpp
          int COLS,
          int NPC = 1,
          bool USE_URAM = false,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_X = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_Y = _XFCVDEPTH_DEFAULT>
void Sobel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_X>& _dst_matx,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_Y>& _dst_maty) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    uint16_t width = _src_mat.cols >> XF_BITSHIFT(NPC);
    uint16_t height = _src_mat.rows;

#ifndef __SYNTHESIS__
    assert(((FILTER_TYPE == XF_FILTER_3X3) || (FILTER_TYPE == XF_FILTER_5X5) || (FILTER_TYPE == XF_FILTER_7X7)) &&
           " Filter width must be XF_FILTER_3X3, XF_FILTER_5X5 or XF_FILTER_7X7 ");

    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1 or XF_NPPC8");

```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.Sobel(@input_0, cv.CV_16S, 1, 0, ksize=@FILTER_WIDTH, borderType=@BORDER_TYPE)
@output_1 = cv.Sobel(@input_0, cv.CV_16S, 0, 1, ksize=@FILTER_WIDTH, borderType=@BORDER_TYPE)
```
- Tokens de puertos: @input_0, @output_0, @output_1
- Tokens de parámetros: @BORDER_TYPE, @FILTER_WIDTH
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `stereo_lbm`

**Funcionalidad:** Compute disparity using local block matching.

**Categoría:** `stereo`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `num_disparities` / `@num_disparities` (`integer`): Stereo matcher disparity range.
- `block_size` / `@block_size` (`integer`): Stereo matcher block size.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_stereolbm.hpp`
- Source: `vision/L1/include/imgproc/xf_stereolbm.hpp`
- Function:
```cpp
xf::cv::StereoBM<@TYPE, @ROWS, @COLS, @NPC, @num_disparities, @block_size>(@input_0, @input_1, @output_0);
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @block_size, @num_disparities
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_stereolbm.hpp`, firma detectada para `StereoBM` alrededor de línea 906:
```cpp
          int COLS,
          int NPC,
          bool USE_URAM = false,
          int XFCVDEPTH_left = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_right = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_disp = _XFCVDEPTH_DEFAULT>
void StereoBM(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_left>& _left_mat,
              xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_right>& _right_mat,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH_disp>& _disp_mat,
              xf::cv::xFSBMState<WSIZE, NDISP, NDISP_UNIT>& sbmstate) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xFFindStereoCorrespondenceLBM<ROWS, COLS, SRC_T, DST_T, NPC, XFCVDEPTH_left, XFCVDEPTH_right, XFCVDEPTH_disp, WSIZE,
                                  NDISP, NDISP_UNIT, USE_URAM>(_left_mat, _right_mat, _disp_mat, sbmstate,
                                                               _left_mat.rows, _left_mat.cols);
}
} // namespace cv
} // namespace xf

#endif // _XF_STEREOBM_HPP_
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
matcher = cv.StereoBM_create(numDisparities=@num_disparities, blockSize=@block_size)
@output_0 = matcher.compute(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: @block_size, @num_disparities
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `stereo_pipeline`

**Funcionalidad:** Run stereo preprocessing and matching pipeline.

**Categoría:** `stereo`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_stereo_pipeline.hpp`
- Source: `vision/L1/include/imgproc/xf_stereo_pipeline.hpp`
- Function:
```cpp
xf::cv::stereoPipeline<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
No se localizó automáticamente la firma de `stereoPipeline` en `Vitis_Libraries/vision/L1/include/imgproc/xf_stereo_pipeline.hpp`.

**Implementación funcional/Python:** no declarada.

**Discrepancias / puntos de revisión:**
- Firma HLS no localizada automáticamente.
- No hay implementación funcional declarada.

## `subtract`

**Funcionalidad:** Compute per-pixel image subtraction.

**Categoría:** `arithmetic`

- Inputs: @input_0:image, @input_1:image
- Outputs: @output_0:image

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `core/xf_arithm.hpp`
- Source: `vision/L1/include/core/xf_arithm.hpp`
- Function:
```cpp
xf::cv::subtract<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/core/xf_arithm.hpp`, firma detectada para `subtract` alrededor de línea 998:
```cpp
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void subtract(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_1>& _src1,
              xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN_2>& _src2,
              xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT_1>& _dst) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    uint16_t image_width = _src1.cols >> XF_BITSHIFT(NPC);
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3) || (SRC_T == XF_16SC1) || (SRC_T == XF_16SC3)) &&
           "Image type must be XF_8UC1 or XF_8UC3, XF_16SC1, XF_16SC3");
    assert((POLICY_TYPE == XF_CONVERT_POLICY_SATURATE || POLICY_TYPE == XF_CONVERT_POLICY_TRUNCATE) &&
           "_policytype must be 'XF_CONVERT_POLICY_SATURATE' or 'XF_CONVERT_POLICY_TRUNCATE'");
    assert((_src1.rows <= ROWS) && "ROWS and COLS should be greater than input image");
    assert((_src1.cols <= COLS) && "ROWS and COLS should be greater than input image");
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.subtract(@input_0, @input_1)
```
- Tokens de puertos: @input_0, @input_1, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `sum`

**Funcionalidad:** Compute sum of image pixels.

**Categoría:** `statistics`

- Inputs: @input_0:image
- Outputs: @output_0:scalar

**Parámetros declarados:**
- ninguno

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_sum.hpp`
- Source: `vision/L1/include/imgproc/xf_sum.hpp`
- Function:
```cpp
xf::cv::sum<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_sum.hpp`, firma detectada para `sum` alrededor de línea 95:
```cpp
        scl.val[2] = (ap_uint<64>)internal_sum[2];
    }
    return 0;
}

template <int SRC_T, int ROWS, int COLS, int NPC = 1, int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT>
void sum(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& src1, double sum[XF_CHANNELS(SRC_T, NPC)]) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1)) && "Input TYPE must be XF_8UC1 for 1-channel image");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4, XF_NPPC8 ");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);
    xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), double> scl;

    sumKernel<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC),
              XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(src1, scl, src1.rows, width);
    for (int i = 0; i < XF_CHANNELS(SRC_T, NPC); i++) {
        sum[i] = scl.val[i];
    }
}
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.sumElems(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `svm`

**Funcionalidad:** Run support vector machine classification primitive.

**Categoría:** `classification`

- Inputs: @input_0:array
- Outputs: @output_0:array

**Parámetros declarados:**
- `model` / `@model` (`object`): Model supplied as stage value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_svm.hpp`
- Source: `vision/L1/include/imgproc/xf_svm.hpp`
- Function:
```cpp
xf::cv::SVM<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_svm.hpp`, firma detectada para `SVM` alrededor de línea 100:
```cpp
          int ROWS2,
          int COLS2,
          int NPC = 1,
          int N,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT>
void SVM(xf::cv::Mat<SRC1_T, ROWS1, COLS1, NPC, XFCVDEPTH_IN_1>& in_1,
         xf::cv::Mat<SRC2_T, ROWS2, COLS2, NPC, XFCVDEPTH_IN_2>& in_2,
         uint16_t idx1,
         uint16_t idx2,
         uchar_t frac1,
         uchar_t frac2,
         uint16_t n,
         uchar_t* out_frac,
         ap_int<DST_WIDTH>* result) {
#ifndef __SYNTHESIS__
    assert(((SRC1_T == XF_16SC1)) && "Only 16 bit, single channel images are supported");
    assert(((SRC2_T == XF_16SC1)) && "Only 16 bit, single channel images are supported");
#endif
    ap_int<DST_WIDTH> svm_res =
        xfSVM<SRC1_T, SRC2_T, DST_WIDTH, ROWS1, COLS1, ROWS2, COLS2, NPC, XFCVDEPTH_IN_1, XFCVDEPTH_IN_2, N>(
            in_1, in_2, idx1, idx2, frac1, frac2, n, out_frac);
```

**Implementación funcional/Python:**
- Backend: `scikit_learn`
- Source: `scikit-learn`
- Function:
```python
@output_0 = @model.predict(@input_0)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @model
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `three_d_lut`

**Funcionalidad:** Apply 3D look-up table color transform.

**Categoría:** `enhancement`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `lut` / `@lut` (`array`): Look-up table supplied as stage value.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_3dlut.hpp`
- Source: `vision/L1/include/imgproc/xf_3dlut.hpp`
- Function:
```cpp
xf::cv::lut3d<@TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: ninguno
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_3dlut.hpp`, firma detectada para `lut3d` alrededor de línea 92:
```cpp
          int COLS,
          int NPPC = 1,
          int URAM = 0,
          int XFCVDEPTH_IN_1 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_IN_2 = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT_1 = _XFCVDEPTH_DEFAULT>
void lut3d(xf::cv::Mat<INTYPE, ROWS, COLS, NPPC, XFCVDEPTH_IN_1>& in_img,
           xf::cv::Mat<XF_32FC3, SQLUTDIM, LUTDIM, NPPC, XFCVDEPTH_IN_2>& lut,
           xf::cv::Mat<OUTTYPE, ROWS, COLS, NPPC, XFCVDEPTH_OUT_1>& out_img,
           unsigned char lutdim) {
#ifndef __SYNTHESIS__
    assert(((COLS >= in_img.cols) && (ROWS >= in_img.rows)) &&
           "ROWS and COLS values should be greater than input image rows and columns");
    assert((lutdim <= LUTDIM) && "LUT dimensions should be greater than or equal to lutdim value");
    assert((SQLUTDIM == LUTDIM * LUTDIM) && "SQLUTDIM value should be equal to LUTDIM*LUTDIM");
    assert((INTYPE == XF_8UC3) || (OUTTYPE == XF_8UC3) || (INTYPE == XF_10UC3) || (OUTTYPE == XF_10UC3) ||
           (INTYPE == XF_12UC3) || (OUTTYPE == XF_12UC3) || (INTYPE == XF_16UC3) ||
           (OUTTYPE == XF_16UC3) && "Only XF_8UC3, XF_10UC3, XF_12UC3, XF_16UC3 types are supported");
    assert((NPPC == 1) && "Only 1 pixel parallelism (NPPC=1) is supported");
#endif

#pragma HLS INLINE OFF
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import numpy as np`
- Function:
```python
@output_0 = @lut[@input_0[..., 0], @input_0[..., 1], @input_0[..., 2]]
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @lut
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `threshold`

**Funcionalidad:** Apply fixed-level thresholding.

**Categoría:** `thresholding`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `threshold` / `@threshold` (`number`): Threshold value.
- `maxval` / `@maxval` (`number`): Maximum output value.
- `threshold_type` / `@THRESHOLD_TYPE` (`enum`): Thresholding mode.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_threshold.hpp`
- Source: `vision/L1/include/imgproc/xf_threshold.hpp`
- Function:
```cpp
xf::cv::Threshold<@THRESHOLD_TYPE, @TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @threshold, @maxval);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @THRESHOLD_TYPE, @maxval, @threshold
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_threshold.hpp`, firma detectada para `Threshold` alrededor de línea 121:
```cpp
          int SRC_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void Threshold(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat,
               short int thresh,
               short int maxval) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) && "Type must be XF_8UC1");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8");

    assert(((THRESHOLD_TYPE == XF_THRESHOLD_TYPE_BINARY) || (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_BINARY_INV) ||
            (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_TRUNC) || (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_TOZERO) ||
            (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_TOZERO_INV)) &&
           "_thresh_type must be either XF_THRESHOLD_TYPE_BINARY or XF_THRESHOLD_TYPE_BINARY or "
           "XF_THRESHOLD_TYPE_BINARY_INV or XF_THRESHOLD_TYPE_TRUNC or XF_THRESHOLD_TYPE_TOZERO or "
           "XF_THRESHOLD_TYPE_TOZERO_INV");
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
_, @output_0 = cv.threshold(@input_0, @threshold, @maxval, @THRESHOLD_TYPE)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @THRESHOLD_TYPE, @maxval, @threshold
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

## `warp_transform`

**Funcionalidad:** Apply affine or perspective warp transform.

**Categoría:** `geometry`

- Inputs: @input_0:image
- Outputs: @output_0:image

**Parámetros declarados:**
- `interpolation` / `@INTERPOLATION` (`enum`): Interpolation policy.
- `matrix` / `@matrix` (`matrix`): Transformation/calibration matrix supplied as stage value.
- `dsize` / `@dsize` (`array`): Output size for functional warp implementation.

**Implementación HLS/Vitis:**
- Backend: `vitis_vision`
- Include: `imgproc/xf_warp_transform.hpp`
- Source: `vision/L1/include/imgproc/xf_warp_transform.hpp`
- Function:
```cpp
xf::cv::warpTransform<@STORE_LINES, @START_ROW, @TRANSFORM, @INTERPOLATION, @TYPE, @ROWS, @COLS, @NPC>(@input_0, @output_0, @matrix);
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @INTERPOLATION, @matrix
- Tokens resueltos por composer: @COLS, @NPC, @ROWS, @START_ROW, @STORE_LINES, @TRANSFORM, @TYPE
- Contraste con header real:
Header real `Vitis_Libraries/vision/L1/include/imgproc/xf_warp_transform.hpp`, firma detectada para `warpTransform` alrededor de línea 877:
```cpp
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM = false,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void warpTransform(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& _src_mat,
                   xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT>& _dst_mat,
                   float P_matrix[9]) {
// clang-format off
    #pragma HLS INLINE OFF

	warpTransformKrnl<STORE_LINES, START_ROW, TRANSFORM, INTERPOLATION_TYPE, TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN, XFCVDEPTH_OUT,
                            USE_URAM>(_src_mat, _dst_mat, P_matrix);
}
} // namespace cv
} // namespace xf
#endif
```

**Implementación funcional/Python:**
- Backend: `opencv`
- Source: `opencv`
- Imports: `import cv2 as cv`
- Function:
```python
@output_0 = cv.warpPerspective(@input_0, @matrix, @dsize, flags=@INTERPOLATION)
```
- Tokens de puertos: @input_0, @output_0
- Tokens de parámetros: @INTERPOLATION, @dsize, @matrix
- Tokens resueltos por composer: ninguno

**Discrepancias / puntos de revisión:**
- No se detectan discrepancias estructurales obvias en esta revisión estática.

# Resumen de etapas con puntos de revisión

- Firmas/coincidencias HLS localizadas automáticamente: `90`.
- Firmas HLS no localizadas automáticamente: `27`.

- `aec`: No hay implementación funcional declarada.
- `agc`: Firma HLS no localizada automáticamente. No hay implementación funcional declarada.
- `auto_white_balance`: Firma HLS no localizada automáticamente.
- `awb_normalization`: Firma HLS no localizada automáticamente.
- `bad_pixel_correction`: Firma HLS no localizada automáticamente.
- `canny_sobel`: Firma HLS no localizada automáticamente. No hay implementación funcional declarada.
- `clahe`: Firma HLS no localizada automáticamente.
- `color_convert_extended`: Firma HLS no localizada automáticamente. No hay implementación funcional declarada.
- `connected_components`: Firma HLS no localizada automáticamente.
- `convert_to`: Firma HLS no localizada automáticamente.
- `corner_img_to_list`: No hay implementación funcional declarada.
- `corner_update`: No hay implementación funcional declarada.
- `crop`: Firma HLS no localizada automáticamente.
- `custom_bgr_to_y8`: Firma HLS no localizada automáticamente.
- `delay`: No hay implementación funcional declarada.
- `depth_3d`: Firma HLS no localizada automáticamente. No hay implementación funcional declarada.
- `edge_tracing`: No hay implementación funcional declarada.
- `find_contours`: Firma HLS no localizada automáticamente.
- `hdr_decompand`: Firma HLS no localizada automáticamente.
- `hdr_merge`: Firma HLS no localizada automáticamente.
- `hog_compute_hist`: Firma HLS no localizada automáticamente.
- `hog_gradients`: Firma HLS no localizada automáticamente.
- `hog_hist_norm`: Firma HLS no localizada automáticamente.
- `hog_norm`: Firma HLS no localizada automáticamente.
- `hog_pm`: Firma HLS no localizada automáticamente.
- `isp_stats`: Firma HLS no localizada automáticamente. No hay implementación funcional declarada.
- `lens_shading_correction`: No hay implementación funcional declarada.
- `local_tone_mapping`: Firma HLS no localizada automáticamente.
- `nop`: No hay implementación HLS declarada. No hay implementación funcional declarada.
- `pyr_down_gaussian_blur`: Firma HLS no localizada automáticamente.
- `pyr_up_gaussian_blur`: Firma HLS no localizada automáticamente.
- `reproject_3d`: Firma HLS no localizada automáticamente.
- `rgbir_bilinear`: Firma HLS no localizada automáticamente.
- `stereo_pipeline`: Firma HLS no localizada automáticamente. No hay implementación funcional declarada.
