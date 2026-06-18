# Etapas directas para pipelines clásicos de procesamiento de imágenes

Este listado resume las etapas del catálogo que tendrían una aplicación directa en un pipeline clásico de procesamiento de imágenes.

## Control de pipeline

- `nop`: no aplica ninguna operación; permite dejar vacío un punto del pipeline.

## Conversión de color

- `bgr_to_gray`: convierte BGR a escala de grises.
- `rgb_to_gray`: convierte RGB a escala de grises.
- `gray_to_bgr`: convierte escala de grises a BGR.
- `gray_to_rgb`: convierte escala de grises a RGB.
- `bgr_to_rgb`: convierte BGR a RGB.
- `rgb_to_bgr`: convierte RGB a BGR.
- `bgr_to_hsv`: convierte BGR a HSV.
- `rgb_to_hsv`: convierte RGB a HSV.

## Geometría

- `resize`: redimensiona la imagen.
- `crop`: recorta una región de interés.
- `flip`: invierte la imagen.
- `rotate`: rota la imagen.
- `warp_transform`: aplica una transformación geométrica mediante matriz.
- `remap`: aplica un remapeo mediante mapas de coordenadas.
- `pyr_down`: reduce resolución mediante pirámide.
- `pyr_up`: aumenta resolución mediante pirámide.

## Filtrado

- `gaussian_filter`: suavizado gaussiano.
- `median_blur`: filtrado de mediana.
- `bilateral_filter`: suavizado preservando bordes.
- `box_filter`: filtrado promedio/local.
- `custom_convolution`: convolución con kernel definido.

## Umbralización y segmentación básica

- `threshold`: umbralización fija.
- `otsu_threshold`: umbralización automática mediante Otsu.
- `in_range`: segmentación por rango de valores/canales.
- `color_thresholding`: segmentación por umbral de color.
- `connected_components`: etiquetado de componentes conectados.
- `distance_transform`: transformada de distancia.

## Morfología

- `erode`: erosión morfológica.
- `dilate`: dilatación morfológica.

## Gradientes y bordes

- `sobel`: gradientes Sobel.
- `scharr`: gradientes Scharr.
- `magnitude`: magnitud del gradiente.
- `phase`: fase/orientación del gradiente.
- `canny`: detección de bordes Canny.

## Canales y composición

- `channel_extract`: extrae un canal.
- `channel_combine`: combina canales.
- `duplicate_image`: duplica una imagen para ramas paralelas.

## Mejora y ajuste de imagen

- `hist_equalize`: ecualización de histograma.
- `clahe`: ecualización adaptativa limitada por contraste.
- `lut`: aplica una tabla LUT.
- `gamma_correction`: corrección gamma.
- `de_gamma`: corrección gamma inversa.
- `convert_scale_abs`: escala, aplica valor absoluto y convierte.
- `convert_to`: conversión con escala/desplazamiento.
- `convert_bitdepth`: conversión de profundidad de bits.

## Estadística de imagen

- `histogram`: calcula histograma.
- `mean_stddev`: calcula media y desviación estándar.
- `min_max_loc`: obtiene mínimos/máximos y sus posiciones.
- `sum`: suma de valores.
- `reduce`: reducción por filas o columnas.
- `integral_image`: imagen integral.

## Características clásicas

- `find_contours`: detección de contornos.
- `hough_lines`: detección de líneas mediante Hough.
- `hog_descriptor`: descriptor HOG.
- `bfmatcher`: emparejamiento por fuerza bruta.

## Estéreo y profundidad

- `stereo_lbm`: correspondencia estéreo por block matching.
- `sgbm`: correspondencia estéreo semi-global.
- `reproject_3d`: reproyección de disparidad a 3D.
