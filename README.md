# Laboratorio interactivo de máscaras de convolución

Aplicación de escritorio para Computación Gráfica. Carga una imagen, la sube a la GPU como textura y le aplica máscaras de convolución en fragment shaders de OpenGL 3.3 core (moderngl + GLFW). Se cambia de filtro, modo de borde, etapa del pipeline Canny y método de desenfoque en tiempo real con el teclado.

- La ventana mide exactamente lo mismo que la imagen y no se reescala.
- Solo se dibuja un quad de 4 vértices en coordenadas de recorte (±1). No hay matrices, cámara ni transformaciones geométricas.
- Toda la convolución ocurre en los shaders GLSL. La CPU solo calcula los pesos y fija los uniforms.

## Instalación (Windows 11, PowerShell)

Requiere Python 3.10 o superior (probado con 3.13) y un driver de GPU con OpenGL 3.3. Funciona en Windows nativo, sin WSL.

```powershell
cd ruta\al\proyecto
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Para ejecutar los tests:
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Dependencias de ejecución (versiones exactas en `requirements.txt`): moderngl, glfw, Pillow y numpy.

Si al arrancar aparece "Error de OpenGL" (código 3), lo habitual es que falte el driver del fabricante, que la sesión sea de Escritorio Remoto o que Windows esté usando el "Adaptador de pantalla básico de Microsoft".

## Ejecución

```powershell
# Genera las imágenes sintéticas de prueba en assets\images\ (orientacion, escalon, tablero, circulo, ruido)
.venv\Scripts\python.exe tools\make_test_images.py

# Modo interactivo (sin --image usa default_image de config\settings.json)
.venv\Scripts\python.exe main.py --image assets\images\orientacion.png

# Benchmark sin ventana: tabla en consola y results\benchmark_<fecha>.csv
.venv\Scripts\python.exe main.py --benchmark

# Todos los experimentos del informe: results\experimentos_<fecha>\ con manifest.csv
.venv\Scripts\python.exe main.py --experiments --image assets\images\orientacion.png
```

`--quick` (con `--benchmark` o `--experiments`) reduce el benchmark a 1 ejecución de calentamiento y 5 medidas por caso, para pruebas rápidas.

Al arrancar, la consola muestra GL_VERSION, GL_RENDERER, GL_VENDOR y `imagen WxH | framebuffer WxH`.

Códigos de salida: 0 normal, 1 error inesperado, 2 error de imagen (no existe, no es válida o supera GL_MAX_TEXTURE_SIZE), 3 error de OpenGL/GLFW, 4 error de shader, 130 Ctrl+C.

## Controles

| Tecla | Acción |
|---|---|
| 1–8 | Kernel: 1 identidad, 2 box, 3 gaussiano, 4 sharpen, 5 Sobel Gx, 6 Sobel Gy, 7 Laplaciano, 8 emboss |
| 0 | Kernel personalizado (`custom_kernel` de `config\settings.json`) |
| B | Modo de borde: zero → replicate → reflect → wrap |
| N | Normalización on/off (dividir por la suma si no es 0) |
| V | Mapeo de visualización en modo Kernel: recorte → \|v\| → offset 0.5 + 0.5·v |
| M | Modo: Kernel → Desenfoque → Unsharp |
| X | Perfil del modo Desenfoque: gaussiano / box de varianza equivalente |
| Arriba / Abajo | σ del desenfoque (0.5 a 10, paso 0.5); en Canny, σ del gaussiano de Canny |
| S | Desenfoque 2D directo / separable (Desenfoque, Unsharp y Canny) |
| RePág / AvPág | α de Unsharp (0 a 5, paso 0.25) o umbral alto de Canny (paso 0.05) |
| C | Pipeline Canny on/off (al salir vuelve al modo anterior) |
| Izquierda / Derecha | Etapa de Canny: Original, Gris, Blur H, Blur, Gx, Gy, Magnitud, Dirección, NMS, Clasificación, Bordes |
| G | Guardar captura del resultado visible en `results\` |
| P | Benchmark 2D frente a separable con el borde activo (CSV en `results\`) |
| R | Recargar `config\settings.json` |
| H | Mostrar los controles en consola |
| Esc | Salir |

El título de la ventana muestra siempre el estado: modo, kernel o etapa, borde, normalización, mapeo, σ, α, umbrales, 2D/separable y el tiempo de GPU del último procesado. Las flechas y RePág/AvPág responden a la autorrepetición. Se usa RePág/AvPág en lugar de +/- porque en el teclado español esas teclas no están donde GLFW las espera.

### Kernel personalizado

Edita `config\settings.json` (matriz 3x3, fila superior primero) y pulsa 0. Con el kernel personalizado activo, al guardar el archivo se recarga solo en menos de 1 s. Si el JSON tiene un error, la consola indica la línea y la columna y se conserva la configuración anterior. Las demás claves (`sigma`, `canny_sigma`, `alpha`, `canny_high`, `canny_low_ratio`) fijan los valores iniciales.

### Capturas

G guarda un PNG del tamaño de la imagen con el mapeo aplicado, con nombres del tipo `results\20250307-090502_kernel_k5-sobel-gx_wrap_norm-si_vista-abs.png` (fecha_modo_kernel o etapa_borde_parámetros). La carpeta se crea si falta y la ruta se imprime en consola. La captura se dibuja con el mismo `display.frag` que la pantalla, así que es idéntica a lo que se ve.

## Reproducción de los experimentos

`main.py --experiments` genera todo en una carpeta nueva `results\experimentos_<fecha>\`. `manifest.csv` recoge los parámetros de cada archivo: experimento, modo, kernel o etapa, borde, σ, perfil, α, umbrales y método. E2, E3 y E5 usan el borde replicate, que no oscurece el marco; E1 recorre los 4. El resto de parámetros son los valores por defecto (normalización activa, σ de Canny 1.4, umbral alto 0.3 y bajo 0.12, separable).

| Exp. | Archivos | A mano en la app |
|---|---|---|
| E1. 4 modos de borde con el mismo filtro | `E1_*`: gaussiano σ = 8, Laplaciano y Sobel Gx × 4 bordes (12 PNG) | M (Desenfoque) con σ = 8 o tecla 7/5, y B para recorrer los bordes. Fíjate en el marco de la imagen. |
| E2. Box frente a gaussiano | `E2_*`: gaussiano y box de varianza equivalente con σ = 1, 2, 4, 8 (8 PNG) | M (Desenfoque), Arriba/Abajo para σ y X para el perfil. |
| E3. Sharpen frente a unsharp masking | `E3_*`: preset sharpen y unsharp con α = 0.5, 1, 2 y σ = 1, 2 (7 PNG) | Tecla 4; después M dos veces (Unsharp), RePág/AvPág para α y Arriba/Abajo para σ. |
| E4. Gaussiano 2D directo frente a separable | `E4_benchmark.csv` | Tecla P, o `main.py --benchmark`. S cambia el método en vivo y el título muestra el tiempo de GPU. |
| E5. Sobel solo frente a Canny completo | `E5_*`: Sobel Gx y Gy sin suavizado y las 11 etapas de Canny (13 PNG) | Teclas 5 y 6; después C y Izquierda/Derecha. |

### Notas del benchmark

- Se miden en la GPU, con `ctx.query(time=True)`, solo las pasadas de desenfoque: 1 de `blur_2d.frag` con (2r+1)² lecturas por píxel, o 2 de `blur_1d.frag` con 2·(2r+1).
- Para cada σ ∈ {1, 2, 3, 5, 8, 10} y cada método: 5 ejecuciones de calentamiento y 50 medidas. El CSV tiene las columnas fecha, gpu, W, H, borde, sigma, r, metodo, lecturas_px, media_ms, mediana_ms, desv_ms, min_ms y aceleracion (media 2D / media separable).
- La aceleración medida suele ser menor que la teórica, (2r+1)/2, porque la caché de texturas favorece al 2D. En una AMD Radeon 680M a 640x480, con σ = 10 (r = 30), se midió unas 28 veces, frente a 30.5 teóricas.
- Para resultados estables, mide con el portátil enchufado, el perfil de energía de alto rendimiento y sin otras aplicaciones usando la GPU. Mide siempre la misma imagen.
- Si el driver no admite consultas de tiempo, se avisa y se continúa sin tiempos; en ese caso no se guarda CSV.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Los tests comparan cada shader con una referencia en numpy (error ≤ 1e-4) y comprueban las restricciones del curso sobre el código: sin `mat2/3/4`, sin `texture(`, un único vertex shader sin uniforms y un quad de 4 vértices en ±1. Los tests de GPU usan un contexto OpenGL 3.3 sin ventana; si no hay ninguno disponible, se saltan e indican el motivo. La relación entre requisitos y tests está en `.kiro\specs\laboratorio-convolucion\tasks.md` (matriz de trazabilidad).

## Limitaciones conocidas

- **Umbral de Canny dependiente de σ.** Los umbrales son absolutos sobre |G|/4, la fracción de un escalón negro→blanco ideal sin suavizar. El gaussiano previo atenúa el pico de |G|: un escalón ideal da |G|/4 máx. ≈ 0.89 con σ = 0.5, 0.64 con σ = 1, 0.51 con σ = 1.4 y 0.38 con σ = 2. En `orientacion.png` (contraste 0.81), el máximo con σ = 1.4 es 0.41. Por eso el umbral alto por defecto es 0.3 (bajo 0.12). Al subir σ conviene bajar el umbral con AvPág, o la etapa Bordes se queda negra.
- **Canny sin histéresis iterativa.** Un píxel débil solo se acepta si alguno de sus 8 vecinos es fuerte; no se propagan cadenas largas de débiles (fuera de alcance).
- **Sin orientación EXIF.** Las fotos de móvil pueden verse giradas, porque rotar sería una transformación geométrica. Prepara la imagen girada de antemano.
- **Escalado de DPI de Windows.** La ventana se crea a W×H píxeles físicos sin reescalar. Con escalado del 125 % o más, se ve más pequeña que otras ventanas. Si el framebuffer no coincide con W×H, la consola avisa y se sigue dibujando sin reescalar.
- **Imágenes mayores que el monitor.** Se avisa y se continúa; parte de la ventana puede quedar fuera de la pantalla. Las capturas y los experimentos salen siempre a resolución completa. Si la imagen supera GL_MAX_TEXTURE_SIZE, termina con código 2.
- **Valores sRGB sin linealizar** y gris Rec.601 (como OpenCV). El alfa de entrada se ignora y la salida siempre es opaca.
- **Consola.** Con la salida redirigida a un archivo, Windows usa la codificación cp1252. Por eso la consola escribe "sigma" en lugar de σ; el título de la ventana sí usa σ y α.
