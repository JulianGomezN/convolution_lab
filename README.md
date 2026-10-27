# Laboratorio interactivo de máscaras de convolución

Aplicación de escritorio para Computación Gráfica. Carga una imagen, la sube a la GPU como textura y le aplica máscaras de convolución en fragment shaders de OpenGL 3.3 core (moderngl + GLFW). Se cambia de filtro, modo de borde, etapa del pipeline Canny y método de desenfoque en tiempo real con el teclado.

- La imagen se procesa y se dibuja a su resolución original; la ventana tiene un
  ancho mínimo de 1280 px para que el título muestre completo el estado actual.
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

Al arrancar, la consola muestra GL_VERSION, GL_RENDERER, GL_VENDOR y `imagen WxH | ventana WxH`.

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

La imagen se dibuja centrada en la ventana. Las franjas restantes son intencionales y sirven
para que el título de estado tenga espacio sin deformar la imagen.

### Ideas para mejorar las franjas

- Usar un fondo gris azulado oscuro en lugar de negro puro, para que el contraste sea menos agresivo.
- Aplicar una versión ampliada y desenfocada de la propia imagen como fondo, manteniendo la imagen
  principal nítida al centro.
- Mostrar una cuadrícula o un patrón sutil en las franjas para que parezcan parte del visor.
- Mantener el fondo negro cuando se analizan bordes, ya que ofrece una referencia neutra y no altera
  la lectura de los resultados.

Actualmente las franjas laterales incluyen un HUD discreto con el nombre, la resolución y el modo
activo. El HUD solo pertenece a la ventana interactiva; las capturas siguen teniendo exactamente
el tamaño de la imagen y no incluyen esa información.

El marco de la imagen también cambia de color según el borde activo: rojo para `zero`, verde para
`replicate`, azul para `reflect` y violeta para `wrap`. Esto permite comparar visualmente las
opciones de borde sin confundirlas con el fondo.

### Kernel personalizado

Edita `config\settings.json` (matriz 3x3, fila superior primero) y pulsa 0. Con el kernel personalizado activo, al guardar el archivo se recarga solo en menos de 1 s. Si el JSON tiene un error, la consola indica la línea y la columna y se conserva la configuración anterior. Las demás claves (`sigma`, `canny_sigma`, `alpha`, `canny_high`, `canny_low_ratio`) fijan los valores iniciales.

### Capturas

G guarda un PNG del tamaño de la imagen con el mapeo aplicado, con nombres del tipo `results\20250307-090502_kernel_k5-sobel-gx_wrap_norm-si_vista-abs.png` (fecha_modo_kernel o etapa_borde_parámetros). La carpeta se crea si falta y la ruta se imprime en consola. La captura se dibuja con el mismo `display.frag` que la pantalla, así que es idéntica a lo que se ve.

## Vídeo

Además de imágenes fijas, la app acepta animaciones y vídeos con `--video RUTA`. El vídeo es solo otra fuente de entrada: cada frame se sube con un único `texture.write` a la misma textura de entrada W×H y pasa por los mismos shaders, el panel, la lupa y las capturas. La CPU solo decodifica; no filtra, no reescala ni opera entre píxeles vecinos. Se ignora el audio.

### Instalación opcional

Los GIF, APNG y WebP animados se leen con Pillow y no necesitan nada más. Para MP4, AVI, MKV, MOV y WebM hay que instalar `requirements-video.txt` después de `requirements.txt`. El modo imagen no lo necesita.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Solo para vídeo real (MP4, AVI, MKV, MOV, WebM):
.venv\Scripts\python.exe -m pip install -r requirements-video.txt
```

Versiones fijadas en `requirements-video.txt`: imageio==2.38.1 e imageio-ffmpeg==0.6.0. Las ruedas de imageio-ffmpeg para Windows incluyen el ejecutable de ffmpeg, así que no hace falta instalarlo aparte. hypothesis, que usan los tests de propiedades, está en `requirements-dev.txt`.

| Backend | Extensiones | Requiere `requirements-video.txt` |
|---|---|---|
| Pillow | .gif .png .apng .webp | No |
| imageio-ffmpeg | .mp4 .avi .mkv .mov .webm | Sí |

Las extensiones se comparan sin distinguir mayúsculas. Un PNG estático abierto con `--video` se trata como un vídeo de 1 frame.

### Ejecución

```powershell
# Genera la animación de prueba assets\videos\demo.gif (320x240, 60 frames de 50 ms)
.venv\Scripts\python.exe tools\make_test_video.py

# Reproduce la animación (solo Pillow, sin requirements-video.txt)
.venv\Scripts\python.exe main.py --video assets\videos\demo.gif

# Benchmark y experimentos sobre el frame 0 del vídeo
.venv\Scripts\python.exe main.py --benchmark --video assets\videos\demo.gif
.venv\Scripts\python.exe main.py --experiments --video assets\videos\demo.gif
```

El generador es determinista: dos ejecuciones dan los mismos frames y el mismo contenido de píxeles. La animación dura 3 s (20 fps) e incluye la "F" de `orientacion.png` desplazándose, una barra que gira, un marcador rojo fijo y un contador binario en la fila inferior que identifica el frame.

`--video` y `--image` son excluyentes. `--image` no cambia: con un GIF animado carga el primer frame como imagen fija, y con un `.mp4` termina con código 2 y pide usar `--video RUTA`.

Al abrir el vídeo, la consola muestra una línea como `Vídeo: demo.gif | 320x240 | 60 frames | 20.00 fps | backend Pillow`. El vídeo arranca reproduciéndose y con el bucle activado, a su velocidad real: si la GPU o el decodificado van por detrás, se saltan frames en lugar de ralentizar.

| Tecla | Acción |
|---|---|
| Espacio | Pausa / reproducir (en pausa en el último frame sin bucle, vuelve al frame 0) |
| . (punto) | Pausa y avanza un frame |
| , (coma) | Pausa y retrocede un frame |
| Inicio | Vuelve al frame 0 sin cambiar la pausa ni el bucle |
| O | Bucle on/off |

Las demás teclas funcionan igual que con una imagen, y los cambios de filtro se ven en vivo. Con una imagen fija estas cinco teclas no hacen nada. Punto y coma están en la misma posición en los teclados español y estadounidense.

El panel añade una sección "Vídeo" con el frame actual (`frame 12/240`, base 0, o `frame 12/?` si el total es desconocido), el instante de inicio del frame (`0.40 s`), los FPS nominales y efectivos (el efectivo muestra "—" en pausa o durante el primer segundo), "Reproduciendo" o "En pausa" y "bucle: sí/no". El título añade el frame y "pausa" o "reproduciendo". Con una imagen fija el panel y el título no cambian.

Las capturas (G) de un vídeo llevan el índice del frame con 5 dígitos, por ejemplo `results\20250307-090502_kernel_k5-sobel-gx_wrap_norm-si_vista-abs_f00012.png`. Guardar no pausa la reproducción.

`--benchmark` y `--experiments` aceptan `--video`, decodifican solo el frame 0 y lo usan como imagen de entrada. Los PNG de los experimentos son los mismos que con `--image`; `benchmark_<fecha>.csv` y `manifest.csv` añaden las columnas `video` (ruta) y `frame` (0).

### Sin el backend de vídeo

Si se abre un `.mp4`, `.avi`, `.mkv`, `.mov` o `.webm` sin haber instalado `requirements-video.txt`, la app no crea la ventana y termina con código 2. La consola muestra algo así:

```text
Error de vídeo: No se pudo abrir el vídeo.
Ruta: assets\videos\clip.mp4
Motivo: falta(n) imageio, imageio-ffmpeg; instálalo(s) con pip install -r requirements-video.txt (.venv\Scripts\python.exe -m pip install -r requirements-video.txt)
```

Se resuelve con:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-video.txt
```

Los demás errores de apertura (ruta inexistente o que no es un archivo, extensión no admitida, primer frame ilegible o tamaño mayor que GL_MAX_TEXTURE_SIZE) también empiezan por "Error de vídeo:" y terminan con código 2. Las imágenes y las animaciones GIF, APNG y WebP siguen funcionando sin imageio, que no se importa al arrancar.

### Dependencia opcional y licencias

El requisito R1.4 del spec original (solo moderngl, glfw, Pillow y numpy) se relaja únicamente para esta dependencia opcional. Sin `requirements-video.txt`, el proyecto cumple R1.4 tal cual y `requirements.txt` sigue teniendo esas cuatro dependencias.

- imageio: licencia BSD-2-Clause.
- imageio-ffmpeg: licencia BSD-2-Clause; el binario de ffmpeg que incluye se distribuye con licencia LGPL/GPL.

### Supuestos

- V1. Pillow siempre y el backend de vídeo como opción, con imageio e imageio-ffmpeg fijados en `requirements-video.txt`, separado de `requirements.txt`. R1.4 se relaja solo para esta dependencia.
- V2. El vídeo se abre con `--video RUTA`, excluyente con `--image`. `--image` no cambia: con un GIF animado carga el primer frame como imagen fija.
- V3. Se ignora el audio.
- V4. El tamaño W×H es el del primer frame y es constante. Pillow compone cada frame de la animación sobre el lienzo completo.
- V5. Si el archivo no declara FPS válidos (≤ 0, no finitos o > 240), se usan 25 fps. En animaciones se usa la duración de cada frame; las duraciones ≤ 0 ms cuentan como 100 ms, igual que en los navegadores.
- V6. La reproducción sigue el reloj de pared: si va por detrás, se saltan frames en vez de ralentizar. Los frames saltados se decodifican pero no se suben a la GPU.
- V7. Mientras se reproduce, se redibuja cada vez que cambia el frame actual. En pausa se vuelve al render bajo demanda.
- V8. Teclas nuevas: Espacio, punto, coma, Inicio y O. No chocan con las existentes y punto y coma están en el mismo sitio en los teclados ES y US.
- V9. El vídeo arranca reproduciéndose con el bucle activado. No hay control de velocidad.
- V10. `--benchmark` y `--experiments` aceptan `--video` y usan el frame 0 como imagen de entrada.
- V11. No se garantiza tiempo real por encima de 1920×1080 a 30 fps. El límite duro sigue siendo GL_MAX_TEXTURE_SIZE; el FPS efectivo del panel deja ver el coste de cada modo.
- V12. Retroceder un frame con el backend de vídeo puede requerir reabrir el archivo y avanzar hasta el índice pedido, sin latencia máxima fijada, porque ffmpeg no ofrece acceso aleatorio exacto por frame en todos los contenedores.

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
- **Escalado de DPI de Windows.** La ventana se crea con el tamaño solicitado en píxeles físicos. Con escalado del 125 % o más, el framebuffer puede no coincidir; la consola avisa y la imagen se sigue dibujando a su resolución original.
- **Imágenes mayores que el monitor.** Se avisa y se continúa; parte de la ventana puede quedar fuera de la pantalla. Las capturas y los experimentos salen siempre a resolución completa. Si la imagen supera GL_MAX_TEXTURE_SIZE, termina con código 2.
- **Valores sRGB sin linealizar** y gris Rec.601 (como OpenCV). El alfa de entrada se ignora y la salida siempre es opaca.
- **Consola.** Con la salida redirigida a un archivo, Windows usa la codificación cp1252. Por eso la consola escribe "sigma" en lugar de σ; el título de la ventana sí usa σ y α.
