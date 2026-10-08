# Requisitos: Laboratorio interactivo de máscaras de convolución

Aplicación de escritorio (Python + moderngl + GLFW, OpenGL 3.3 core, Windows 11 nativo) que carga una imagen, la sube a la GPU como textura y le aplica máscaras de convolución en fragment shaders. Permite cambiar de filtro, modo de borde y etapa del pipeline Canny en tiempo real con el teclado.

Objetivo pedagógico: mostrar que los pesos del kernel determinan el efecto, cómo influye el tratamiento de bordes y cómo se encadenan convoluciones en un pipeline de detección de bordes.

Formato EARS: WHEN / IF…THEN / WHILE / THE SYSTEM SHALL.

## Supuestos documentados

| # | Supuesto | Motivo |
|---|---|---|
| S1 | Los kernels se aplican como correlación, sin voltearlos. La fila 0 del kernel corresponde a los vecinos superiores. | Convención habitual en procesamiento de imágenes; solo cambia el signo en Sobel y emboss. |
| S2 | `reflect` no repite el borde (dcb\|abcd\|cba), igual que `numpy.pad(mode="reflect")`. `zero` devuelve 0 en RGB. | Los 4 modos equivalen a `constant/edge/reflect/wrap` de numpy y se verifican automáticamente. |
| S3 | Se ignora el canal alfa y la salida siempre tiene alfa = 1. | Evita capturas transparentes. |
| S4 | Se trabaja con valores sRGB sin linealizar. El gris usa Rec.601 (0.299, 0.587, 0.114). | Igual que OpenCV, así los resultados son comparables. |
| S5 | Las texturas se guardan con la fila 0 arriba (como `img[y, x]` en numpy). La inversión vertical ocurre solo en `display.frag`. | Ver decisión D2 en `design.md`. |
| S6 | La edición manual del kernel se hace con `config/settings.json`, recarga con R y recarga automática por mtime. | Más precisa y fácil de documentar que la edición celda a celda. |
| S7 | Umbrales de Canny absolutos sobre la magnitud normalizada (\|G\|/4): alto = 0.3 y bajo = 0.4·alto (0.12), ambos ajustables. | Calcular el máximo de la imagen requiere una reducción en GPU, fuera de alcance. Se elige 0.3 y no 0.6 porque el gaussiano atenúa la magnitud: tras él, un escalón negro→blanco ideal da \|G\|/4 ≈ 0.51 con σ = 1.4, y `orientacion.png` llega a 0.41. Con alto = 0.6 la etapa Bordes saldría negra. El umbral efectivo depende de σ (ver D3 en `design.md`). |
| S8 | σ de 0.5 a 10, con radio r = ceil(3σ) ≤ 30 (61 pesos como máximo). El box usa varianza equivalente, r = round(√3·σ). | Comparar box y gaussiano con la misma "fuerza" de desenfoque. |
| S9 | No se aplica la orientación EXIF. | Rotar sería una transformación. Se recomienda preparar las imágenes antes. |
| S10 | Si la imagen es más grande que el monitor, se avisa y se continúa. Las capturas salen siempre a resolución completa. | Reescalar violaría la restricción de tamaño de ventana. |
| S11 | Se usa RePág/AvPág en lugar de +/-. | En el teclado español, +/- no ocupan la posición US que espera GLFW. |
| S12 | El render es bajo demanda: solo se redibuja cuando cambia el estado. | No consume GPU en reposo; el benchmark mide aparte. |

## R1. Contexto OpenGL y plataforma

Historia: como estudiante, quiero que la app use OpenGL 3.3 core en Windows 11 nativo para cumplir las restricciones del curso.

- 1.1 WHEN la aplicación arranca THE SYSTEM SHALL solicitar a GLFW un contexto OpenGL 3.3 core profile forward-compatible y crear sobre él el contexto moderngl.
- 1.2 WHEN se ha creado el contexto THE SYSTEM SHALL imprimir GL_VERSION, GL_RENDERER y GL_VENDOR en consola.
- 1.3 IF GLFW no se inicializa, la ventana no se crea o `ctx.version_code < 330` THEN THE SYSTEM SHALL mostrar un mensaje en español con la causa probable (driver, Escritorio Remoto, adaptador básico de Microsoft) y terminar con código 3, sin traceback.
- 1.4 THE SYSTEM SHALL depender en ejecución solo de Python ≥ 3.10, moderngl, glfw, Pillow y numpy, con versiones exactas en `requirements.txt`.
- 1.5 THE SYSTEM SHALL ejecutarse con `python main.py` desde PowerShell en Windows 11, sin WSL.
- Verificación: salida de consola al arrancar y test `test_constraints.py` (`version_code >= 330`).

## R2. Sin transformaciones geométricas

Historia: como profesor, quiero comprobar que no hay transformaciones geométricas para que el proyecto se centre en la convolución.

- 2.1 THE SYSTEM SHALL dibujar solo un quad de 4 vértices en (±1, ±1), en coordenadas de recorte, como TRIANGLE_STRIP.
- 2.2 THE SYSTEM SHALL usar un único vertex shader (`fullscreen.vert`) con `gl_Position = vec4(in_pos, 0.0, 1.0)` y sin uniforms.
- 2.3 THE SYSTEM SHALL NOT declarar tipos `mat2/mat3/mat4` ni uniforms de transformación, cámara o proyección en ningún shader.
- 2.4 THE SYSTEM SHALL leer texels solo con `texelFetch`, usando coordenadas enteras derivadas de `gl_FragCoord`.
- Verificación: un test analiza los shaders (regex `\bmat[234]\b`, uniforms en `.vert`, ausencia de `texture(`) y comprueba que el VBO tiene 4 vértices en ±1.

## R3. Ventana del tamaño de la imagen

Historia: como estudiante, quiero ver la imagen a resolución nativa para que ningún reescalado contamine el resultado del filtro.

- 3.1 WHEN se carga una imagen W×H THE SYSTEM SHALL crear una ventana no redimensionable con área cliente W×H y viewport (0, 0, W, H).
- 3.2 WHEN se ha creado la ventana THE SYSTEM SHALL comparar el tamaño del framebuffer con W×H. IF difieren THEN THE SYSTEM SHALL avisar y seguir dibujando con viewport W×H, sin reescalar.
- 3.3 IF W o H supera `GL_MAX_TEXTURE_SIZE` THEN THE SYSTEM SHALL terminar con código 2. IF supera el área de trabajo del monitor THEN THE SYSTEM SHALL avisar y continuar.
- 3.4 THE SYSTEM SHALL crear todas las texturas intermedias y de captura con tamaño exactamente W×H.
- Verificación: la consola muestra "imagen 640x480 | framebuffer 640x480" y la captura PNG mide W×H.

## R4. Carga de imagen

Historia: como estudiante, quiero cargar mis propias imágenes y recibir mensajes claros si algo falla.

- 4.1 WHEN se ejecuta con `--image RUTA` THE SYSTEM SHALL cargar la imagen con Pillow, convertirla a RGBA de 8 bits y subirla como textura W×H.
- 4.2 WHEN se ejecuta sin `--image` THE SYSTEM SHALL usar la imagen por defecto definida en `settings.json`.
- 4.3 IF el archivo no existe, no es legible o no es una imagen válida THEN THE SYSTEM SHALL mostrar la ruta, el motivo y las imágenes disponibles en `assets/images/`, y terminar con código 2 antes de crear la ventana.
- 4.4 THE SYSTEM SHALL mostrar la imagen con la misma orientación que un visor de imágenes, sin espejo vertical.
- 4.5 THE SYSTEM SHALL ignorar el alfa de entrada y producir alfa = 1.
- Verificación: tests unitarios de errores e inspección visual de `orientacion.png` (una "F" asimétrica).

## R5. Cómputo en GPU

Historia: como profesor, quiero que todo el cálculo de convolución ocurra en shaders GLSL.

- 5.1 THE SYSTEM SHALL calcular convoluciones, gris, Sobel, NMS, umbral y unsharp exclusivamente en fragment shaders. La CPU solo calcula pesos y fija uniforms.
- 5.2 THE SYSTEM SHALL aplicar los kernels como correlación: fila r → dy = r−1, columna c → dx = c−1, con y hacia abajo.
- 5.3 WHEN se aplica cualquier kernel 3x3 con cualquier modo de borde THE SYSTEM SHALL producir un resultado con error absoluto ≤ 1e-4 respecto a la referencia numpy.
- Verificación: tests en GPU con contexto standalone.

## R6. Presets (Módulo A)

Historia: como estudiante, quiero cambiar entre kernels clásicos con una tecla para comparar sus efectos.

- 6.1 WHEN el usuario pulsa una tecla del 1 al 8 THE SYSTEM SHALL seleccionar el preset correspondiente (identidad, box, gaussiano, sharpen, Sobel Gx, Sobel Gy, Laplaciano, emboss), entrar en modo Kernel y mostrar el resultado en el siguiente frame.
- 6.2 THE SYSTEM SHALL enviar los 9 pesos como `uniform float u_kernel[9]`.
- 6.3 WHEN cambia el kernel THE SYSTEM SHALL imprimir la matriz efectiva (ya normalizada) y su suma, y actualizar el título.

## R7. Edición manual del kernel

Historia: como estudiante, quiero escribir mis propios pesos y ver el resultado sin reiniciar.

- 7.1 THE SYSTEM SHALL leer `custom_kernel` (3×3 números) de `config/settings.json`.
- 7.2 WHEN se pulsa 0 THE SYSTEM SHALL seleccionar el kernel personalizado.
- 7.3 WHEN se pulsa R, o WHILE está activo el kernel personalizado y cambia el mtime del archivo, THE SYSTEM SHALL recargar y aplicar la configuración en ≤ 1 s.
- 7.4 IF el JSON es inválido o el kernel no tiene 9 valores finitos THEN THE SYSTEM SHALL conservar la configuración anterior y mostrar el error (con línea y columna si es de sintaxis), sin cerrarse.

## R8. Normalización

Historia: como estudiante, quiero activar o desactivar la normalización para ver por qué un box sin normalizar satura la imagen.

- 8.1 WHILE la normalización está activa, IF |suma| > 1e-6 THEN THE SYSTEM SHALL dividir cada peso por la suma antes de enviarlo.
- 8.2 WHILE la normalización está activa, IF |suma| ≤ 1e-6 THEN THE SYSTEM SHALL enviar los pesos sin cambios e indicar "suma 0: sin normalizar".
- 8.3 WHEN se pulsa N THE SYSTEM SHALL alternar la normalización (activa por defecto). Los presets se guardan sin normalizar (por ejemplo, el box son nueve unos).

## R9. Visualización de valores negativos

Historia: como estudiante, quiero ver correctamente los resultados negativos de Sobel y Laplaciano.

- 9.1 THE SYSTEM SHALL guardar los resultados en texturas RGBA32F, sin recortar.
- 9.2 THE SYSTEM SHALL ofrecer tres mapeos de visualización: recorte a [0, 1], |v| y offset 0.5 + 0.5·v.
- 9.3 WHEN se selecciona un preset THE SYSTEM SHALL aplicar su mapeo por defecto: |v| para Sobel y Laplaciano, recorte para el resto.
- 9.4 WHEN se pulsa V THE SYSTEM SHALL rotar el mapeo y mostrarlo en el título.

## R10. Modos de borde (Módulo B)

Historia: como estudiante, quiero comparar los modos de borde con el mismo filtro.

- 10.1 THE SYSTEM SHALL ofrecer zero, replicate, reflect (sin repetir el borde) y wrap.
- 10.2 WHEN se pulsa B THE SYSTEM SHALL rotar entre zero → replicate → reflect → wrap y volver a renderizar.
- 10.3 THE SYSTEM SHALL implementar los modos en una única función GLSL compartida que remapea las coordenadas enteras antes de `texelFetch`, y usarla en todo shader que lea vecinos.
- 10.4 THE SYSTEM SHALL remapear correctamente desplazamientos mayores que el tamaño de la imagen, con radios de hasta 30.
- 10.5 THE SYSTEM SHALL coincidir con `numpy.pad` (constant/edge/reflect/wrap) con error ≤ 1e-6.

## R11. Desenfoque variable (box frente a gaussiano)

Historia: como estudiante, quiero comparar box blur y gaussiano con distintos σ.

- 11.1 WHEN se pulsa M THE SYSTEM SHALL rotar entre los modos Kernel → Desenfoque → Unsharp.
- 11.2 WHILE el modo usa desenfoque (Desenfoque, Unsharp o Canny), WHEN se pulsa Arriba o Abajo THE SYSTEM SHALL ajustar σ en [0.5, 10] con paso 0.5, con r = ceil(3σ).
- 11.3 WHEN se pulsa X THE SYSTEM SHALL alternar entre perfil gaussiano y box de varianza equivalente (r = round(√3·σ)).
- 11.4 THE SYSTEM SHALL calcular los pesos 1D en la CPU, normalizados a suma 1, y enviarlos como `uniform float u_weights[61]`.

## R12. Pipeline Canny simplificado (Módulo C)

Historia: como estudiante, quiero recorrer cada etapa de un detector de bordes encadenado para entender qué aporta cada una.

- 12.1 WHEN se pulsa C THE SYSTEM SHALL activar o desactivar el pipeline. Al desactivarlo, vuelve al modo anterior.
- 12.2 WHILE el pipeline está activo THE SYSTEM SHALL ejecutar las pasadas gris → blur H → blur V → Sobel → NMS → umbral, cada una sobre un render target RGBA32F con esquema ping-pong.
- 12.3 THE SYSTEM SHALL NOT leer y escribir la misma textura en una pasada; una aserción falla si ocurre.
- 12.4 WHEN se pulsa Izquierda o Derecha THE SYSTEM SHALL cambiar la etapa visible: Original, Gris, Blur H, Blur, Gx, Gy, Magnitud, Dirección, NMS, Clasificación o Bordes. El título muestra "Canny k/11: nombre".
- 12.5 THE SYSTEM SHALL ejecutar solo las pasadas necesarias hasta la etapa visible.
- 12.6 THE SYSTEM SHALL calcular el gris como Y = 0.299R + 0.587G + 0.114B.
- 12.7 THE SYSTEM SHALL aplicar un gaussiano separable con `u_direction`, σ = 1.4 por defecto y el modo de borde activo.
- 12.8 THE SYSTEM SHALL producir en una sola textura (Gx, Gy, |G|, θ = atan(Gy, Gx)).
- 12.9 THE SYSTEM SHALL cuantizar θ a 0°, 45°, 90° o 135°, y conservar |G| solo si es máximo frente a sus 2 vecinos en la dirección del gradiente.
- 12.10 THE SYSTEM SHALL marcar como borde los píxeles con NMS ≥ alto, y los débiles (bajo ≤ NMS < alto) que tengan al menos un vecino fuerte entre los 8. Se guarda además la clasificación 0 / 0.5 / 1.
- 12.11 WHEN se pulsa RePág o AvPág en modo Canny THE SYSTEM SHALL ajustar el umbral alto en ±0.05. El umbral bajo es ratio·alto, con el ratio leído de `settings.json`.

## R13. Unsharp masking

Historia: como estudiante, quiero comparar el sharpen 3x3 con unsharp masking.

- 13.1 WHILE el modo es Unsharp THE SYSTEM SHALL calcular `orig + α·(orig − blur_σ(orig))` en `unsharp.frag`.
- 13.2 WHEN se pulsa RePág o AvPág THE SYSTEM SHALL ajustar α en [0, 5] con paso 0.25 (1.0 por defecto).

## R14. Gaussiano 2D frente a separable y benchmark (Módulo D)

Historia: como estudiante, quiero medir en la GPU cuánto se gana con la separabilidad para incluirlo en el informe.

- 14.1 WHEN se pulsa S THE SYSTEM SHALL alternar entre gaussiano 2D directo ((2r+1)² lecturas) y separable (2·(2r+1) lecturas) en todos los modos que desenfocan.
- 14.2 THE SYSTEM SHALL producir resultados 2D y separable que difieran ≤ 1e-4 en los 4 modos de borde.
- 14.3 WHEN se pulsa P o se ejecuta con `--benchmark` THE SYSTEM SHALL medir con `ctx.query(time=True)` solo las pasadas de desenfoque, para σ ∈ {1, 2, 3, 5, 8, 10}, con 5 ejecuciones de calentamiento y 50 repeticiones.
- 14.4 THE SYSTEM SHALL reportar media, mediana, desviación y mínimo en ms, lecturas por píxel, resolución y GPU, y guardar `results/benchmark_<fecha>.csv`.
- 14.5 WHILE la app está en marcha THE SYSTEM SHALL mostrar en el título el tiempo de GPU del último procesado.
- 14.6 IF las consultas de tiempo fallan THEN THE SYSTEM SHALL avisar y continuar sin tiempos.
- 14.7 WHEN se ejecuta con `--benchmark` THE SYSTEM SHALL usar una ventana oculta (o contexto standalone) y terminar con código 0.

## R15. Capturas

Historia: como estudiante, quiero guardar el resultado visible para el informe.

- 15.1 WHEN se pulsa G THE SYSTEM SHALL guardar un PNG W×H del resultado visible, con el mapeo aplicado, en `results/AAAAMMDD-HHMMSS_<modo>_<kernel|etapa>_<borde>_<params>.png`.
- 15.2 THE SYSTEM SHALL crear `results/` si no existe e imprimir la ruta del archivo.
- 15.3 THE SYSTEM SHALL guardar la captura con la misma orientación que se ve en pantalla.
- 15.4 IF falla la escritura THEN THE SYSTEM SHALL mostrar el error y continuar.

## R16. Controles generales

Historia: como usuario, quiero controlar todo con el teclado y saber siempre en qué estado está la app.

- 16.1 WHEN se pulsa Esc THE SYSTEM SHALL liberar los recursos de GL y terminar con código 0.
- 16.2 WHEN se pulsa H THE SYSTEM SHALL imprimir la tabla de controles.
- 16.3 WHEN cambia el estado THE SYSTEM SHALL actualizar el título con modo, kernel o etapa, borde, normalización, mapeo, σ, α, umbrales, 2D/separable y tiempo de GPU.
- 16.4 THE SYSTEM SHALL renderizar solo cuando cambie el estado o la ventana pida redibujarse.
- 16.5 THE SYSTEM SHALL ignorar las teclas sin asignar.

## R17. Experimentos y documentación

Historia: como estudiante, quiero reproducir todos los experimentos del informe con un comando.

- 17.1 WHEN se ejecuta con `--experiments` THE SYSTEM SHALL generar en `results/experimentos_<fecha>/` las capturas de los 5 experimentos y un `manifest.csv` con los parámetros de cada una.
- 17.2 THE SYSTEM SHALL incluir un README con instalación en Windows 11, ejecución, controles, reproducción de cada experimento y limitaciones.
- 17.3 WHEN se ejecuta `python -m pytest` THE SYSTEM SHALL pasar todos los tests. Los tests de GPU se saltan con un motivo explícito si no hay contexto.

## Fuera de alcance

- Escenas 3D, modelos, iluminación, cámara o cualquier transformación geométrica.
- Interfaz gráfica compleja (basta teclado, consola y título de la ventana).
- Histéresis completa iterativa del Canny original.
