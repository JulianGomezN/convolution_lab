# Tareas: Laboratorio interactivo de máscaras de convolución

Requisitos: #[[file:requirements.md]]
Diseño: #[[file:design.md]]

Regla general: al terminar cada tarea, `.venv\Scripts\python.exe -m pytest` debe pasar. Al terminar cada fase debe existir algo funcional y visible.

## Fase 1: Base (ventana, textura, quad, identidad)

- [x] 1. Esqueleto del proyecto y dependencias
  - Estructura de carpetas, `requirements.txt` con versiones exactas verificadas, `requirements-dev.txt` (`-r requirements.txt` + pytest fijado), `pytest.ini` (testpaths = tests), `.gitignore` (.venv, __pycache__, results/* salvo .gitkeep), README mínimo y `config/settings.json`.
  - Pruebas: `tests/test_smoke.py` importa los 4 paquetes.
  - Requisitos: R1.4, R1.5, R17.3.
  - Hecho cuando: `.venv\Scripts\pip install -r requirements-dev.txt` y `.venv\Scripts\python -m pytest` pasan.

- [x] 2. Carga de imagen e imágenes de prueba
  - `texture_loader.load_image(path) → ndarray uint8 H×W×4` que lanza `ImageLoadError` (ruta, motivo, lista de `assets/images`) y `create_texture(ctx, arr)`. `tools/make_test_images.py` genera `orientacion.png` (una "F" asimétrica), escalón, tablero, círculo y ruido en `assets/images/`.
  - Pruebas: archivo inexistente, archivo no válido, conversión L/P/RGB → RGBA, forma correcta.
  - Requisitos: R4.1, R4.3, R4.5.
  - Hecho cuando: el script genera las imágenes y los tests pasan.

- [x] 3. Shader loader, quad, `run_pass` y `RenderTarget`
  - `shader_loader` con `#include` y `#line` y `ShaderError` legible; quad de 4 vértices; `run_pass` con aserción anti-feedback; `RenderTarget` f4 W×H; `common.glsl` mínimo; `convolve3x3.frag` con kernel identidad (o passthrough temporal reemplazado en la tarea 7).
  - Pruebas: en GPU, imagen 7×5 subida, procesada y leída igual a la entrada; la aserción salta si el destino es también entrada; `test_constraints.py` (sin `mat[234]`, `.vert` sin uniforms, sin `texture(`, 4 vértices en ±1, `version_code >= 330`).
  - Requisitos: R2.1–R2.4, R3.4, R5.1.
  - Hecho cuando: `pytest` pasa, incluidos los tests de GPU.

- [x] 4. Ventana GLFW y `display.frag`
  - `app.py` con hints 3.3 core forward-compatible, ventana no redimensionable W×H, comprobación del framebuffer, aviso si supera el monitor, impresión de info GL, `display.frag` con `u_flip_y`, Esc, callback de refresco y códigos de salida 2 y 3. `main.py` con argparse (`--image`).
  - Pruebas: código 2 con ruta falsa (subprocess o llamando a `main`); orientación verificada manualmente.
  - Requisitos: R1.1–R1.3, R3.1–R3.3, R4.2–R4.4, R16.1.
  - Hecho cuando: `python main.py --image assets/images/orientacion.png` abre una ventana del tamaño exacto con la "F" bien orientada; con ruta falsa, mensaje y código 2.

### Registro de verificación de la fase 1

Entorno: Windows 11, Python 3.13.12 (`.venv`), GPU AMD Radeon(TM) 680M, `GL_VERSION 3.3.0 Core Profile Context 24.10.20.240607`, `GL_VENDOR ATI Technologies Inc.`

- Pins: moderngl 5.13.0, glfw 2.10.2, pillow 12.3.0, numpy 2.2.6, pytest 9.1.1. numpy se baja de la 2.5.3 sugerida a la 2.2.6 porque la 2.5.3 declara `Requires-Python >=3.12` y rompería "Python 3.10+" (R1.4); la 2.2.6 es la última con wheels cp310–cp313 win_amd64. El runtime solo usa `asarray`, `zeros`, `empty`, `frombuffer` y `default_rng`, estables entre ambas versiones.
- `pip install --only-binary=:all: -r requirements-dev.txt` en el `.venv` (cp313): instala numpy 2.2.6, código 0 (`.agents/tmp/pip_numpy.txt`).
- `pip install --dry-run --ignore-installed --only-binary=:all: --python-version {3.10,3.11} --platform win_amd64 --implementation cp --target <tmp> -r requirements-dev.txt`: ambas resuelven solo con wheels (`moderngl-5.13.0-cp310`, `pillow-12.3.0-cp310`, `numpy-2.2.6-cp310`, ídem cp311), EXIT=0 (`.agents/tmp/pip_py310.txt`).
- `tests/test_smoke.py::test_pin_supports_min_python` comprueba, para cada pin de `requirements.txt` y `requirements-dev.txt`, que la versión instalada es la fijada y que su `Requires-Python` admite 3.10.
- `python tools\make_test_images.py`: genera orientacion.png (640x480), escalon.png, tablero.png, circulo.png y ruido.png en `assets/images/`.
- `moderngl.create_standalone_context(require=330)`: funciona (`version_code 330`, GL_MAX_TEXTURE_SIZE 16384). Los tests de GPU se ejecutan, no se saltan; la cabecera de `pytest -v` muestra `OpenGL (standalone) ... GL_RENDERER: AMD Radeon(TM) 680M`.
- `python -m pytest -q`: 70 passed, EXIT=0 (`.agents/tmp/pytest.txt`). `pytest -v -rs`: 70 passed, 0 skipped, cabecera con GL_RENDERER AMD Radeon(TM) 680M (`.agents/tmp/pytest_v2.u8.txt`). Incluye los tests de código 3 para fallo de `moderngl.create_context` y para `version_code < 330` (320), con GLFW simulado.
- `python main.py --image assets/images/no_existe.png`: muestra ruta, motivo ("el archivo no existe") y las 5 imágenes de `assets/images/`; EXIT=2.
- `python main.py --image assets/images/orientacion.png`, lanzado en segundo plano (`.agents/tmp/demo.ps1`): la consola imprime GL_VERSION, GL_RENDERER, GL_VENDOR e `imagen 640x480 | framebuffer 640x480`. Win32 informa área cliente 640x480, sin WS_THICKFRAME ni WS_MAXIMIZEBOX, y título `Laboratorio de convolución | orientacion.png 640x480 | Kernel: Identidad`. La captura de pantalla de la ventana (`.agents/tmp/demo_screen.png`) tiene rojo arriba a la izquierda, verde abajo a la derecha y la barra superior de la "F" en (400, 100). Al enviar Esc (WM_KEYDOWN) el proceso termina con código 0. Repetido con numpy 2.2.6: mismo resultado (`.agents/tmp/demo_report.txt`, `.agents/tmp/demo_stdout.txt`).
- Queda pendiente la verificación manual: abrir la ventana y comprobar a ojo que la "F" se lee bien (sin espejo ni inversión).

Notas de implementación:
- `common.glsl` es mínimo: `fetch()` devuelve 0 fuera de rango y `u_border` está declarado sin usar. La tarea 6 añade los 4 modos.
- `convolve3x3.frag` ya es la correlación 3x3 real con `u_kernel[9]`; la fase 1 la usa con el kernel identidad. La tarea 7 añade el Processor y los tests contra numpy.
- `display.frag` solo tiene `u_flip_y`; la tarea 8 añade `u_map`/`u_gain` y la 11 `u_channel`.
- `run_pass(quad, prog, target, textures, uniforms)` recibe el `Quad` como primer argumento, porque el VAO depende del programa y el VBO se comparte.
- La traducción de argparse en `main.py` parchea `argparse._`, un nombre privado. Si Python lo cambia, la ayuda vuelve al inglés sin error; `test_help_is_spanish` lo detecta. El riesgo está documentado en un comentario.
- moderngl también busca `#include "..."` en el código, comentarios incluidos, y falla si encuentra uno sin resolver. Por eso ningún shader debe escribir la directiva literal en comentarios; un test lo comprueba.

## Fase 2: Módulos A y B

- [x] 5. `kernels.py`
  - 8 presets sin normalizar, con nombre y vista por defecto; `normalize(k)` con umbral 1e-6 (devuelve pesos y flag "sin normalizar").
  - Pruebas: sumas de la tabla, normalización de box y gauss, caso suma 0.
  - Requisitos: R6, R8.1, R8.2, R9.3.
  - Hecho cuando: tests en verde.

- [x] 6. `common.glsl` con los modos de borde
  - `imod`, `reflect_idx` y `fetch` según el diseño.
  - Pruebas: en GPU, shader de prueba con desplazamiento uniforme `u_offset` (incluidos ≥ W, por ejemplo ±13 en 7×5) comparado con `np.pad` en los 4 modos, en imágenes 1×1, 2×3 y 7×5.
  - Requisitos: R10.1, R10.3–R10.5.
  - Hecho cuando: tests en verde.

- [x] 7. `convolve3x3.frag` y modo Kernel en el `Processor`
  - Shader real, `Processor.run` para el modo Kernel (con normalización según el estado), `reference.correlate3x3` en numpy, app conectada al Processor.
  - Pruebas: 9 kernels (8 presets + uno aleatorio) × 4 bordes, error ≤ 1e-4, alfa = 1.
  - Requisitos: R5.2, R5.3, R6.2, R9.1.
  - Hecho cuando: tests en verde y la ventana muestra el preset fijado.

- [x] 8. Estado, teclado, título y mapeos de visualización
  - `state.py` con `handle_key`, key callback en `app.py`, render bajo demanda, mapeos recorte/abs/offset, título y ayuda con H.
  - Pruebas: transiciones de 1–8, B, N, V y teclas ignoradas (CPU).
  - Requisitos: R6.1, R6.3, R8.3, R9.2–R9.4, R10.2, R16.2–R16.5.
  - Hecho cuando: las teclas 1–8, B, N y V cambian el resultado en vivo; box con N desactivada satura.

- [x] 9. Kernel personalizado (`config.py` y `settings.json`)
  - Carga y validación (`ConfigError` con línea/columna), tecla 0, tecla R y vigilancia del mtime.
  - Pruebas: JSON válido, sintaxis rota (línea y columna), 8 valores, NaN/infinito, conservación de la configuración previa.
  - Requisitos: R7.1–R7.4.
  - Hecho cuando: editar y guardar `settings.json` con la app abierta se refleja en < 1 s; un JSON roto da mensaje y la app sigue abierta.

### Registro de verificación de la fase 2

- `python -m pytest -q`: 246 passed, EXIT=0 (`.agents/tmp/pytest.txt`). `pytest -v -rs`: 0 skipped, cabecera `OpenGL (standalone) ... GL_RENDERER: AMD Radeon(TM) 680M` (`.agents/tmp/pytest_v.u8.txt`).
- Tests nuevos: `test_kernels.py` (tabla, normalización, umbral 1e-6), `test_borders.py` (shader de prueba con `u_offset` hasta ±30 en 1×1, 2×3 y 7×5, más múltiplos exactos del tamaño y del periodo, frente a `np.pad` en los 4 modos), `test_convolve.py` (9 kernels × 4 bordes × normalización on/off, error ≤ 1e-4, alfa = 1; box sin normalizar da 4.5 sobre gris 0.5), `test_state.py` (transiciones de 1–8, 0, B, N, V, R, H, Esc y teclas ignoradas; título; informe del kernel; ningún módulo del núcleo importa glfw), `test_config.py` (validación y recarga con la `App` y GLFW simulado), `test_render_pass.py::test_display_maps` y dos tests nuevos en `test_constraints.py` (los `.frag` de proceso solo leen vecinos con `fetch` de `common.glsl`).
- Demo en vivo (`.agents/tmp/demo_f2.ps1`, informe en `.agents/tmp/demo_f2_report.txt`): con la app abierta se enviaron 2, N, B, V, 5, 7, H y 0 como WM_KEYDOWN, y el título cambió en cada una. La captura de box con N desactivada (`.agents/tmp/demo_f2_box_sin_norm.png`) sale blanca: el fondo (32, 32, 48) × 9 satura. Al reescribir `settings.json` con el kernel 0 activo, se recargó en 151 ms. Con un JSON roto, stderr muestra "línea 1, columna 33" y la app sigue abierta. Esc → código 0. `settings.json` se restauró al terminar.
- Pendiente de verificación manual del usuario: recorrer a ojo los filtros y bordes.

Notas de implementación:
- `imod` corrige un posible ±1 tras `floor(float(a)/float(n))`: la división en float de la GPU no tiene por qué estar correctamente redondeada (GLSL admite 2.5 ULP), así que en un múltiplo exacto (14/7) podría devolver n. `test_exact_multiples_of_size` lo cubre.
- `display.frag` añade `u_map` y `u_gain`. Si no se fijan, GL los deja a 0, y con `u_gain = 0` la salida sería negra; por eso `test_display_flip` pasa ahora `u_gain = 1`.
- `custom_kernel` admite una matriz 3x3 o una lista plana de 9 números. Se rechazan booleanos, NaN, ±Infinity y 1e999. Se lee con `utf-8-sig` porque el Bloc de notas puede guardar con BOM.
- Si `settings.json` es inválido al arrancar, se avisa y el kernel personalizado pasa a ser la identidad hasta que se corrija (R para recargar). Durante la ejecución se conserva el anterior.
- La vigilancia compara (mtime_ns, tamaño) en cada vuelta del bucle (timeout de 0.25 s), solo con el kernel 0 activo. Al pulsar 0 se comprueba primero si el archivo cambió mientras estaba inactivo. Un JSON roto da un único mensaje hasta que el archivo vuelve a cambiar.
- `app.KEY_MAP` traduce la fila de números y el teclado numérico, más B, N, V, R, H y Esc. Solo se procesa PRESS; el REPEAT para flechas y RePág llega en fases posteriores.

## Fase 3: Módulo C

- [x] 10. Desenfoque separable variable
  - `gaussian_1d(sigma)` y `box_1d(sigma)` en `kernels.py`, `blur_1d.frag`, modo Desenfoque (teclas M, X, Arriba/Abajo).
  - Pruebas: pesos suman 1, r = ceil(3σ), r box = round(√3·σ), `MAX_TAPS` coincide con los shaders (parseando los `.frag`), GPU frente a referencia numpy separable en 4 bordes.
  - Requisitos: R11.1–R11.4, R12.7 (parcial).
  - Hecho cuando: M lleva a Desenfoque, Arriba/Abajo cambia σ y X alterna a box.

- [x] 11. `PingPong`, gris y armazón de etapas de Canny
  - `framebuffers.PingPong`, `grayscale.frag`, tabla de etapas, teclas C e Izquierda/Derecha, selección de canal en `display.frag`, ejecución hasta la etapa k. Etapas 1–4 operativas.
  - Pruebas: gris coincide con numpy; `swap` nunca deja src = dst; C restaura el modo previo; límites de etapa 1..11.
  - Requisitos: R12.1–R12.6.
  - Hecho cuando: C activa Canny y las flechas muestran Original, Gris, Blur H y Blur.

- [x] 12. `sobel.frag`, etapas 5–8
  - Salida (Gx, Gy, |G|, θ) y vista de dirección en hsv.
  - Pruebas: 4 canales frente a numpy; signo correcto en escalones horizontal y vertical (oscuro→claro hacia la derecha da Gx > 0).
  - Requisitos: R12.8.
  - Hecho cuando: se ven Gx, Gy, Magnitud y Dirección en color.

- [x] 13. `nms.frag`, etapa 9
  - Cuantización y supresión según el diseño.
  - Pruebas: escalón vertical suavizado → línea de 1 px; escalón diagonal → 1 px en diagonal; referencia numpy de NMS.
  - Requisitos: R12.9.
  - Hecho cuando: la etapa NMS muestra bordes finos.

- [x] 14. `threshold.frag`, etapas 10–11
  - Doble umbral con comprobación de vecinos, umbrales desde `settings.json`, RePág/AvPág para el umbral alto.
  - Pruebas: casos sintéticos (fuerte, débil con vecino fuerte, débil aislado, bajo umbral).
  - Requisitos: R12.10, R12.11.
  - Hecho cuando: pipeline completo; al cambiar el umbral alto los bordes cambian en vivo.

### Registro de verificación de la fase 3

- `python -m pytest -q`: 454 passed, EXIT=0 (`.agents/tmp/pytest.txt`), tras el cambio del umbral por defecto. `pytest -rs tests/test_canny.py tests/test_blur.py`: 137 passed, 0 skipped, cabecera `OpenGL (standalone) ... GL_RENDERER: AMD Radeon(TM) 680M` (`.agents/tmp/pytest_v.txt`).
- Tests nuevos:
  - `test_blur.py`: pesos gauss y box para σ = 0.5…10 (suma 1, r = ceil(3σ), r box = round(√3·σ), simetría, frente a la fórmula), σ fuera de rango, `MAX_TAPS` = 61 leído de los `.frag` y `array_length` en GPU. El modo Desenfoque se compara con numpy separable en 2 perfiles × 4 σ (incluido σ = 10, r = 30 sobre 7×5) × 4 bordes, con error ≤ 1e-4.
  - `test_canny.py`:
    - `PingPong` (`swap` nunca deja src = dst, rechaza el mismo target).
    - Pasadas ejecutadas por etapa 1..11 (R12.5) y vista de cada etapa.
    - Gris frente a numpy con entrada f4 y RGBA8. Blur H y Blur en 4 bordes × 3 σ. Canny usa siempre el gaussiano.
    - Sobel: 4 canales frente a numpy en 4 bordes. Signo en escalones horizontal y vertical, con |G| = 4 en el escalón ideal sin suavizar. θ = 0 en zonas planas.
    - NMS frente a numpy en 4 bordes. El escalón vertical suavizado queda en 1 px por fila. El escalón a 45° da una línea fina, sin bloques 2x2.
    - Doble umbral con casos sintéticos, incluidas las igualdades exactas con alto y bajo. Los vecinos respetan el modo de borde (zero frente a wrap). Etapas 10/11 frente a numpy en 4 bordes. Más umbral da menos bordes.
    - `display.frag`: canales r/g/b y vista de dirección hsv frente a `colorsys`.
  - `test_state.py`: M, X, Arriba/Abajo con sus límites, C restaura el modo previo, Izquierda/Derecha dentro de 1..11, RePág/AvPág solo en Canny, título por modo.
  - `test_config.py`: claves numéricas, recarga solo de las claves cambiadas, REPEAT solo para flechas y RePág/AvPág.
- Demo en vivo (`.agents/tmp/demo_f3.ps1`, informe en `.agents/tmp/demo_f3_report.txt`), con WM_KEYDOWN reales sobre `orientacion.png`:
  - M → "Desenfoque: gaussiano σ=2 r=6". Arriba → σ=2.5 r=8 y σ=3 r=9. X → "box σ=2.5 r=4".
  - C → "Canny k/11". Izquierda/Derecha recorren las 11 etapas y se detienen en 1 y en 11.
  - RePág/AvPág cambian "alto=… bajo=…" (bajo = 0.4·alto). En Canny, Arriba pasa a σ=1.9 r=6.
  - C vuelve a Desenfoque, M vuelve a Kernel y Esc sale con código 0.
  - Las capturas con la ventana en primer plano muestran la "F" en gris, en Blur H y en Blur. Gx y Gy aparecen como líneas de bordes verticales y horizontales; la Dirección sale en color (cian a la izquierda, rojo a la derecha, morado arriba y verde abajo) y NMS como contornos de 1 px (`.agents/tmp/demo_f3/`). Con alto = 0.30 la etapa Bordes tiene 1456 píxeles blancos; con alto = 0.60, ninguno. Después del cambio del valor por defecto a 0.3, se repitió la demo solo con C y Derecha, sin tocar umbrales ni σ (`DEMO_DEFAULTS=1`, informe en `.agents/tmp/demo_f3b_report.txt`). El título muestra "σ=1.4 r=5 | alto=0.30 bajo=0.12" y la captura de la etapa 11 (`.agents/tmp/demo_f3/defecto_canny_11.png`, analizada con `.agents/tmp/check_default_f3.py`) tiene 1456 píxeles de borde. La barra superior de la F tiene borde en las 220 columnas con grosor de 1 px, y el lado izquierdo en las 300 filas.
- Pendiente de verificación manual del usuario: recorrer a ojo las etapas.

Notas de implementación:
- `threshold.frag` usa `u_mag_gain` en lugar de `u_mag_scale` (diseño D3). `test_constraints.py` prohíbe "scale" en nombres de uniforms para detectar transformaciones, y el nombre no se ve fuera del código. `design.md` está actualizado.
- Al activar Canny se empieza en la etapa 1 (Original), y la etapa se conserva al salir con C y volver.
- M desde Canny avanza a partir del modo previo. Las teclas 1–8 y 0 salen de Canny al modo Kernel (R6.1). El ciclo de M es Kernel → Desenfoque; la tarea 18 añade Unsharp.
- En el modo Desenfoque la vista es siempre recorte, y V solo afecta al modo Kernel. Canny usa siempre el perfil gaussiano (R12.7); X solo cambia el perfil del modo Desenfoque.
- Límites: σ en [0.5, 10] con paso 0.5 (también canny_sigma, aunque empiece en 1.4) y alto en [0.05, 1.0] con paso 0.05. `settings.json` valida las mismas claves y rangos (sigma, canny_sigma, canny_high, canny_low_ratio ∈ [0, 1]). Son opcionales y tienen valores por defecto.
- Recarga (R o mtime): solo se aplican las claves que cambiaron en el archivo respecto a la última carga. Así, editar `custom_kernel` no deshace un σ o un umbral ajustado con el teclado.
- `sobel.frag` devuelve θ = 0 si Gx = Gy = 0, porque `atan(0, 0)` no está definido en GLSL.
- NMS a 45°: con vecinos discretos, la diagonal k = x − y se compara con k ± 2, así que sobreviven el máximo y uno de sus vecinos k ± 1. Queda una línea en escalera de conectividad 4, nunca un bloque 2x2. Es lo esperable en un Canny de vecinos discretos.
- `Processor.last_passes` lista las pasadas del último `run()`. Lo usan los tests de R12.5.
- Umbral alto por defecto = 0.3 (decisión del usuario, opción a). Tras el gaussiano, un escalón negro → blanco ideal no llega a |G|/4 = 1: da 0.89 con σ = 0.5, 0.64 con σ = 1, 0.51 con σ = 1.4 y 0.38 con σ = 2. En `orientacion.png` (contraste 0.81), el máximo con σ = 1.4 es 0.41, así que con alto = 0.6 la etapa Bordes salía negra. Se cambió el valor por defecto a 0.3 (bajo = 0.12) en `settings.json`, `AppState` y `config.py`. La fórmula de D3 no cambia: la ganancia sigue fija en 0.25. S7 y D3 están actualizados con la tabla de atenuación. `test_canny.py::test_default_values_show_edges_of_default_image` comprueba, con los valores de `settings.json`, que la etapa 11 dibuja el contorno continuo y fino de la barra superior de la F, y que con 0.6 no saldría nada. `test_config.py::test_defaults_agree_between_state_config_and_settings` comprueba que los tres valores por defecto coinciden.
- Pendiente para la tarea 20: el README debe explicar como limitación conocida que el umbral efectivo depende de σ (al subir σ conviene bajar el umbral con AvPág).

## Fase 4: Módulo D, experimentos y capturas

- [ ] 15. `blur_2d.frag` y tecla S
  - Gaussiano 2D directo integrado en Desenfoque y Canny (y en Unsharp en la tarea 18).
  - Pruebas: 2D y separable difieren ≤ 1e-4 en 4 bordes, también con radio ≥ tamaño de imagen.
  - Requisitos: R14.1, R14.2.
  - Hecho cuando: S cambia el método sin diferencia visible y el título indica 2D o separable.

- [ ] 16. `benchmark.py`
  - Consultas de tiempo, estadísticos, CSV, tecla P (luego `dirty`), `--benchmark` con ventana oculta o contexto standalone, tiempo de GPU en el título.
  - Pruebas: contexto standalone con 2 repeticiones y 1 σ, tiempos ≥ 0, columnas del CSV; fallo de query simulado (monkeypatch) no rompe.
  - Requisitos: R14.3–R14.7.
  - Hecho cuando: `python main.py --benchmark` imprime la tabla y crea el CSV en `results/`.

- [ ] 17. Capturas con G
  - `capture.py` dibuja `display.frag` en un FBO RGBA8 con `u_flip_y = 0`, lee, guarda el PNG con nombre descriptivo y crea `results/` si falta.
  - Pruebas: PNG W×H, orientación correcta (comparado con la entrada en modo identidad), nombre correcto, error de escritura manejado.
  - Requisitos: R15.1–R15.4, R3.4.
  - Hecho cuando: G guarda una imagen idéntica a la ventana.

- [ ] 18. Modo Unsharp
  - `unsharp.frag`, modo en el ciclo de M, α con RePág/AvPág, compatible con S.
  - Pruebas: coincide con la fórmula en numpy.
  - Requisitos: R13.1, R13.2.
  - Hecho cuando: se compara en vivo sharpen (tecla 4) con unsharp a distintos α.

- [ ] 19. Experimentos por lotes (`--experiments`)
  - `experiments.py` genera E1–E5 y `manifest.csv` reutilizando Processor, captura y benchmark (`--quick` reduce repeticiones).
  - Pruebas: número de archivos esperado y manifest coherente (GPU, imagen pequeña, benchmark reducido).
  - Requisitos: R17.1.
  - Hecho cuando: `python main.py --experiments --image assets/images/<foto>.png` llena `results/experimentos_<fecha>/`.

- [ ] 20. README final y trazabilidad
  - README con instalación en Windows 11, ejecución, controles, reproducción de experimentos, notas del benchmark y limitaciones (EXIF, Canny sin histéresis iterativa, escalado DPI, imágenes mayores que el monitor, umbral de Canny dependiente de σ por la atenuación del gaussiano; D3). Matriz requisitos ↔ tests al final de este archivo.
  - Pruebas: `python -m pytest` completo en verde; recorrido de controles.
  - Requisitos: R17.2, R17.3.
  - Hecho cuando: alguien sin contexto sigue el README y reproduce los experimentos.

## Matriz de trazabilidad

(Se completa en la tarea 20.)
