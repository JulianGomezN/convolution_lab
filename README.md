# Laboratorio interactivo de máscaras de convolución

Aplicación de escritorio para Computación Gráfica: carga una imagen, la sube a la GPU como textura y le aplica máscaras de convolución en fragment shaders de OpenGL 3.3 core (moderngl + GLFW). La ventana tiene exactamente el tamaño de la imagen y solo se dibuja un quad a pantalla completa, sin transformaciones geométricas.

Estado: fase 1 (ventana, textura, quad y shader de identidad). El README completo llega en la tarea 20.

## Instalación (Windows 11, PowerShell)

Requiere Python 3.10 o superior y un driver de GPU con OpenGL 3.3.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Para ejecutar los tests:
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Ejecución

```powershell
# Genera las imágenes sintéticas de prueba en assets/images/
.venv\Scripts\python.exe tools\make_test_images.py

# Abre la imagen indicada (sin --image usa default_image de config/settings.json)
.venv\Scripts\python.exe main.py --image assets\images\orientacion.png
```

Controles disponibles en esta fase: `Esc` para salir.

Códigos de salida: 0 normal, 1 error inesperado, 2 error de imagen, 3 error de OpenGL/GLFW, 4 error de shader, 130 Ctrl+C.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Los tests de GPU usan un contexto OpenGL 3.3 sin ventana y se saltan, indicando el motivo, si no hay contexto disponible.
