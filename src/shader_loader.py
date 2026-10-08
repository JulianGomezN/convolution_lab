"""Carga de shaders GLSL con soporte de #include y errores legibles.

`#include "archivo.glsl"` se sustituye por el contenido del archivo, rodeado de
directivas `#line línea fuente`. Cada archivo recibe su propio número de fuente
(0 = archivo principal), así el log del compilador se puede traducir a
`archivo:línea` del archivo original.
"""

from __future__ import annotations

import re
from pathlib import Path

import moderngl

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"
VERTEX_SHADER = "fullscreen.vert"

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_VERSION_RE = re.compile(r"^\s*#\s*version\b")
# Prefijo "fuente:línea" (AMD, Intel, Mesa) o "fuente(línea)" (NVIDIA) del log del compilador.
_LOG_LOCATION_RE = re.compile(r"^(\s*(?:ERROR|WARNING|error|warning):\s*)?(\d+)(?::(\d+)|\((\d+)\))")
_STAGE_HEADERS = {"vertex_shader": "vert", "fragment_shader": "frag"}


class ShaderError(Exception):
    """Error al leer, compilar o enlazar un shader (código de salida 4)."""


def preprocess(name: str, shader_dir: Path = SHADER_DIR) -> tuple[str, list[str]]:
    """Resuelve los #include de `name`.

    Devuelve (código fuente, lista de archivos indexada por número de fuente GLSL).
    """
    shader_dir = Path(shader_dir)
    files: list[str] = []
    out: list[str] = []

    def read(fname: str) -> list[str]:
        path = shader_dir / fname
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ShaderError(f"No se pudo leer el shader '{path}': {exc.strerror or exc}") from None

    def emit(fname: str, stack: tuple[str, ...]) -> None:
        if fname in stack:
            raise ShaderError("#include circular: " + " -> ".join(stack + (fname,)))
        src_id = len(files)
        files.append(fname)
        lines = read(fname)
        if stack:
            # La siguiente línea es la 1 del archivo incluido.
            out.append(f"#line 1 {src_id}")
        for lineno, line in enumerate(lines, start=1):
            m = _INCLUDE_RE.match(line)
            if m:
                emit(m.group(1), stack + (fname,))
                # Vuelve a la numeración del archivo actual en la línea siguiente.
                out.append(f"#line {lineno + 1} {src_id}")
            elif stack and _VERSION_RE.match(line):
                raise ShaderError(f"'{fname}' es un archivo incluido y no puede contener #version")
            else:
                out.append(line)

    emit(name, ())
    return "\n".join(out) + "\n", files


def translate_log(log: str, files_by_stage: dict[str, list[str]]) -> str:
    """Sustituye los prefijos `fuente:línea` del log por `archivo:línea`."""
    current = None
    result = []
    for line in log.splitlines():
        stripped = line.strip()
        if stripped in _STAGE_HEADERS:
            current = files_by_stage.get(_STAGE_HEADERS[stripped])
            result.append(line)
            continue
        m = _LOG_LOCATION_RE.match(line)
        if current and m and int(m.group(2)) < len(current):
            prefix = m.group(1) or ""
            lineno = m.group(3) or m.group(4)
            line = f"{prefix}{current[int(m.group(2))]}:{lineno}" + line[m.end():]
        result.append(line)
    return "\n".join(result)


def load_program(
    ctx: moderngl.Context,
    frag_name: str,
    vert_name: str = VERTEX_SHADER,
    shader_dir: Path = SHADER_DIR,
) -> moderngl.Program:
    """Compila y enlaza `vert_name` + `frag_name`. Lanza ShaderError con el log traducido."""
    vert_src, vert_files = preprocess(vert_name, shader_dir)
    frag_src, frag_files = preprocess(frag_name, shader_dir)
    try:
        return ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    except moderngl.Error as exc:
        log = translate_log(str(exc), {"vert": vert_files, "frag": frag_files})
        raise ShaderError(
            f"Error al compilar o enlazar el shader '{frag_name}' (con '{vert_name}'):\n{log.strip()}"
        ) from None
