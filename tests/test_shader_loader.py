"""Preprocesado de #include, #line y errores legibles de shaders."""

import re

import pytest

from src.shader_loader import ShaderError, load_program, preprocess, translate_log


def _write(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")


def test_include_inserts_line_directives(tmp_path):
    _write(
        tmp_path,
        {
            "lib.glsl": "float a() { return 1.0; }\nfloat b() { return 2.0; }\n",
            "main.frag": '#version 330 core\n#include "lib.glsl"\nout vec4 o;\nvoid main() { o = vec4(a()); }\n',
        },
    )
    src, files = preprocess("main.frag", tmp_path)
    assert files == ["main.frag", "lib.glsl"]
    lines = src.splitlines()
    assert lines[0] == "#version 330 core"
    assert lines[1] == "#line 1 1"
    assert lines[2].startswith("float a()")
    assert lines[4] == "#line 3 0"
    assert lines[5] == "out vec4 o;"


def test_real_shaders_preprocess():
    src, files = preprocess("convolve3x3.frag")
    assert files == ["convolve3x3.frag", "common.glsl"]
    assert "vec4 fetch(" in src
    # moderngl busca este patrón incluso en comentarios y falla si queda alguno.
    assert not re.search(r'#include\s+"', src)


def test_missing_file(tmp_path):
    with pytest.raises(ShaderError, match="No se pudo leer"):
        preprocess("no_existe.frag", tmp_path)


def test_circular_include(tmp_path):
    _write(tmp_path, {"a.glsl": '#include "b.glsl"\n', "b.glsl": '#include "a.glsl"\n',
                      "m.frag": '#version 330 core\n#include "a.glsl"\n'})
    with pytest.raises(ShaderError, match="circular"):
        preprocess("m.frag", tmp_path)


def test_included_file_cannot_have_version(tmp_path):
    _write(tmp_path, {"x.glsl": "#version 330 core\n", "m.frag": '#version 330 core\n#include "x.glsl"\n'})
    with pytest.raises(ShaderError, match="#version"):
        preprocess("m.frag", tmp_path)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ERROR: 1:3: 'x' : undeclared identifier", "ERROR: lib.glsl:3: 'x' : undeclared identifier"),
        ("0:12(5): error: syntax error", "main.frag:12(5): error: syntax error"),
        ("0(7) : error C0000: syntax error", "main.frag:7 : error C0000: syntax error"),
        ("ERROR: 2 compilation errors.  No code generated.", "ERROR: 2 compilation errors.  No code generated."),
    ],
)
def test_translate_log_formats(raw, expected):
    log = f"GLSL Compiler failed\n\nfragment_shader\n===============\n{raw}\n"
    out = translate_log(log, {"frag": ["main.frag", "lib.glsl"], "vert": ["fullscreen.vert"]})
    assert expected in out


def test_compile_error_points_to_included_file(ctx, tmp_path):
    _write(
        tmp_path,
        {
            "fullscreen.vert": "#version 330 core\nin vec2 in_pos;\nvoid main() { gl_Position = vec4(in_pos, 0.0, 1.0); }\n",
            "bad.glsl": "// comentario\n// otro\nfloat f() { return undefined_symbol; }\n",
            "main.frag": '#version 330 core\n#include "bad.glsl"\nout vec4 o;\nvoid main() { o = vec4(f()); }\n',
        },
    )
    with pytest.raises(ShaderError) as info:
        load_program(ctx, "main.frag", shader_dir=tmp_path)
    msg = str(info.value)
    assert "main.frag" in msg
    assert "bad.glsl:3" in msg, msg
