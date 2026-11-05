import math

import numpy as np
import pytest
import trimesh

from codemesh.agent import codegen, meshapi


@pytest.mark.parametrize("code", [
    "import os\ndef build(api, evidence):\n    return api.box(1, 1, 1)",
    "def build(api, evidence):\n    return api.__class__",
    "def build(api, evidence):\n    open('x', 'w')",
    "def build(api, evidence):\n    m = api.box(1, 1, 1)\n    m.export('x.ply')\n    return m",
    "def build(api, evidence):\n    global x\n    return x",
    "def build(api, evidence):\n    return eval('1')",
    "def build(api, evidence):\n    return (lambda: 1)()",
    "x = 1",
])
def test_check_rejects(code):
    with pytest.raises(codegen.SandboxError):
        codegen.check(code)


def test_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(codegen, "TIMEOUT", 3)
    (tmp_path / "mesh").mkdir()
    trimesh.creation.box().export(tmp_path / "mesh" / "high.ply")
    with pytest.raises(codegen.SandboxError, match="exceeded"):
        codegen.execute("def build(api, evidence):\n    while True:\n        pass", tmp_path)


def test_execute_returns_mesh(tmp_path):
    (tmp_path / "mesh").mkdir()
    trimesh.creation.box(extents=(.1, .2, .1)).export(tmp_path / "mesh" / "high.ply")
    code = ("def build(api, evidence):\n"
            "    return api.combine([api.cylinder(0.04, evidence.height), api.torus(0.03, 0.005)])")
    mesh = codegen.execute(code, tmp_path)
    assert len(mesh.faces) and codegen.validate(mesh, {"min": [-1] * 3, "max": [1] * 3})["watertight"]


@pytest.mark.parametrize("body", [
    meshapi.box(1, 2, 3), meshapi.cylinder(.1, .2), meshapi.capsule(.1, .2), meshapi.torus(.1, .02),
    meshapi.revolve([(.04, 0), (.05, .1)]), meshapi.revolve([(.04, 0), (.05, .1)], wall=.004, floor=.006),
    meshapi.sweep([(0, 0, 0), (.05, .02, 0), (.06, .06, 0)], (.006, .004)),
    meshapi.sweep([(math.cos(t), 0, math.sin(t)) for t in np.linspace(0, 6, 12)], .1, closed=True),
    meshapi.extrude([(0, 0), (1, 0), (1, 1), (0, 1)], .5), meshapi.mirror(meshapi.box(1, 1, 1), 0),
])
def test_primitives_watertight(body):
    assert body.is_watertight and body.volume > 0
