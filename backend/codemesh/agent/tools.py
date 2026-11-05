"""The restricted tool set Muse may call. Each tool validates its arguments and edits `Plan`;
the loop then re-runs only the stages the change affects."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

from codemesh.pipeline import mesh as mesh_stage

from .ops import SYMMETRIES

# Re-run order; a change re-runs its stage and everything after it.
STAGES = ("mesh", "texture", "export", "evaluate")
MESH_METHODS = ("hull_depth", "poisson")


@dataclass
class Plan:
    """Settings the agent overrides on top of the job's reconstruct settings."""

    mesh_method: str | None = None
    triangle_budget: int | None = None
    symmetry: str = "none"
    roughness: float | None = None
    metallic: float | None = None
    texture_size: int | None = None

    def settings(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and k != "symmetry"}


def _fn(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {**properties, "rationale": {
            "type": "string", "description": "one sentence: why, citing what you saw"}},
            "required": [*properties, "rationale"]}}}


TOOLS = [
    _fn("set_mesh_method", "Surface reconstruction: 'hull_depth' (visual hull from the masks refined by depth; "
        "robust when depth on the object is sparse) or 'poisson' (fused depth points only).",
        {"method": {"type": "string", "enum": list(MESH_METHODS)}}),
    _fn("set_triangle_budget", "Triangle count of the game-ready low mesh.",
        {"tris": {"type": "integer", "minimum": 1000, "maximum": 150000}}),
    _fn("apply_symmetry", "Post-mesh symmetry in the gravity-aligned frame (+Y up). 'rotational_y' rebuilds the "
        "object as a surface of revolution (cups, bottles, vases; removes handles); 'mirror_x'/'mirror_z' "
        "symmetrize across a vertical plane; 'none' removes it. Kept only if silhouette IoU improves.",
        {"kind": {"type": "string", "enum": list(SYMMETRIES)}}),
    _fn("set_material", "PBR material factors of the exported GLB.",
        {"roughness": {"type": "number", "minimum": 0, "maximum": 1},
         "metallic": {"type": "number", "minimum": 0, "maximum": 1}}),
    _fn("set_texture_size", "Albedo/normal texture resolution.", {"size": {"type": "integer", "enum": [2048, 4096]}}),
    _fn("accept", "The current model is good enough (or nothing more can help); stop.", {}),
]


def _unit(x: Any, name: str) -> float:
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in 0..1")
    return x


def hull_depth_available() -> bool:
    # shortcut: detects the method by name in mesh.py; replace with an exported METHODS tuple once mesh.py has one
    return "hull_depth" in inspect.getsource(mesh_stage)


def apply(plan: Plan, name: str, args: dict[str, Any]) -> tuple[str, str | None]:
    """Apply one tool call to `plan`. Returns (result text for Muse, first stage to re-run or None).
    Invalid arguments raise ValueError (reported back to Muse as the tool result)."""
    if name == "set_mesh_method":
        method = args.get("method")
        if method not in MESH_METHODS:
            raise ValueError(f"method must be one of {MESH_METHODS}")
        if method == "hull_depth" and not hull_depth_available():
            return "not supported in this build (hull_depth meshing not available); keeping poisson", None
        plan.mesh_method = method
        return f"mesh method = {method}", "mesh"
    if name == "set_triangle_budget":
        tris = int(args.get("tris", 0))
        if not 1000 <= tris <= 150000:
            raise ValueError("tris must be in 1000..150000")
        plan.triangle_budget = tris
        return f"triangle budget = {tris}", "mesh"
    if name == "apply_symmetry":
        kind = args.get("kind")
        if kind not in SYMMETRIES:
            raise ValueError(f"kind must be one of {SYMMETRIES}")
        plan.symmetry = kind
        return f"symmetry = {kind}", "mesh"
    if name == "set_material":
        plan.roughness, plan.metallic = _unit(args.get("roughness"), "roughness"), _unit(args.get("metallic"), "metallic")
        return f"material roughness={plan.roughness} metallic={plan.metallic}", "export"
    if name == "set_texture_size":
        size = int(args.get("size", 0))
        if size not in (2048, 4096):
            raise ValueError("size must be 2048 or 4096")
        plan.texture_size = size
        return f"texture size = {size}", "texture"
    if name == "accept":
        return "accepted", None
    raise ValueError(f"unknown tool {name!r}")
