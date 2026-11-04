"""Request/response bodies of the v1 API (docs/PLAN.md). The Job model lives in jobs/models.py."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Preset = Literal["draft", "standard", "high"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(_Body):
    x: float = Field(ge=0.0, le=1.0)  # normalized image coords, origin top-left
    y: float = Field(ge=0.0, le=1.0)
    label: Literal[0, 1]  # 1 = object, 0 = background


class SelectRequest(_Body):
    frame_index: int = Field(ge=0)
    points: list[Point] = Field(default_factory=list, max_length=64)
    text: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _needs_prompt(self) -> SelectRequest:
        if not self.points and not (self.text and self.text.strip()):
            raise ValueError("give at least one point or a text prompt")
        return self


class KnownDimension(_Body):
    axis: Literal["x", "y", "z"]
    meters: float = Field(gt=0.0, le=1000.0)


class ReconstructRequest(_Body):
    preset: Preset = "standard"
    triangle_budget: Literal["auto"] | Annotated[int, Field(ge=500, le=2_000_000)] = "auto"
    texture_size: Literal[2048, 4096] = 2048
    known_dimension: KnownDimension | None = None


class RefineRequest(_Body):
    instruction: str | None = Field(default=None, max_length=2000)


class FrameInfo(BaseModel):
    index: int
    url: str  # path under the API base, e.g. /api/jobs/<id>/artifacts/frames/0000.jpg
    thumb_url: str
    sharpness: float
    width: int
    height: int


class Health(BaseModel):
    ok: bool
    version: str


class Capabilities(BaseModel):
    geometry_engine: str
    device: str
    precision: str
    blender: bool
    meta_api: bool
    engines: dict[str, bool]  # runtime -> importable
    heic: bool
    max_upload_bytes: int
