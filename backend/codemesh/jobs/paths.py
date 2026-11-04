"""Job working-directory layout: jobs/<id>/{input,frames,masks,...}."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

SUBDIRS = ("input", "frames", "masks", "geometry", "mesh", "texture", "export", "logs")
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class JobPaths:
    """Absolute paths of one job's folders. Each stage writes only into its own folder."""

    root: Path

    @property
    def input(self) -> Path:
        return self.root / "input"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def geometry(self) -> Path:
        return self.root / "geometry"

    @property
    def mesh(self) -> Path:
        return self.root / "mesh"

    @property
    def texture(self) -> Path:
        return self.root / "texture"

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def create(self) -> None:
        """Create the job folder and all stage subfolders."""
        for name in SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def remove(self) -> None:
        """Delete the job folder and everything in it."""
        shutil.rmtree(self.root, ignore_errors=True)

    def resolve_artifact(self, rel: str) -> Path | None:
        """Map a client-supplied relative path to a file inside the job folder.

        Returns None for anything that escapes the job folder (.., absolute paths,
        drive letters, symlinks pointing outside) or is not an existing file.
        """
        if not rel or "\\" in rel or ":" in rel or rel.startswith("/"):
            return None
        root = self.root.resolve()
        try:
            path = (root / rel).resolve()
            path.relative_to(root)
        except (ValueError, OSError):
            return None
        return path if path.is_file() else None


def job_paths(jobs_dir: Path, job_id: str) -> JobPaths:
    """Paths for `job_id`. Raises ValueError for ids that are not ours (prevents traversal)."""
    if not JOB_ID_RE.match(job_id):
        raise ValueError(f"invalid job id: {job_id!r}")
    return JobPaths(jobs_dir / job_id)
