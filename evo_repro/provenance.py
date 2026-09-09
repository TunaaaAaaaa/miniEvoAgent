"""Explicit experiment provenance; missing source metadata is never guessed."""

from pathlib import Path
import subprocess

from .base import BaseModel


class RunProvenance(BaseModel):
    git_sha: str | None = None
    git_dirty: bool | None = None
    seed: int | None = None
    dataset_name: str | None = None
    dataset_revision: str | None = None

    @classmethod
    def capture(
        cls, *, seed: int | None = None, dataset_name: str | None = None,
        dataset_revision: str | None = None, repo_path: str | Path | None = None,
    ) -> "RunProvenance":
        directory = Path(repo_path) if repo_path is not None else Path(__file__).resolve().parents[1]
        sha = None
        dirty = None
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "-C", str(directory), "status", "--porcelain"],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip())
        except (OSError, subprocess.SubprocessError):
            pass
        return cls(git_sha=sha, git_dirty=dirty, seed=seed,
                   dataset_name=dataset_name, dataset_revision=dataset_revision)
