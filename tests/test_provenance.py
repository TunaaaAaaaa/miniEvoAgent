import subprocess

from evo_repro import RunProvenance


def test_capture_missing_git_checkout_keeps_unknown_fields(tmp_path):
    result = RunProvenance.capture(repo_path=tmp_path, seed=42,
                                   dataset_name="fixture", dataset_revision="v1")
    assert result.git_sha is None
    assert result.git_dirty is None
    assert result.seed == 42
    assert result.dataset_revision == "v1"


def test_capture_git_metadata_without_persisting_command_output(monkeypatch):
    calls = []

    def check_output(args, **kwargs):
        calls.append(args)
        return "abc123\n" if "rev-parse" in args else " M file.py\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)
    result = RunProvenance.capture(seed=0)
    assert result.git_sha == "abc123"
    assert result.git_dirty is True
    assert len(calls) == 2
    assert "file.py" not in result.to_json()
