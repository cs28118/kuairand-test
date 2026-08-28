"""Docker-backed disposable workspaces for generated experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import csv
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from .contracts import ExperimentSpec
from .dependencies import request_profile
from .guardrails import reject_protected_paths


class IsolationError(RuntimeError):
    """Raised when an experiment cannot be safely isolated or executed."""


@dataclass
class ExecutionOutcome:
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    modified_files: list[str]
    failure_reason: str | None = None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            snapshot[str(path.relative_to(root))] = _file_hash(path)
    return snapshot


def diff_paths(git_diff: str) -> list[str]:
    paths: list[str] = []
    for line in git_diff.splitlines():
        match = re.match(r"^(?:\+\+\+|---) [ab]/(.+)$", line)
        if match and match.group(1) != "/dev/null":
            paths.append(match.group(1))
    return sorted(set(paths))


class DockerWorkspace:
    """Creates a copy that excludes protected judge files and host metadata."""

    def __init__(self, repo_root: str | Path, parent: str | Path):
        self.repo_root = Path(repo_root).resolve()
        self.parent = Path(parent).resolve()
        self.path = self.parent / f"workspace-{uuid.uuid4().hex[:10]}"
        self._before: dict[str, str] = {}

    def prepare(self, spec: ExperimentSpec) -> Path:
        self.parent.mkdir(parents=True, exist_ok=True)
        changed_paths = diff_paths(spec.git_diff)
        reject_protected_paths(changed_paths)
        for candidate in changed_paths:
            if ".." in Path(candidate).parts or Path(candidate).is_absolute():
                raise IsolationError(f"git diff contains an unsafe path: {candidate}")
        ignore = shutil.ignore_patterns(
            ".git", "runs", "__pycache__", "*.pyc", ".env", ".env.*", "data", "submission.csv"
        )
        shutil.copytree(self.repo_root, self.path, ignore=ignore)
        self._copy_development_data()
        # Protected files are intentionally absent from the container.  The
        # host framework remains responsible for official validation.
        protected = json.loads((self.repo_root / "configs" / "official_files.json").read_text(encoding="utf-8"))["sha256"]
        for relative in protected:
            target = self.path / relative
            if target.exists():
                target.unlink()
        self._before = _snapshot(self.path)
        if spec.git_diff:
            process = subprocess.run(
                ["git", "apply", "--recount", "--unidiff-zero", "--whitespace=nowarn", "-"],
                input=spec.git_diff.encode("utf-8"),
                cwd=self.path,
                capture_output=True,
                check=False,
            )
            if process.returncode != 0:
                error = process.stderr.decode("utf-8", errors="replace").strip()
                raise IsolationError(f"git diff could not be applied: {error}")
        return self.path

    def _copy_development_data(self) -> None:
        """Copy only train/valid rows; hidden-test dates never enter Docker."""
        source = self.repo_root / "KuaiRand-Pure" / "data"
        target = self.path / "KuaiRand-Pure" / "data"
        if not source.is_dir():
            return
        target.mkdir(parents=True, exist_ok=True)
        video_features = source / "video_features_basic_pure.csv"
        if video_features.is_file():
            shutil.copy2(video_features, target / video_features.name)
        for name in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
            input_path = source / name
            output_path = target / name
            if not input_path.is_file():
                continue
            with input_path.open(newline="", encoding="utf-8") as input_handle, output_path.open(
                "w", newline="", encoding="utf-8"
            ) as output_handle:
                reader = csv.DictReader(input_handle)
                writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames or [])
                writer.writeheader()
                for row in reader:
                    date = int(row["date"])
                    if 20220408 <= date <= 20220428:
                        writer.writerow(row)

    def modified_files(self) -> list[str]:
        after = _snapshot(self.path)
        return sorted(
            name for name in set(self._before) | set(after) if self._before.get(name) != after.get(name)
        )

    def cleanup(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)


class DockerExecutor:
    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        timeout_seconds: float = 1800,
        memory_limit: str = "4g",
        cpus: float = 2.0,
        docker_executable: str = "docker",
    ):
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.cpus = cpus
        self.docker_executable = docker_executable

    def execute(self, spec: ExperimentSpec, workspace: DockerWorkspace) -> ExecutionOutcome:
        profile = request_profile(spec.dependency_profile)
        if shutil.which(self.docker_executable) is None:
            raise IsolationError(
                f"Docker executable not found: {self.docker_executable}. "
                "Install Docker Desktop before running the supervised pilot."
            )
        image = self.image or profile.docker_image
        container = f"kuairand-exp-{uuid.uuid4().hex[:12]}"
        result_path = "/workspace/" + spec.result_file.replace("\\", "/")
        command = [
            self.docker_executable,
            "run",
            "--rm",
            "--name",
            container,
            "--network",
            "none",
            "--read-only",
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpus),
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "-e",
            f"EXPERIMENT_SEED={spec.seed}",
            "-e",
            f"EXPERIMENT_RESULT_PATH={result_path}",
            "-v",
            f"{workspace.path.resolve()}:/workspace:rw",
            "-w",
            "/workspace",
            image,
            *spec.command,
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    [self.docker_executable, "kill", container],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            process.kill()
            stdout, stderr = process.communicate()
            return ExecutionOutcome(
                returncode=None,
                timed_out=True,
                stdout=stdout,
                stderr=stderr,
                modified_files=workspace.modified_files(),
                failure_reason=f"experiment exceeded timeout of {self.timeout_seconds:g} seconds",
            )
        return ExecutionOutcome(
            returncode=process.returncode,
            timed_out=False,
            stdout=stdout,
            stderr=stderr,
            modified_files=workspace.modified_files(),
            failure_reason=None if process.returncode == 0 else f"experiment exited with code {process.returncode}",
        )
