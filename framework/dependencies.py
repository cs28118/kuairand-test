"""Approved dependency profiles; this module never installs packages."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util


class DependencyViolation(RuntimeError):
    """Raised for an unknown or disallowed dependency profile."""


@dataclass(frozen=True)
class DependencyProfile:
    name: str
    packages: tuple[str, ...]
    docker_image: str


APPROVED_PROFILES: dict[str, DependencyProfile] = {
    "base": DependencyProfile("base", ("numpy",), "python:3.12-slim"),
    "pytorch": DependencyProfile("pytorch", ("numpy", "torch"), "python:3.12-slim"),
    "lightgbm": DependencyProfile("lightgbm", ("numpy", "lightgbm"), "python:3.12-slim"),
    "recbole": DependencyProfile("recbole", ("numpy", "torch", "recbole"), "python:3.12-slim"),
}


def request_profile(name: str) -> DependencyProfile:
    try:
        return APPROVED_PROFILES[name]
    except KeyError as exc:
        raise DependencyViolation(f"dependency profile is not approved: {name}") from exc


def missing_packages(profile: DependencyProfile) -> list[str]:
    return [package for package in profile.packages if importlib.util.find_spec(package) is None]
