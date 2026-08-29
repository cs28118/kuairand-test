"""Provider wrapper for structured, auditable proposal generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol
from urllib import error, request

from .proposal import experiment_spec_schema


class LLMError(RuntimeError):
    """Raised when the configured LLM cannot return a proposal."""


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""
    dotenv_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.is_file():
        return
    key_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key_pattern.fullmatch(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class LLMRequest:
    provider: str
    model: str
    prompt: str

    def audit_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    response_id: str | None
    usage: dict[str, int]
    raw: dict[str, Any]

    def audit_dict(self) -> dict[str, Any]:
        return {"response_id": self.response_id, "text": self.text, "usage": self.usage, "raw": self.raw}


class ProposalClient(Protocol):
    def generate(self, proposal_request: LLMRequest) -> LLMResponse: ...


class OpenAIResponsesClient:
    """Minimal stdlib OpenAI Responses API client; no SDK dependency is needed."""

    default_base_url = "https://api.openai.com/v1"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required to generate a proposal")
        candidate = (base_url or os.environ.get("OPENAI_BASE_URL") or self.default_base_url).rstrip("/")
        if not candidate.startswith(("https://", "http://")):
            raise LLMError("OPENAI_BASE_URL must begin with http:// or https://")
        self.endpoint = candidate + "/responses"

    @staticmethod
    def _payload(proposal_request: LLMRequest) -> dict[str, Any]:
        return {
            "model": proposal_request.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": proposal_request.prompt}]}],
            "text": {"format": {"type": "json_schema", "name": "experiment_spec", "strict": True, "schema": experiment_spec_schema()}},
        }

    def generate(self, proposal_request: LLMRequest) -> LLMResponse:
        encoded = json.dumps(self._payload(proposal_request)).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=encoded,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with request.urlopen(http_request, timeout=90) as http_response:
                raw = json.loads(http_response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"OpenAI request failed with HTTP {exc.code}: {detail}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        text = raw.get("output_text")
        if not isinstance(text, str):
            text = "".join(
                item.get("text", "")
                for output in raw.get("output", []) if isinstance(output, dict)
                for item in output.get("content", []) if isinstance(item, dict) and item.get("type") == "output_text"
            )
        if not text:
            raise LLMError("OpenAI response did not contain output_text")
        usage_raw = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens", 0)),
            "output_tokens": int(usage_raw.get("output_tokens", 0)),
            "cached_tokens": int((usage_raw.get("input_tokens_details") or {}).get("cached_tokens", 0)),
            "total_tokens": int(usage_raw.get("total_tokens", 0)),
        }
        return LLMResponse(text=text, response_id=raw.get("id"), usage=usage, raw=raw)


def configured_client(provider: str) -> ProposalClient:
    if provider.lower() == "openai":
        return OpenAIResponsesClient()
    raise LLMError(f"unsupported LLM provider: {provider}")
