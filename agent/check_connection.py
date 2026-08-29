"""Check organization LLM credentials and model routing without running a pilot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib import error, request

from .llm import load_dotenv


def _get_json(url: str, api_key: str) -> dict:
    http_request = request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with request.urlopen(http_request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("gateway returned a non-object JSON response")
    return value


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Check LLM gateway authentication and model availability.")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--probe", action="store_true", help="also make one minimal Responses API request")
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        print("FAIL: OPENAI_API_KEY is missing", file=sys.stderr)
        return 1
    if not args.model:
        print("FAIL: LLM_MODEL is missing", file=sys.stderr)
        return 1

    try:
        catalog = _get_json(base_url + "/models", api_key)
    except RuntimeError as exc:
        print(f"FAIL: gateway authentication or /models request failed: {exc}", file=sys.stderr)
        return 1
    model_ids = sorted({str(item["id"]) for item in catalog.get("data", []) if isinstance(item, dict) and item.get("id")})
    print(f"Gateway: {base_url}")
    print("API key: accepted by /models")
    print(f"Configured model: {args.model}")
    print("Available models:")
    for model_id in model_ids:
        print(f"  {model_id}")
    if args.model not in model_ids:
        print(f"FAIL: configured model is not available: {args.model}", file=sys.stderr)
        return 2
    if not args.probe:
        print("PASS: credentials and configured model are available")
        return 0

    payload = json.dumps({"model": args.model, "input": "Reply with exactly OK.", "max_output_tokens": 8}).encode("utf-8")
    http_request = request.Request(
        base_url + "/responses", data=payload, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with request.urlopen(http_request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        print(f"FAIL: /responses request failed with HTTP {exc.code}: {detail}", file=sys.stderr)
        return 3
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: /responses request failed: {exc}", file=sys.stderr)
        return 3
    print(f"Responses API: accepted (id={result.get('id', 'unknown')})")
    print("PASS: API key, model, and Responses API request are working")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
