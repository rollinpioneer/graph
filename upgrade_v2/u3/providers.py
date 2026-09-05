"""OpenAI-compatible provider adapters with secret-safe result persistence."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from .common import redact_text


@dataclass
class ModelCallResult:
    request_id: str
    provider: str
    requested_model: str
    returned_model: str | None
    response_id: str | None
    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str | None
    latency_seconds: float
    network_retries: int
    schema_repair_attempted: bool = False
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["content"] = self.content
        return value


def parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        raise ValueError("code fence is not allowed")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("candidate must be one JSON object")
    return value


def sanitize_response(response: object) -> dict[str, Any]:
    """Retain only public metadata and final content, never headers or CoT."""
    raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)  # type: ignore[arg-type]
    raw.pop("reasoning_content", None)
    choices = raw.get("choices") or []
    safe_choices: list[dict[str, Any]] = []
    for choice in choices:
        choice = dict(choice)
        message = dict(choice.get("message") or {})
        present = "reasoning_content" in message
        message.pop("reasoning_content", None)
        if present:
            message["reasoning_content_present"] = True
        choice["message"] = message
        safe_choices.append(choice)
    usage = dict(raw.get("usage") or {})
    # Some compatible endpoints nest thought-token usage under details.
    return {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "created": raw.get("created"),
        "usage": usage,
        "choices": safe_choices,
    }


def _usage(response: object) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    return prompt, completion, reasoning


def _error_status(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return int(value) if isinstance(value, int) else None


def _call(
    *,
    provider: str,
    request_id: str,
    model: str,
    api_key_env: str,
    base_url_env: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    timeout: int,
    response_format: dict[str, Any],
    extra_body: dict[str, Any],
    network_retries: int,
) -> tuple[ModelCallResult, dict[str, Any] | None]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - validated before live run
        raise RuntimeError("openai package is required") from exc
    key = os.environ.get(api_key_env, "")
    base_url = os.environ.get(base_url_env, "")
    if len(key) < 20:
        raise RuntimeError(f"{api_key_env} is missing or too short")
    if not base_url:
        raise RuntimeError(f"{base_url_env} is not configured")
    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
    last_error: Exception | None = None
    started = time.monotonic()
    for attempt in range(network_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_output_tokens,
                response_format=response_format,  # type: ignore[arg-type]
                extra_body=extra_body,
            )
            message = response.choices[0].message if response.choices else None
            content = (getattr(message, "content", None) or "") if message else ""
            input_tokens, output_tokens, reasoning_tokens = _usage(response)
            return ModelCallResult(
                request_id=request_id, provider=provider, requested_model=model,
                returned_model=getattr(response, "model", None), response_id=getattr(response, "id", None),
                content=content, input_tokens=input_tokens, output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                finish_reason=(getattr(response.choices[0], "finish_reason", None) if response.choices else None),
                latency_seconds=round(time.monotonic() - started, 6), network_retries=attempt,
            ), sanitize_response(response)
        except Exception as exc:  # endpoint-specific exception classes vary
            last_error = exc
            status = _error_status(exc)
            retryable = status in {429, 500, 502, 503, 504} or exc.__class__.__name__ in {"APITimeoutError", "APIConnectionError"}
            if not retryable or attempt >= network_retries:
                break
            time.sleep(2**(attempt + 1))
    assert last_error is not None
    return ModelCallResult(
        request_id=request_id, provider=provider, requested_model=model,
        returned_model=None, response_id=None, content="", input_tokens=None,
        output_tokens=None, reasoning_tokens=None, finish_reason=None,
        latency_seconds=round(time.monotonic() - started, 6), network_retries=network_retries,
        http_status=_error_status(last_error), error_type=last_error.__class__.__name__,
        error_message=redact_text(str(last_error))[:500],
    ), None


def call_qwen(
    *, request_id: str, model: str, messages: list[dict[str, str]], schema: dict[str, Any],
    response_mode: str, max_output_tokens: int, timeout: int, network_retries: int,
) -> tuple[ModelCallResult, dict[str, Any] | None]:
    if response_mode == "strict_json_schema":
        response_format = {"type": "json_schema", "json_schema": {"name": "u3_candidate_graph_v1", "strict": True, "schema": schema}}
    elif response_mode == "json_object_local_validation":
        response_format = {"type": "json_object"}
    else:
        raise ValueError(f"unknown Qwen response mode: {response_mode}")
    return _call(
        provider="qwen", request_id=request_id, model=model, api_key_env="QWEN_API_KEY", base_url_env="QWEN_BASE_URL",
        messages=messages, max_output_tokens=max_output_tokens, timeout=timeout, response_format=response_format,
        extra_body={"enable_thinking": True}, network_retries=network_retries,
    )


def call_deepseek(
    *, request_id: str, model: str, messages: list[dict[str, str]], max_output_tokens: int,
    timeout: int, network_retries: int,
) -> tuple[ModelCallResult, dict[str, Any] | None]:
    return _call(
        provider="deepseek", request_id=request_id, model=model, api_key_env="DEEPSEEK_API_KEY", base_url_env="DEEPSEEK_BASE_URL",
        messages=messages, max_output_tokens=max_output_tokens, timeout=timeout, response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "enabled"}}, network_retries=network_retries,
    )


def is_qwen_schema_format_error(result: ModelCallResult) -> bool:
    text = (result.error_message or "").lower()
    return result.error_type is not None and any(token in text for token in ("json_schema", "response_format", "schema", "unsupported"))
