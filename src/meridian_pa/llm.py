"""Thin wrapper around the Anthropic SDK.

Claude is optional. Every caller has a deterministic fallback, and any API
error, refusal, or schema mismatch drops the caller back to that fallback
instead of failing the review. Only PHI-redacted text is ever sent.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from . import config

log = logging.getLogger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _client():
    import anthropic

    return anthropic.Anthropic()


def _request(**kwargs):
    """messages.create with server-side refusal fallbacks enabled."""
    client = _client()
    return client.beta.messages.create(
        model=config.CLAUDE_MODEL,
        betas=[_FALLBACK_BETA],
        extra_body={"fallbacks": "default"},
        **kwargs,
    )


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def structured_call(system: str, user: str, schema: dict, max_tokens: int = 4000) -> dict | None:
    """Single call constrained to a JSON schema. Returns None on any failure."""
    if not config.llm_enabled():
        return None
    try:
        resp = _request(
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": config.CLAUDE_EFFORT, "format": {"type": "json_schema", "schema": schema}},
        )
        if resp.stop_reason in {"refusal", "max_tokens"}:
            log.warning("Claude stop_reason=%s; using deterministic fallback", resp.stop_reason)
            return None
        return json.loads(_text(resp))
    except Exception as exc:  # network, auth, rate limit, bad JSON
        log.warning("Claude call failed (%s); using deterministic fallback", exc)
        return None


def tool_loop(
    system: str,
    user: str,
    tools: list[dict],
    registry: dict[str, Callable[..., Any]],
    max_turns: int = 6,
) -> tuple[str, list[dict]] | None:
    """Manual agentic loop. Returns (final_text, tool_call_log) or None on failure."""
    if not config.llm_enabled():
        return None
    messages: list[dict] = [{"role": "user", "content": user}]
    calls: list[dict] = []
    try:
        for _ in range(max_turns):
            resp = _request(
                max_tokens=4000,
                system=system,
                tools=tools,
                tool_choice={"type": "auto"},
                messages=messages,
                output_config={"effort": config.CLAUDE_EFFORT},
            )
            if resp.stop_reason in {"refusal", "max_tokens"}:
                return None
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return _text(resp), calls
            results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                fn = registry.get(block.name)
                try:
                    out = fn(**block.input) if fn else {"error": f"unknown tool {block.name}"}
                    is_error = "error" in out if isinstance(out, dict) else False
                except Exception as exc:
                    out, is_error = {"error": str(exc)}, True
                calls.append({"tool": block.name, "input": dict(block.input), "output": out})
                results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(out, default=str), "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})
        return None
    except Exception as exc:
        log.warning("Claude tool loop failed (%s); using deterministic fallback", exc)
        return None
