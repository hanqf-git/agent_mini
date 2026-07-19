from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 30


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat client over HTTP."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc}") from exc

        try:
            result = json.loads(body)
            message = result["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            tool_calls_raw = message.get("tool_calls") or []
            tool_calls: list[ToolCall] = []
            for call in tool_calls_raw:
                fn = call.get("function") or {}
                tool_calls.append(
                    ToolCall(
                        id=str(call.get("id") or ""),
                        name=str(fn.get("name") or ""),
                        arguments=str(fn.get("arguments") or "{}"),
                    )
                )

            # Fallback: some providers place function call payloads in content as
            # <tool_call>{"name":"...","arguments":{...}}</tool_call>.
            if not tool_calls and content:
                match = re.search(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", content)
                if match:
                    payload_text = match.group(1)
                    payload = json.loads(payload_text)
                    name = str(payload.get("name") or "")
                    arguments_value = payload.get("arguments") or {}
                    if not isinstance(arguments_value, dict):
                        arguments_value = {}
                    tool_calls.append(
                        ToolCall(
                            id="content-tool-call-1",
                            name=name,
                            arguments=json.dumps(arguments_value, ensure_ascii=False),
                        )
                    )
                    content = ""
            return ChatResponse(content=content, tool_calls=tool_calls)
        except Exception as exc:
            raise RuntimeError(f"Invalid LLM response: {body}") from exc
