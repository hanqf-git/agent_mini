from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .llm import OpenAICompatLLM
from .tools import ToolRegistry


@dataclass
class AgentResult:
    ok: bool
    message: str


class SimpleAgent:
    """A minimal agent with explicit command routing and tool support."""

    def __init__(
        self,
        registry: ToolRegistry,
        llm: OpenAICompatLLM | None = None,
        debug: bool = False,
        debug_printer: Callable[[str], None] | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.history: list[dict[str, Any]] = []
        self.debug = debug
        self.debug_printer = debug_printer or (lambda msg: print(msg))

    def _debug(self, message: str) -> None:
        if not self.debug:
            return
        self.debug_printer(f"[DEBUG] {message}")

    def _compact(self, value: Any, limit: int = 300) -> str:
        text = str(value).replace("\n", "\\n")
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    def handle(self, user_input: str) -> AgentResult:
        text = user_input.strip()
        if not text:
            return AgentResult(ok=False, message="请输入内容。")

        self._debug(f"input={self._compact(text)}")

        if text == "tools":
            self._debug("route=tools")
            return AgentResult(ok=True, message=self._render_tools())

        if text.startswith("call "):
            self._debug("route=manual_tool_call")
            return self._handle_call(text)

        if self.llm is not None:
            self._debug("route=llm")
            return self._handle_llm(text)

        self._debug("route=fallback")
        return AgentResult(ok=True, message=self._default_response(text))

    def _handle_llm(self, text: str) -> AgentResult:
        assert self.llm is not None
        system_prompt = (
            "你是一个可调用工具的智能体。"
            "你可以直接使用已提供的工具，不需要用户手动输入 call 命令。"
            "若工具足以回答问题，先调用工具再给最终答案。"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *self.history,
            {"role": "user", "content": text},
        ]

        tool_specs = self.registry.to_openai_tools()
        max_tool_rounds = 30
        self._debug(
            f"llm_start history_messages={len(self.history)} tools={len(tool_specs)} max_rounds={max_tool_rounds}"
        )

        try:
            for round_index in range(max_tool_rounds):
                self._debug(f"llm_round={round_index + 1} request_messages={len(messages)}")
                response = self.llm.chat(messages, tools=tool_specs)
                self._debug(
                    "llm_response "
                    f"content={self._compact(response.content)} "
                    f"tool_calls={len(response.tool_calls)}"
                )

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                }
                if response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in response.tool_calls
                    ]
                messages.append(assistant_msg)

                if not response.tool_calls:
                    answer = response.content or "模型没有返回内容。"
                    self._debug(f"llm_final_answer={self._compact(answer)}")
                    self.history.extend(messages[1:])
                    self._debug(f"history_updated total_messages={len(self.history)}")
                    return AgentResult(ok=True, message=answer)

                for tc in response.tool_calls:
                    self._debug(
                        "tool_call_received "
                        f"id={self._compact(tc.id, 80)} "
                        f"name={self._compact(tc.name, 80)} "
                        f"arguments={self._compact(tc.arguments)}"
                    )
                    if not tc.name:
                        tool_output = "工具调用失败: 缺少工具名。"
                    elif not self.registry.has(tc.name):
                        tool_output = f"工具调用失败: 未知工具 {tc.name}"
                    else:
                        try:
                            args = json.loads(tc.arguments or "{}")
                            if not isinstance(args, dict):
                                raise ValueError("参数必须是 JSON 对象")
                            self._debug(
                                f"tool_exec name={tc.name} args={self._compact(json.dumps(args, ensure_ascii=False))}"
                            )
                            tool_output = self.registry.call(tc.name, args)
                        except Exception as exc:
                            tool_output = f"工具调用失败: {exc}"

                    self._debug(
                        f"tool_result name={self._compact(tc.name, 80)} output={self._compact(tool_output)}"
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_output),
                        }
                    )

            return AgentResult(ok=False, message="工具调用轮次超限，请重试或简化问题。")
        except Exception as exc:
            self._debug(f"llm_error={self._compact(exc)}")
            return AgentResult(ok=False, message=f"LLM 调用失败: {exc}")

    def _handle_call(self, text: str) -> AgentResult:
        # format: call <tool_name> <json_args>
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            return AgentResult(ok=False, message="格式错误。用法: call <tool_name> <json_args>")

        tool_name = parts[1]
        args_raw = parts[2] if len(parts) > 2 else "{}"
        self._debug(f"manual_tool_input name={tool_name} raw_args={self._compact(args_raw)}")

        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return AgentResult(ok=False, message="参数必须是 JSON，例如: {\"a\":1, \"b\":2}")

        if not isinstance(args, dict):
            return AgentResult(ok=False, message="JSON 参数必须是对象类型。")

        try:
            output = self.registry.call(tool_name, args)
            self._debug(f"manual_tool_result name={tool_name} output={self._compact(output)}")
            return AgentResult(ok=True, message=f"[tool:{tool_name}] {output}")
        except Exception as exc:
            self._debug(f"manual_tool_error name={tool_name} error={self._compact(exc)}")
            return AgentResult(ok=False, message=f"工具调用失败: {exc}")

    def _render_tools(self) -> str:
        tools = self.registry.list_tools()
        if not tools:
            return "当前没有可用工具。"

        lines = ["可用工具:"]
        for t in tools:
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def _default_response(self, text: str) -> str:
        return (
            "我是一个最小 Agent。\n"
            "你可以输入 `tools` 查看工具，或使用 `call <tool_name> <json_args>` 调用工具。\n"
            f"你刚刚说的是: {text}"
        )
