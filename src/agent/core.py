from __future__ import annotations

import json
import re
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
        reflection_max_rounds: int = 5,
        reflection_pass_score: float = 8.0,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.history: list[dict[str, Any]] = []
        self.debug = debug
        self.debug_printer = debug_printer or (lambda msg: print(msg))
        self.reflection_max_rounds = max(1, int(reflection_max_rounds))
        self.reflection_pass_score = max(0.0, min(float(reflection_pass_score), 10.0))

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
            "若用户问题包含 now/current/currently/today/当前/现在 等时间语义，"
            "优先调用时间工具获取当前时间后再回答。"
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
            used_tools = False
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
                    if used_tools:
                        self._debug("reflection_skipped reason=tool_grounded_response")
                    else:
                        self._debug(
                            "reflection_entry "
                            f"enabled={self.llm is not None} "
                            f"max_rounds={self.reflection_max_rounds} "
                            f"pass_score={self.reflection_pass_score:.1f} "
                            f"draft={self._compact(answer)}"
                        )
                        answer = self._reflect_and_revise_answer(text, answer)
                    messages[-1]["content"] = answer
                    self._debug(f"llm_final_answer={self._compact(answer)}")
                    self.history.extend(messages[1:])
                    self._debug(f"history_updated total_messages={len(self.history)}")
                    return AgentResult(ok=True, message=answer)

                for tc in response.tool_calls:
                    used_tools = True
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

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None

        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            return None

        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
        return None

    def _score_answer(self, question: str, draft_answer: str) -> tuple[float, str]:
        assert self.llm is not None
        system_prompt = (
            "你是严格的回答质量评审。"
            "请只输出 JSON 对象，格式为"
            '{"score": number, "reason": string, "improve": string}。'
            "score 范围 0 到 10，可带 1 位小数。"
        )
        user_prompt = (
            f"用户问题:\n{question}\n\n"
            f"候选答案:\n{draft_answer}\n\n"
            "请评估答案是否正确、完整、清晰，并给出改进建议。"
        )
        response = self.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        self._debug(f"reflection_score_raw={self._compact(response.content)}")

        payload = self._extract_json_object(response.content)
        if payload is None:
            self._debug("reflection_score_parse_failed payload=None")
            return 0.0, "评分解析失败，请明确指出错误并补全关键信息。"

        self._debug(f"reflection_score_parsed={self._compact(payload)}")

        score_raw = payload.get("score", 0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(score, 10.0))

        reason = str(payload.get("reason") or "")
        improve = str(payload.get("improve") or "")
        feedback = "；".join([part for part in [reason, improve] if part.strip()]).strip()
        if not feedback:
            feedback = "请提高正确性、完整性和表达清晰度。"

        return score, feedback

    def _revise_answer(self, question: str, draft_answer: str, feedback: str) -> str:
        assert self.llm is not None
        system_prompt = (
            "你是回答改写助手。"
            "请基于评审意见改进答案，优先保证正确性和完整性。"
            "直接输出改进后的最终答案，不要输出解释或标签。"
        )
        user_prompt = (
            f"用户问题:\n{question}\n\n"
            f"当前答案草稿:\n{draft_answer}\n\n"
            f"评审意见:\n{feedback}\n\n"
            "请给出修正后的答案。"
        )
        response = self.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        revised = (response.content or "").strip()
        self._debug(
            "reflection_revise_raw "
            f"content={self._compact(response.content)}"
        )
        if not revised:
            self._debug("reflection_revise_empty_keep_previous=true")
            return draft_answer
        return revised

    def _reflect_and_revise_answer(self, question: str, draft_answer: str) -> str:
        if self.llm is None:
            self._debug("reflection_skipped reason=no_llm")
            return draft_answer

        current = draft_answer
        last_score = 0.0
        rounds_used = 0
        end_reason = "max_rounds"
        self._debug(
            "reflection_start "
            f"max_rounds={self.reflection_max_rounds} "
            f"pass_score={self.reflection_pass_score:.1f}"
        )
        for idx in range(self.reflection_max_rounds):
            rounds_used = idx + 1
            try:
                score, feedback = self._score_answer(question, current)
            except Exception as exc:
                self._debug(f"reflection_score_error round={idx + 1} error={self._compact(exc)}")
                end_reason = "score_error"
                break

            last_score = score

            self._debug(
                "reflection_score "
                f"round={idx + 1}/{self.reflection_max_rounds} "
                f"score={score:.1f} threshold={self.reflection_pass_score:.1f} "
                f"feedback={self._compact(feedback)}"
            )

            if score >= self.reflection_pass_score:
                self._debug(f"reflection_passed round={idx + 1}")
                end_reason = "passed"
                break

            if idx >= self.reflection_max_rounds - 1:
                self._debug("reflection_reached_max_rounds")
                end_reason = "max_rounds"
                break

            try:
                revised = self._revise_answer(question, current, feedback)
            except Exception as exc:
                self._debug(f"reflection_revise_error round={idx + 1} error={self._compact(exc)}")
                end_reason = "revise_error"
                break

            self._debug(
                "reflection_revised "
                f"round={idx + 1} "
                f"content={self._compact(revised)}"
            )
            current = revised

        self._debug(
            "reflection_end "
            f"reason={end_reason} "
            f"rounds_used={rounds_used} "
            f"last_score={last_score:.1f} "
            f"final={self._compact(current)}"
        )
        return current

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
