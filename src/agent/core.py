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
        execution_mode: str = "auto",
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.history: list[dict[str, Any]] = []
        self.debug = debug
        self.debug_printer = debug_printer or (lambda msg: print(msg))
        self.reflection_max_rounds = max(1, int(reflection_max_rounds))
        self.reflection_pass_score = max(0.0, min(float(reflection_pass_score), 10.0))
        normalized_mode = execution_mode.strip().lower()
        if normalized_mode not in {"auto", "react", "plan_react"}:
            normalized_mode = "auto"
        self.execution_mode = normalized_mode

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

        if text.startswith("/mode"):
            self._debug("route=mode_switch")
            return self._handle_mode_command(text)

        if text.startswith("call "):
            self._debug("route=manual_tool_call")
            return self._handle_call(text)

        if self.llm is not None:
            self._debug("route=llm")
            return self._handle_llm(text)

        self._debug("route=fallback")
        return AgentResult(ok=True, message=self._default_response(text))

    def _handle_llm(self, text: str) -> AgentResult:
        mode = self._select_execution_mode(text)
        self._debug(f"execution_mode_selected configured={self.execution_mode} selected={mode}")
        if mode == "plan_react":
            return self._handle_plan_react(text)
        return self._run_react_flow(text)

    def _select_execution_mode(self, text: str) -> str:
        if self.execution_mode in {"react", "plan_react"}:
            return self.execution_mode

        lower_text = text.lower()
        planning_keywords = [
            "计划",
            "步骤",
            "方案",
            "roadmap",
            "plan",
            "compare",
            "comparison",
            "对比",
            "分析",
            "report",
            "总结",
        ]
        contains_planning_keyword = any(k in lower_text for k in planning_keywords)
        is_long_query = len(text) >= 48
        has_many_clauses = text.count("，") + text.count(",") + text.count(";") >= 2

        if contains_planning_keyword or (is_long_query and has_many_clauses):
            return "plan_react"
        return "react"

    def _build_react_system_prompt(self) -> str:
        return (
            "你是一个可调用工具的智能体。"
            "你可以直接使用已提供的工具，不需要用户手动输入 call 命令。"
            "若工具足以回答问题，先调用工具再给最终答案。"
            "若用户问题包含 now/current/currently/today/当前/现在 等时间语义，"
            "优先调用时间工具获取当前时间后再回答。"
        )

    def _run_react_flow(
        self,
        text: str,
        *,
        include_history: bool = True,
        persist_history: bool = True,
        enable_reflection: bool = True,
    ) -> AgentResult:
        assert self.llm is not None
        system_prompt = self._build_react_system_prompt()
        history_messages = self.history if include_history else []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": text},
        ]

        tool_specs = self.registry.to_openai_tools()
        max_tool_rounds = 30
        self._debug(
            "llm_start "
            f"history_messages={len(history_messages)} "
            f"tools={len(tool_specs)} "
            f"max_rounds={max_tool_rounds} "
            f"reflection={enable_reflection}"
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
                    elif enable_reflection:
                        self._debug(
                            "reflection_entry "
                            f"enabled={self.llm is not None} "
                            f"max_rounds={self.reflection_max_rounds} "
                            f"pass_score={self.reflection_pass_score:.1f} "
                            f"draft={self._compact(answer)}"
                        )
                        answer = self._reflect_and_revise_answer(text, answer)
                    else:
                        self._debug("reflection_skipped reason=disabled_for_flow")
                    messages[-1]["content"] = answer
                    self._debug(f"llm_final_answer={self._compact(answer)}")
                    if persist_history:
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

    def _extract_plan_steps(self, text: str) -> list[str]:
        payload = self._extract_json_object(text)
        if payload is None:
            return []

        steps_raw = payload.get("steps")
        if not isinstance(steps_raw, list):
            return []

        steps: list[str] = []
        for item in steps_raw:
            if not isinstance(item, str):
                continue
            step = item.strip()
            if step:
                steps.append(step)
        return steps[:8]

    def _build_plan(self, user_text: str) -> list[str]:
        assert self.llm is not None
        system_prompt = (
            "你是任务规划助手。"
            "请把用户任务拆解为 3 到 6 个可执行步骤。"
            "只输出 JSON 对象，格式为 {\"steps\": [\"...\"]}。"
        )
        user_prompt = (
            f"用户任务:\n{user_text}\n\n"
            "请输出可执行步骤计划。"
        )
        response = self.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        self._debug(f"plan_raw={self._compact(response.content)}")
        steps = self._extract_plan_steps(response.content)
        if not steps:
            self._debug("plan_parse_failed fallback_to_single_step")
            return ["直接完成用户请求并返回最终结果"]
        self._debug(f"plan_parsed steps={self._compact(steps)}")
        return steps

    def _handle_plan_react(self, text: str) -> AgentResult:
        assert self.llm is not None
        steps = self._build_plan(text)
        step_results: list[str] = []
        self._debug(f"plan_start step_count={len(steps)}")

        for idx, step in enumerate(steps, start=1):
            step_prompt = (
                "你在执行一个多步骤任务。\n"
                f"原始任务:\n{text}\n\n"
                f"当前步骤 ({idx}/{len(steps)}):\n{step}\n\n"
                "请完成此步骤。必要时调用工具。"
                "输出该步骤结果摘要。"
            )
            self._debug(f"plan_step_start index={idx} step={self._compact(step)}")
            step_result = self._run_react_flow(
                step_prompt,
                include_history=False,
                persist_history=False,
                enable_reflection=False,
            )
            if not step_result.ok:
                return AgentResult(ok=False, message=f"计划步骤 {idx} 执行失败: {step_result.message}")
            step_results.append(step_result.message)
            self._debug(f"plan_step_result index={idx} result={self._compact(step_result.message)}")

        plan_lines = "\n".join([f"{i}. {s}" for i, s in enumerate(steps, start=1)])
        result_lines = "\n\n".join(
            [f"步骤{i}结果:\n{r}" for i, r in enumerate(step_results, start=1)]
        )
        final_prompt = (
            f"用户原始请求:\n{text}\n\n"
            f"计划:\n{plan_lines}\n\n"
            f"步骤执行结果:\n{result_lines}\n\n"
            "请基于上述信息给出最终答案。"
        )
        self._debug("plan_synthesize_start")
        return self._run_react_flow(
            final_prompt,
            include_history=True,
            persist_history=True,
            enable_reflection=True,
        )

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

    def _handle_mode_command(self, text: str) -> AgentResult:
        # format: /mode <auto|react|plan_react>
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            return AgentResult(
                ok=True,
                message=(
                    f"当前模式: {self.execution_mode}\n"
                    "用法: /mode <auto|react|plan_react>"
                ),
            )

        requested = parts[1].strip().lower()
        if requested not in {"auto", "react", "plan_react"}:
            return AgentResult(
                ok=False,
                message="模式无效。可选: auto, react, plan_react",
            )

        self.execution_mode = requested
        return AgentResult(ok=True, message=f"已切换模式为: {self.execution_mode}")

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
            "你也可以输入 `/mode <auto|react|plan_react>` 切换执行模式。\n"
            f"你刚刚说的是: {text}"
        )
