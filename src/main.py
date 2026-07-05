from __future__ import annotations

import os
import sys
from pathlib import Path

from .agent.builtins import build_default_registry
from .agent.core import SimpleAgent
from .agent.llm import LLMConfig, OpenAICompatLLM


def _is_debug_enabled(argv: list[str]) -> bool:
    env_value = os.getenv("AIDEMO_DEBUG", "").strip().lower()
    env_enabled = env_value in {"1", "true", "yes", "on"}
    arg_enabled = "--debug" in argv
    return env_enabled or arg_enabled


def load_dotenv_file(dotenv_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env into process environment."""
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_llm_from_env() -> OpenAICompatLLM | None:
    api_key = os.getenv("AIDEMO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("AIDEMO_BASE_URL", "https://aidemo.intel.cn/v1/chat/completions")
    model = os.getenv("AIDEMO_MODEL", "minimax-latest")
    timeout_seconds = int(os.getenv("AIDEMO_TIMEOUT", "30"))
    config = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    return OpenAICompatLLM(config)


def main() -> None:
    load_dotenv_file()
    debug = _is_debug_enabled(sys.argv[1:])
    llm = build_llm_from_env()
    agent = SimpleAgent(build_default_registry(), llm=llm, debug=debug)
    mode = "LLM enabled" if llm else "LLM disabled"
    debug_mode = "DEBUG on" if debug else "DEBUG off"
    print(f"Simple Agent started ({mode}, {debug_mode}). 输入 tools 查看工具，输入 exit 退出。")

    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        result = agent.handle(user_input)
        prefix = "Agent" if result.ok else "Agent[error]"
        print(f"{prefix}> {result.message}")


if __name__ == "__main__":
    main()
