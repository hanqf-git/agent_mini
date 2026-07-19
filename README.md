# Minimal Agent (Python)

一个最简单、可扩展、支持工具调用的 Agent 示例。

已支持 OpenAI 兼容 API，可接入真实大模型，并由模型自动决定何时调用工具。

## 1. 运行

- Windows PowerShell:

```powershell
cd d:\2026-06-28-agent-dev
py -m src.main
```

开启 Debug 模式（打印路由、LLM 交互与工具调用过程）：

```powershell
py -m src.main --debug
```

或通过环境变量开启：

```powershell
$env:AIDEMO_DEBUG = "1"
py -m src.main
```

如果你的环境有 `python` 命令，也可以使用：

```powershell
python -m src.main
```

## 2. 接入真实大模型（OpenAI 兼容）

本项目默认读取以下环境变量：

- `AIDEMO_BASE_URL`，默认 `https://gateway.aichina.intel.com/v1`
- `AIDEMO_MODEL`，默认 `qwen3-vl-235b-a22b-instruct-fp8`
- `AIDEMO_API_KEY`（或 `OPENAI_API_KEY`）
- `TAVILY_API_KEY`（启用 `tavily_search` 工具时需要）
- `TAVILY_PROXY_URL`（可选，默认 `http://child-prc.intel.com:913`）

Windows PowerShell 示例：

```powershell
$env:AIDEMO_BASE_URL = "https://gateway.aichina.intel.com/v1"
$env:AIDEMO_MODEL = "qwen3-vl-235b-a22b-instruct-fp8"
$env:AIDEMO_API_KEY = "<your_api_key>"
$env:TAVILY_API_KEY = "<your_tavily_api_key>"
$env:TAVILY_PROXY_URL = "http://child-prc.intel.com:913"
py -m src.main
```

设置 API Key 后，启动会显示 `LLM enabled`。

## 3. 交互命令

- 查看工具列表: `tools`
- 手动调试工具（可选）: `call <tool_name> <json_args>`
- 退出: `exit`

示例:

- 自动工具调用示例: `现在几点了？`（模型会自动调用 `now`）
- 自动工具调用示例: `帮我算 3 + 5`（模型会自动调用 `add`）
- 自动工具调用示例: `帮我搜索今天的 AI 新闻`（模型可调用 `tavily_search`）
- 手动调试: `call add {"a": 3, "b": 5}`
- 手动调试: `call tavily_search {"query": "latest AI news", "max_results": 3}`

## 4. 目录结构

- `src/agent/core.py`: Agent 核心流程（解析输入、调用工具、兜底回复）
- `src/agent/tools.py`: 工具抽象与注册中心
- `src/agent/builtins.py`: 示例内置工具
- `src/agent/llm.py`: OpenAI 兼容 API 客户端
- `src/main.py`: CLI 入口与 LLM 初始化

## 5. 扩展方式

- 新增工具:
  1. 在 `src/agent/builtins.py` 写一个函数，签名 `fn(args: dict) -> str`
  2. 注册到 `build_default_registry()`
- 替换思考逻辑:
  - 在 `SimpleAgent._default_response()` 中接入 LLM 或规划模块
- 增加状态:
  - 在 `SimpleAgent` 增加会话记忆字段（如 history）
