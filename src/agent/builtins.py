from __future__ import annotations

from datetime import datetime
import json
import os
import re
import urllib.error
import urllib.request

from .tools import Tool, ToolRegistry


def add_tool(args: dict) -> str:
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    return str(a + b)


def now_tool(args: dict) -> str:
    _ = args
    return datetime.now().isoformat(timespec="seconds")


def current_time_tool(args: dict) -> str:
    _ = args
    now = datetime.now().astimezone()
    return "\n".join(
        [
            f"iso_datetime: {now.isoformat(timespec='seconds')}",
            f"date: {now.date().isoformat()}",
            f"time: {now.time().strftime('%H:%M:%S')}",
            f"timezone: {now.tzname() or 'local'}",
            f"unix_timestamp: {int(now.timestamp())}",
        ]
    )


def fetch_url_tool(args: dict) -> str:
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("url must start with http:// or https://")

    timeout_seconds = float(args.get("timeout_seconds", 10))
    timeout_seconds = max(1.0, min(timeout_seconds, 30.0))

    max_chars = int(args.get("max_chars", 3000))
    max_chars = max(200, min(max_chars, 10000))

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MinimalAgent/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", "unknown")
            content_type = resp.headers.get("Content-Type", "")
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read(max_chars * 4)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch URL: {exc}") from exc

    text = raw.decode(charset, errors="replace")

    if "html" in content_type.lower():
        # Basic HTML cleanup to make model/tool output easier to consume.
        text = re.sub(r"<script[\\s\\S]*?</script>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\\s+", " ", text).strip()
    snippet = text[:max_chars] if text else "(empty response body)"
    return f"URL: {url}\\nStatus: {status}\\nContent-Type: {content_type}\\nBody: {snippet}"


def tavily_search_tool(args: dict) -> str:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    max_results = int(args.get("max_results", 5))
    max_results = max(1, min(max_results, 10))

    search_depth = str(args.get("search_depth", "basic")).strip().lower()
    if search_depth not in {"basic", "advanced"}:
        raise ValueError("search_depth must be 'basic' or 'advanced'")

    include_answer = bool(args.get("include_answer", True))
    include_raw_content = bool(args.get("include_raw_content", False))
    proxy_url = str(
        args.get("proxy_url")
        or os.getenv("TAVILY_PROXY_URL")
        or "http://child-prc.intel.com:913"
    ).strip()

    payload = {
        "api_key": api_key,
        "query": query.strip(),
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "MinimalAgent/1.0",
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
        },
    )

    try:
        if proxy_url:
            proxy_handler = urllib.request.ProxyHandler(
                {
                    "http": proxy_url,
                    "https": proxy_url,
                }
            )
            opener = urllib.request.build_opener(proxy_handler)
            with opener.open(request, timeout=20) as resp:
                body = resp.read().decode("utf-8")
        else:
            with urllib.request.urlopen(request, timeout=20) as resp:
                body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Tavily HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        if proxy_url:
            raise RuntimeError(f"Tavily connection error via proxy {proxy_url}: {exc}") from exc
        raise RuntimeError(f"Tavily connection error: {exc}") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Tavily response: {body}") from exc

    answer = str(result.get("answer") or "").strip()
    results = result.get("results") or []
    if not isinstance(results, list):
        results = []

    lines: list[str] = []
    if answer:
        lines.append(f"Answer: {answer}")

    if not results:
        lines.append("Results: (none)")
        return "\n".join(lines)

    lines.append("Results:")
    for idx, item in enumerate(results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "(untitled)").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        snippet = re.sub(r"\s+", " ", content)[:500] if content else ""
        lines.append(f"{idx}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")

    return "\n".join(lines)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Add two numbers and return the sum.",
            handler=add_tool,
            parameters_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        )
    )
    registry.register(
        Tool(
            name="now",
            description="Get current local time in ISO format.",
            handler=now_tool,
            parameters_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
    )
    registry.register(
        Tool(
            name="current_time",
            description="Get current local date/time, timezone and unix timestamp.",
            handler=current_time_tool,
            parameters_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
    )
    registry.register(
        Tool(
            name="fetch_url",
            description="Fetch public web content from a URL and return cleaned text.",
            handler=fetch_url_tool,
            parameters_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target URL, must start with http:// or https://",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Optional network timeout, defaults to 10 (1-30).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional max output length, defaults to 3000 (200-10000).",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        )
    )
    registry.register(
        Tool(
            name="tavily_search",
            description="Search the web with Tavily and return a concise answer plus top results.",
            handler=tavily_search_tool,
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for web results.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Optional number of results to return, defaults to 5 (1-10).",
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Optional Tavily search depth, defaults to basic.",
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "Whether Tavily should return a synthesized answer, defaults to true.",
                    },
                    "include_raw_content": {
                        "type": "boolean",
                        "description": "Whether Tavily should include raw page content, defaults to false.",
                    },
                    "proxy_url": {
                        "type": "string",
                        "description": "Optional proxy URL. Defaults to env TAVILY_PROXY_URL or http://child-prc.intel.com:913.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
    )
    return registry
