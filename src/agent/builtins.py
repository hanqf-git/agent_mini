from __future__ import annotations

from datetime import datetime
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
    return registry
