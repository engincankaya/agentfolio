import ast
import json


def coerce_to_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    stripped = value.strip()
    if not stripped:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    return {}


def normalize_github_user_context(raw_context) -> dict:
    """Normalize GitHub MCP get_me output into a flat dict with a login field."""
    if isinstance(raw_context, dict) and "login" in raw_context:
        return raw_context

    if isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, dict):
                if "login" in item:
                    return item
                nested = coerce_to_dict(item.get("text"))
                if nested.get("login"):
                    return nested
        return {}

    if isinstance(raw_context, dict):
        nested = coerce_to_dict(raw_context.get("text"))
        if nested.get("login"):
            return nested

    return coerce_to_dict(raw_context)


def extract_repository_items(raw_result) -> list[dict]:
    """Normalize search_repositories output into a plain list of repo dicts."""
    if isinstance(raw_result, str):
        parsed = coerce_to_dict(raw_result)
        if parsed:
            return extract_repository_items(parsed)
        return []

    if isinstance(raw_result, list):
        if all(isinstance(item, dict) and any(key in item for key in ("id", "text", "type")) for item in raw_result):
            for block in raw_result:
                parsed = coerce_to_dict(block.get("text"))
                items = extract_repository_items(parsed)
                if items:
                    return items
            return []

        if all(isinstance(item, dict) and ("name" in item or "full_name" in item) for item in raw_result):
            return [item for item in raw_result if isinstance(item, dict)]

        for item in raw_result:
            if not isinstance(item, dict):
                continue
            items = extract_repository_items(item)
            if items:
                return items
        return []

    if not isinstance(raw_result, dict):
        return []

    for key in ("items", "repositories", "results"):
        if isinstance(raw_result.get(key), list):
            return [item for item in raw_result[key] if isinstance(item, dict)]

    output_blocks = raw_result.get("output")
    if isinstance(output_blocks, list):
        for block in output_blocks:
            if not isinstance(block, dict):
                continue
            parsed = coerce_to_dict(block.get("text"))
            for key in ("items", "repositories", "results"):
                if isinstance(parsed.get(key), list):
                    return [item for item in parsed[key] if isinstance(item, dict)]

    parsed_result = coerce_to_dict(raw_result.get("text"))
    for key in ("items", "repositories", "results"):
        if isinstance(parsed_result.get(key), list):
            return [item for item in parsed_result[key] if isinstance(item, dict)]

    return []
