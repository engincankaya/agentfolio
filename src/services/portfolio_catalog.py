from pathlib import Path


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse simple YAML-like frontmatter from a markdown document."""
    if not content.startswith("---\n"):
        return {}, content

    lines = content.splitlines()
    if not lines or lines[0] != "---":
        return {}, content

    metadata: dict[str, object] = {}
    closing_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_index = idx
            break

        line = lines[idx].strip()
        if not line or ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            metadata[key] = items
        elif value.lower() in {"true", "false"}:
            metadata[key] = value.lower() == "true"
        else:
            metadata[key] = value.strip("'\"")

    if closing_index is None:
        return {}, content

    body = "\n".join(lines[closing_index + 1 :]).lstrip()
    return metadata, body


def summarize_text(content: str, limit: int = 220) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def load_private_portfolio_catalog(raw_path: str) -> list[dict]:
    """Build a lightweight private project/company catalog from raw markdown files."""
    catalog: list[dict] = []

    for file_path in sorted(Path(raw_path).glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(content)
        summary = metadata.get("summary") or summarize_text(body)

        entry = {
            "project_name": metadata.get("project_name") or file_path.stem.replace("-", " ").title(),
            "company_name": metadata.get("company_name") or metadata.get("company") or "",
            "visibility": metadata.get("visibility") or "private",
            "kind": metadata.get("kind") or "project",
            "tags": metadata.get("tags") or [],
            "summary": summary,
            "filename": file_path.name,
        }
        catalog.append(entry)

    return catalog
