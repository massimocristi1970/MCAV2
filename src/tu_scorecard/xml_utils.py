from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional


def strip_ns(tag: str) -> str:
    """Remove namespace from a tag: {ns}Tag -> Tag"""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def find_first_by_local(root: ET.Element, local_path: str) -> Optional[ET.Element]:
    """
    Find first element by a slash-delimited local-name path, ignoring namespaces.
    Example: "Body/Search07aResponse/SearchResult/jobdetails/searchdate"
    """
    parts = [p for p in local_path.strip("/").split("/") if p]
    node = root
    for part in parts:
        found = None
        for child in list(node):
            if strip_ns(child.tag) == part:
                found = child
                break
        if found is None:
            return None
        node = found
    return node


def find_all_by_local(root: ET.Element, local_path: str) -> list[ET.Element]:
    """
    Find all elements matching the final local-name under the given local path.
    Only supports "fixed depth" paths (no wildcards), ignoring namespaces.
    """
    parts = [p for p in local_path.strip("/").split("/") if p]
    if not parts:
        return []
    *prefix, last = parts

    node = root
    for part in prefix:
        found = None
        for child in list(node):
            if strip_ns(child.tag) == part:
                found = child
                break
        if found is None:
            return []
        node = found

    return [c for c in list(node) if strip_ns(c.tag) == last]


def get_text(el: Optional[ET.Element], default: str = "") -> str:
    if el is None or el.text is None:
        return default
    return el.text.strip()


def to_int(s: str, default: int = 0) -> int:
    try:
        return int(float(s))
    except Exception:
        return default


def to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default
