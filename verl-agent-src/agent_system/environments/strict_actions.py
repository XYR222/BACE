"""Dependency-light strict action helpers shared by environment managers."""

from __future__ import annotations

import re
from typing import Sequence


def parse_tagged_action_response(response: str):
    lowered = response.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag, start_idx + len(start_tag))
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None, False
    action_body = response[start_idx + len(start_tag):end_idx].strip()
    has_think = "<think>" in lowered and "</think>" in lowered
    has_chinese = re.search(r'[\u4e00-\u9fff]', response) is not None
    return action_body, bool(action_body) and has_think and not has_chinese


def strict_action_identity(response: str, projected_action: str,
                           format_valid: bool, environment_valid: bool):
    if not format_valid:
        return None
    if environment_valid:
        return f"valid::{projected_action}"
    raw_action_body, _ = parse_tagged_action_response(response)
    return f"invalid::{raw_action_body}"


def webshop_action_is_executable(action: str, action_pool: Sequence[str]) -> bool:
    if not isinstance(action, str):
        return False
    normalized = action.strip().lower()
    normalized_pool = {str(candidate).strip().lower() for candidate in action_pool}
    search_match = re.fullmatch(r"search\[(.+)\]", normalized, flags=re.DOTALL)
    if search_match is not None:
        query = search_match.group(1).strip()
        return bool(query) and "search[<your query>]" in normalized_pool
    click_match = re.fullmatch(r"click\[(.*)\]", normalized, flags=re.DOTALL)
    return click_match is not None and normalized in normalized_pool
