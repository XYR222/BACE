"""Dependency-light strict action helpers shared by environment managers."""

from __future__ import annotations

import re
from typing import Sequence


_WEBSHOP_ACTION_PATTERN = re.compile(r"(.+)\[(.+)\]")


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


def parse_webshop_environment_action(action: str):
    """Mirror WebShop's ``engine.parse_action`` without importing the server.

    The upstream parser intentionally uses ``re.match`` rather than a full
    match.  BACE must preserve that behavior because the parsed prefix, not a
    stricter researcher-defined grammar, determines what WebShop executes.
    """
    if not isinstance(action, str):
        return action, None
    match = _WEBSHOP_ACTION_PATTERN.match(action)
    if match is None:
        return action, None
    return match.groups()


def canonical_webshop_environment_action(action: str) -> str | None:
    """Return the parser-level action actually dispatched by WebShop."""
    action_name, action_arg = parse_webshop_environment_action(action)
    if action_arg is None:
        return None
    action_arg = action_arg.lower()
    if action_name == "search" and action_arg != "":
        return f"search[{action_arg}]"
    if action_name == "click" and action_arg != "search":
        return f"click[{action_arg}]"
    return None


def webshop_action_is_executable(action: str, action_pool: Sequence[str]) -> bool:
    """Match WebShop's parser and dispatch semantics for the current page.

    Search dispatch is not gated on whether the rendered page exposes a
    search bar.  Click dispatch is gated on the current ``text_to_clickable``
    keys and explicitly rejects the search button's ``"search"`` label.
    """
    canonical = canonical_webshop_environment_action(action)
    if canonical is None:
        return False
    action_name, action_arg = parse_webshop_environment_action(canonical)
    if action_name == "search":
        return True
    clickable_args = {
        parsed_arg.lower()
        for candidate in action_pool
        for parsed_name, parsed_arg in [
            parse_webshop_environment_action(str(candidate))
        ]
        if parsed_name == "click" and parsed_arg is not None
    }
    return action_name == "click" and action_arg in clickable_args
