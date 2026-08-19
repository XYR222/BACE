from dataclasses import replace

from agent_system.environments.env_package.alfworld.projection import (
    alfworld_action_identity,
    alfworld_projection,
)
from recipe.bace_gigpo.anchor_index import AnchorIndex
from recipe.bace_gigpo.replay.validator import ReplayCategory, ReplayValidator
from recipe.bace_gigpo.root_store import resolve_action_identity
from tests.bace_gigpo.test_anchor_and_replay import event, request, root


def invalid_event(occurrence_id, action):
    return replace(
        event(occurrence_id, 1, {"room": "kitchen"}, action),
        canonical_action=f"invalid::{action}",
        action_identity=f"invalid::{action}",
        action_identity_kind="invalid",
        action_format_valid=True,
        action_environment_valid=False,
        admissible_actions=("open fridge", "go to table"),
        post_action_observation="Nothing happens.",
    )


def valid_event(occurrence_id, action):
    return replace(
        event(occurrence_id, 1, {"room": "kitchen"}, action),
        canonical_action=f"valid::{action}",
        action_identity=f"valid::{action}",
        action_identity_kind="valid",
        action_format_valid=True,
        action_environment_valid=True,
        admissible_actions=("open fridge", "go to table"),
    )


def root_with_event(root_id, selected_event):
    return replace(root(root_id, "open fridge"), events=(event(f"{root_id}:0", 0, "initial", "look"), selected_event))


def test_projection_preserves_raw_response_and_separates_format_from_environment_validity():
    raw = "<think>try another cabinet</think><action>Go To Cabinet 5</action>"
    responses = [raw]
    projected, format_valid = alfworld_projection(responses, [["go to cabinet 1"]])

    assert responses == [raw]
    assert projected == ["go to cabinet 5"]
    assert format_valid == [1]
    assert projected[0] not in ["go to cabinet 1"]
    assert alfworld_action_identity(raw, projected[0], True, False) == "invalid::Go To Cabinet 5"
    assert alfworld_action_identity(raw, projected[0], True, True) == "valid::go to cabinet 5"


def test_missing_environment_specific_identity_uses_strict_fallback():
    assert resolve_action_identity(None, "click[item]", True, True) == "valid::click[item]"
    assert resolve_action_identity(None, "bad action", True, False) == "invalid::bad action"
    assert resolve_action_identity(None, "unparsed", False, False) is None


def test_strict_identity_retains_invalid_actions_and_ablation_modes_are_explicit():
    roots = [
        root_with_event("r1", invalid_event("r1:1", "go to cabinet 5")),
        root_with_event("r2", invalid_event("r2:1", "take mug from fridge")),
        root_with_event("r3", valid_event("r3:1", "open fridge")),
    ]

    strict = AnchorIndex(roots, invalid_action_mode="strict_identity").anchors_for_task("task-1")[0]
    assert strict.observed_action_ids == [
        "invalid::go to cabinet 5",
        "invalid::take mug from fridge",
        "valid::open fridge",
    ]
    assert AnchorIndex(roots, invalid_action_mode="valid_only_branch").anchors_for_task("task-1") == []
    bucket = AnchorIndex(roots, invalid_action_mode="single_invalid_bucket").anchors_for_task("task-1")[0]
    assert bucket.observed_action_ids == ["invalid::<INVALID_BUCKET>", "valid::open fridge"]
    assert len(bucket.origins_by_action["invalid::<INVALID_BUCKET>"]) == 2


def test_invalid_action_is_pre_replay_validated_without_admissible_membership():
    req = replace(
        request(),
        selected_canonical_action="invalid::go to cabinet 5",
        copied_parsed_environment_action="go to cabinet 5",
        copied_action_identity="invalid::go to cabinet 5",
        copied_action_identity_kind="invalid",
        copied_action_environment_valid=False,
    )
    result = ReplayValidator().validate(
        req, req.expected_observation, req.expected_action_set, done=False
    )
    assert result.category == ReplayCategory.VALIDATED.value


def test_invalid_action_requires_reproducible_transition():
    req = replace(
        request(),
        copied_action_identity="invalid::go to cabinet 5",
        copied_action_identity_kind="invalid",
        copied_action_environment_valid=False,
        expected_post_action_observation="Nothing happens.",
        expected_immediate_reward=0.0,
        expected_post_action_done=False,
    )
    validator = ReplayValidator()
    valid = validator.validate_transition(
        req, "Nothing happens.", 0.0, False, "invalid::go to cabinet 5"
    )
    mismatch = validator.validate_transition(
        req, "You arrive at cabinet 1.", 0.0, False, "invalid::go to cabinet 5"
    )

    assert valid.category == ReplayCategory.VALIDATED.value
    assert mismatch.category == ReplayCategory.TRANSITION_OBSERVATION_MISMATCH.value
