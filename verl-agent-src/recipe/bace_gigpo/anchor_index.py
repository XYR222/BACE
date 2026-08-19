from __future__ import annotations

import hashlib
from collections import defaultdict

from gigpo.core_gigpo import to_hashable

from .types import AnchorRecord, OriginOccurrence, RootEvent, RootEventLog


class AnchorIndex:
    """Exact-observation anchor index built only from frozen natural roots."""

    def __init__(self, roots: list[RootEventLog], invalid_action_mode: str = "strict_identity"):
        if invalid_action_mode not in {"strict_identity", "valid_only_branch", "single_invalid_bucket"}:
            raise ValueError(f"Unknown invalid action mode: {invalid_action_mode}")
        self.invalid_action_mode = invalid_action_mode
        self.roots = {root.root_id: root for root in roots}
        self.events = {
            event.occurrence_id: event
            for root in roots
            for event in root.events
        }
        self._anchors = self._build(roots)

    def _statistical_action_id(self, event: RootEvent) -> str | None:
        if not event.action_format_valid:
            return None
        identity = event.action_identity or event.canonical_action
        if not identity or identity == "INVALID":
            return None
        if event.action_environment_valid is False:
            if self.invalid_action_mode == "valid_only_branch":
                return None
            if self.invalid_action_mode == "single_invalid_bucket":
                return "invalid::<INVALID_BUCKET>"
        return identity

    @staticmethod
    def _anchor_id(task_id: str, anchor_key) -> str:
        payload = repr((task_id, to_hashable(anchor_key))).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    def _build(self, roots: list[RootEventLog]) -> dict[str, AnchorRecord]:
        grouped: dict[tuple[str, object], list[tuple[RootEventLog, RootEvent]]] = defaultdict(list)
        for root in roots:
            for event in root.events:
                if event.step_index == 0:
                    continue
                if self._statistical_action_id(event) is None:
                    continue
                grouped[(root.task_id, to_hashable(event.pre_action_observation))].append((root, event))

        anchors = {}
        for (task_id, anchor_key), occurrences in grouped.items():
            actions = sorted({self._statistical_action_id(event) for _, event in occurrences})
            if len(occurrences) < 2 or len(actions) < 2:
                continue
            anchor_id = self._anchor_id(task_id, anchor_key)
            record = AnchorRecord(
                task_id=task_id,
                anchor_id=anchor_id,
                anchor_key=anchor_key,
                occurrence_ids=[event.occurrence_id for _, event in occurrences],
                observed_action_ids=actions,
            )
            for root, event in occurrences:
                action_id = self._statistical_action_id(event)
                origin = OriginOccurrence(
                    occurrence_id=event.occurrence_id,
                    task_id=task_id,
                    root_id=root.root_id,
                    step_index=event.step_index,
                    anchor_id=anchor_id,
                    action_id=action_id,
                    environment_reset_key=root.environment_reset_key,
                    remaining_horizon=event.remaining_horizon,
                )
                record.origins_by_action.setdefault(action_id, []).append(origin)
            anchors[anchor_id] = record
        return anchors

    def anchors_for_task(self, task_id: str) -> list[AnchorRecord]:
        return [record for record in self._anchors.values() if record.task_id == task_id]

    def root_for(self, root_id: str) -> RootEventLog:
        return self.roots[root_id]

    def event_for(self, occurrence_id: str) -> RootEvent:
        return self.events[occurrence_id]
