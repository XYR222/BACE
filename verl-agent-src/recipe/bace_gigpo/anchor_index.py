from __future__ import annotations

import hashlib
from collections import defaultdict

from gigpo.core_gigpo import are_similar, to_hashable

from .types import AnchorRecord, OriginOccurrence, RootEvent, RootEventLog


class AnchorIndex:
    """Exact-observation anchor index built only from frozen natural roots."""

    IDENTITY_MODES = {"legacy_uuid", "stable_v1"}

    def __init__(
        self,
        roots: list[RootEventLog],
        invalid_action_mode: str = "strict_identity",
        tie_break_identity_mode: str = "legacy_uuid",
        anchor_similarity_enabled: bool = False,
        anchor_similarity_threshold: float = 0.9,
        allow_initial_search_anchor: bool = False,
    ):
        if invalid_action_mode not in {"strict_identity", "valid_only_branch", "single_invalid_bucket"}:
            raise ValueError(f"Unknown invalid action mode: {invalid_action_mode}")
        if tie_break_identity_mode not in self.IDENTITY_MODES:
            raise ValueError(
                "tie_break_identity_mode must be legacy_uuid or stable_v1"
            )
        self.invalid_action_mode = invalid_action_mode
        self.tie_break_identity_mode = tie_break_identity_mode
        self.anchor_similarity_enabled = bool(anchor_similarity_enabled)
        self.anchor_similarity_threshold = float(anchor_similarity_threshold)
        self.allow_initial_search_anchor = bool(allow_initial_search_anchor)
        if self.anchor_similarity_enabled and not 0.0 < self.anchor_similarity_threshold < 1.0:
            raise ValueError("anchor_similarity_threshold must lie in (0, 1)")
        self.roots = {root.root_id: root for root in roots}
        self.events = {
            event.occurrence_id: event
            for root in roots
            for event in root.events
        }
        self._decision_key_by_task = self._build_task_decision_keys(roots)
        self._root_decision_key_by_id = (
            {root.root_id: self._root_decision_key(root) for root in roots}
            if self.tie_break_identity_mode == "stable_v1"
            else {}
        )
        self._anchors = self._build(roots)

    def _build_task_decision_keys(self, roots: list[RootEventLog]) -> dict[str, str]:
        if self.tie_break_identity_mode == "legacy_uuid":
            return {root.task_id: root.task_id for root in roots}
        grouped: dict[str, set[tuple[int, str]]] = defaultdict(set)
        for root in roots:
            grouped[root.task_id].add(
                (int(root.task_batch_index), str(root.environment_reset_key))
            )
        result = {}
        for task_id, identities in grouped.items():
            if len(identities) != 1:
                raise ValueError(
                    f"Task {task_id} maps to multiple stable identities: "
                    f"{sorted(identities)!r}"
                )
            task_batch_index, reset_key = next(iter(identities))
            payload = repr(("stable_v1", task_batch_index, reset_key)).encode("utf-8")
            result[task_id] = hashlib.sha256(payload).hexdigest()
        return result

    @staticmethod
    def _root_decision_key(root: RootEventLog) -> str:
        """Content identity for stable origin ordering; excludes UUID lineage."""
        events = tuple(
            (
                int(event.step_index),
                to_hashable(event.pre_action_observation),
                event.action_identity or event.canonical_action,
                tuple(event.response_token_ids),
                to_hashable(event.post_action_observation),
                float(event.reward),
                bool(event.done),
            )
            for event in root.events
        )
        payload = repr((
            "stable_origin_v1",
            int(root.task_batch_index),
            str(root.environment_reset_key),
            events,
            float(root.terminal_reward),
            bool(root.won),
        )).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def decision_key_for_task(self, task_id: str) -> str:
        return self._decision_key_by_task[task_id]

    def ordered_task_ids(self) -> list[str]:
        return sorted(
            self._decision_key_by_task,
            key=lambda task_id: (self._decision_key_by_task[task_id], task_id),
        )

    def ordered_origins(self, origins: list[OriginOccurrence]) -> list[OriginOccurrence]:
        if self.tie_break_identity_mode == "legacy_uuid":
            return list(origins)
        return sorted(
            origins,
            key=lambda origin: (
                self._root_decision_key_by_id[origin.root_id],
                int(origin.step_index),
                str(origin.action_id),
            ),
        )

    def _statistical_action_id(self, event: RootEvent) -> str | None:
        if not event.action_format_valid:
            return None
        # Search ANSWER is a real terminal training action but not a repeatable
        # information-acquisition experiment.
        if event.action_identity_kind == "terminal":
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
    def _anchor_id(identity_key: str, anchor_key) -> str:
        payload = repr((identity_key, to_hashable(anchor_key))).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    def _build(self, roots: list[RootEventLog]) -> dict[str, AnchorRecord]:
        clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
        for root in roots:
            for event in root.events:
                if event.step_index == 0 and not (
                    self.allow_initial_search_anchor and root.task_family == "search"
                ):
                    continue
                if self._statistical_action_id(event) is None:
                    continue
                anchor_key = event.pre_action_observation
                cluster = None
                for candidate in clusters[root.task_id]:
                    representative = candidate["anchor_key"]
                    matches = (
                        are_similar(
                            anchor_key,
                            representative,
                            self.anchor_similarity_threshold,
                        )
                        if self.anchor_similarity_enabled
                        else to_hashable(anchor_key) == to_hashable(representative)
                    )
                    if matches:
                        cluster = candidate
                        break
                if cluster is None:
                    cluster = {"anchor_key": anchor_key, "occurrences": []}
                    clusters[root.task_id].append(cluster)
                cluster["occurrences"].append((root, event))

        anchors = {}
        for task_id, task_clusters in clusters.items():
            for cluster in task_clusters:
                anchor_key = cluster["anchor_key"]
                occurrences = cluster["occurrences"]
                actions = sorted({self._statistical_action_id(event) for _, event in occurrences})
                if len(occurrences) < 2 or len(actions) < 2:
                    continue
                anchor_id = self._anchor_id(
                    self._decision_key_by_task[task_id], anchor_key
                )
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

    def anchor_for_observation(self, task_id: str, observation):
        for anchor in self.anchors_for_task(task_id):
            if self.anchor_similarity_enabled:
                if are_similar(
                    observation, anchor.anchor_key, self.anchor_similarity_threshold
                ):
                    return anchor
            elif to_hashable(observation) == to_hashable(anchor.anchor_key):
                return anchor
        return None

    def anchors_for_task(self, task_id: str) -> list[AnchorRecord]:
        return [record for record in self._anchors.values() if record.task_id == task_id]

    def root_for(self, root_id: str) -> RootEventLog:
        return self.roots[root_id]

    def event_for(self, occurrence_id: str) -> RootEvent:
        return self.events[occurrence_id]
