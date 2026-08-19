from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.special import betainc


@dataclass(frozen=True)
class CompetencePrior:
    mean: float
    strength: float

    @property
    def alpha(self) -> float:
        return self.mean * self.strength

    @property
    def beta(self) -> float:
        return (1.0 - self.mean) * self.strength


@dataclass(frozen=True)
class CompetencePosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def probability_above(self, threshold: float) -> float:
        if not 0 <= threshold <= 1:
            raise ValueError("competence threshold must lie in [0, 1]")
        return float(1.0 - betainc(self.alpha, self.beta, threshold))


@dataclass
class FamilyHistory:
    decayed_successes: float = 0.0
    decayed_failures: float = 0.0


class CompetenceHistory:
    """Lagged task-family history updated from natural roots only."""

    STATE_VERSION = 1

    def __init__(
        self,
        base_alpha: float = 1.0,
        base_beta: float = 1.0,
        forgetting: float = 0.9,
        transfer_fraction: float = 0.1,
        min_strength: float = 2.0,
        max_strength: float = 8.0,
    ):
        if base_alpha <= 0 or base_beta <= 0:
            raise ValueError("base competence prior parameters must be positive")
        if not 0 <= forgetting <= 1:
            raise ValueError("history forgetting must lie in [0, 1]")
        if transfer_fraction <= 0 or not 0 < min_strength <= max_strength:
            raise ValueError("invalid competence transfer strengths")
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.forgetting = forgetting
        self.transfer_fraction = transfer_fraction
        self.min_strength = min_strength
        self.max_strength = max_strength
        self._history: dict[str, FamilyHistory] = {}

    def prior(self, task_family: str) -> CompetencePrior:
        history = self._history.get(task_family, FamilyHistory())
        alpha = self.base_alpha + history.decayed_successes
        beta = self.base_beta + history.decayed_failures
        concentration = alpha + beta
        strength = min(
            self.max_strength,
            max(self.min_strength, self.transfer_fraction * concentration),
        )
        return CompetencePrior(mean=alpha / concentration, strength=strength)

    @staticmethod
    def posterior(prior: CompetencePrior, pilot_outcomes: list[bool]) -> CompetencePosterior:
        successes = sum(bool(outcome) for outcome in pilot_outcomes)
        return CompetencePosterior(
            alpha=prior.alpha + successes,
            beta=prior.beta + len(pilot_outcomes) - successes,
        )

    @staticmethod
    def planned_branch_quota(total_budget: int, pilot_count: int, readiness: float) -> int:
        if not 0 < pilot_count <= total_budget:
            raise ValueError("pilot count must be in [1, total_budget]")
        if not 0 <= readiness <= 1:
            raise ValueError("readiness must lie in [0, 1]")
        return int(math.floor((total_budget - pilot_count) * readiness + 0.5))

    def update(self, natural_outcomes_by_family: dict[str, list[bool]]) -> None:
        families = set(self._history).union(natural_outcomes_by_family)
        for task_family in families:
            history = self._history.setdefault(task_family, FamilyHistory())
            outcomes = natural_outcomes_by_family.get(task_family, [])
            successes = sum(bool(outcome) for outcome in outcomes)
            failures = len(outcomes) - successes
            history.decayed_successes = self.forgetting * history.decayed_successes + successes
            history.decayed_failures = self.forgetting * history.decayed_failures + failures

    def snapshot(self) -> dict[str, tuple[float, float]]:
        return {
            family: (history.decayed_successes, history.decayed_failures)
            for family, history in self._history.items()
        }

    def _parameter_signature(self) -> dict[str, float]:
        return {
            "base_alpha": self.base_alpha,
            "base_beta": self.base_beta,
            "forgetting": self.forgetting,
            "transfer_fraction": self.transfer_fraction,
            "min_strength": self.min_strength,
            "max_strength": self.max_strength,
        }

    def state_dict(self) -> dict:
        """Return a versioned, JSON-serializable controller state."""
        return {
            "version": self.STATE_VERSION,
            "parameters": self._parameter_signature(),
            "families": {
                family: {
                    "decayed_successes": history.decayed_successes,
                    "decayed_failures": history.decayed_failures,
                }
                for family, history in sorted(self._history.items())
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore controller history, rejecting incompatible or corrupt state."""
        if not isinstance(state, dict):
            raise ValueError("competence history state must be a mapping")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(
                f"unsupported competence history state version: {state.get('version')!r}"
            )
        parameters = state.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("competence history state is missing parameters")
        expected = self._parameter_signature()
        for name, expected_value in expected.items():
            try:
                actual_value = float(parameters[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid competence history parameter: {name}") from exc
            if not math.isfinite(actual_value) or not math.isclose(
                actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"competence history parameter mismatch for {name}: "
                    f"checkpoint={actual_value}, current={expected_value}"
                )

        families = state.get("families")
        if not isinstance(families, dict):
            raise ValueError("competence history state is missing families")
        restored = {}
        for family, values in families.items():
            if not isinstance(family, str) or not family:
                raise ValueError("competence history family names must be non-empty strings")
            if not isinstance(values, dict):
                raise ValueError(f"invalid competence history entry for {family}")
            try:
                successes = float(values["decayed_successes"])
                failures = float(values["decayed_failures"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid competence history counts for {family}") from exc
            if (
                not math.isfinite(successes)
                or not math.isfinite(failures)
                or successes < 0
                or failures < 0
            ):
                raise ValueError(f"invalid competence history counts for {family}")
            restored[family] = FamilyHistory(successes, failures)
        self._history = restored
