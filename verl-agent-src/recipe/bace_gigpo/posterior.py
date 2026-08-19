from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BetaPosterior:
    alpha: float
    beta: float
    natural_successes: int = 0
    natural_failures: int = 0
    branch_successes: int = 0
    branch_failures: int = 0

    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta posterior parameters must be positive")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence(self) -> int:
        return self.natural_successes + self.natural_failures + self.branch_successes + self.branch_failures

    def update_branch(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
            self.branch_successes += 1
        else:
            self.beta += 1.0
            self.branch_failures += 1


@dataclass(frozen=True)
class AnchorAcquisition:
    regret: float
    erv_by_action: dict[str, float]
    probability_by_action: dict[str, float]
    utility: float


class PosteriorEngine:
    """Pure-CPU Beta-Bernoulli Bayes regret and one-sample ERV engine."""

    def __init__(self, mc_samples: int = 512, temperature: float = 0.05, seed: int = 0):
        if mc_samples < 1:
            raise ValueError("mc_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.mc_samples = mc_samples
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def evaluate(self, posteriors: dict[str, BetaPosterior]) -> AnchorAcquisition:
        if len(posteriors) < 2:
            raise ValueError("ERV acquisition requires at least two observed actions")
        actions = sorted(posteriors)
        alpha = np.array([posteriors[action].alpha for action in actions], dtype=np.float64)
        beta = np.array([posteriors[action].beta for action in actions], dtype=np.float64)
        means = alpha / (alpha + beta)

        samples = self.rng.beta(alpha, beta, size=(self.mc_samples, len(actions)))
        regret = max(0.0, float(samples.max(axis=1).mean() - means.max()))

        current_best = float(means.max())
        erv = np.empty(len(actions), dtype=np.float64)
        for action_idx, predictive_success in enumerate(means):
            success_means = means.copy()
            failure_means = means.copy()
            success_means[action_idx] = (alpha[action_idx] + 1.0) / (alpha[action_idx] + beta[action_idx] + 1.0)
            failure_means[action_idx] = alpha[action_idx] / (alpha[action_idx] + beta[action_idx] + 1.0)
            expected_next_best = (
                predictive_success * success_means.max()
                + (1.0 - predictive_success) * failure_means.max()
            )
            erv[action_idx] = max(0.0, float(expected_next_best - current_best))

        logits = erv / self.temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        utility = float(np.dot(probabilities, erv))
        return AnchorAcquisition(
            regret=regret,
            erv_by_action=dict(zip(actions, erv.tolist())),
            probability_by_action=dict(zip(actions, probabilities.tolist())),
            utility=utility,
        )


def initialize_local_posterior(
    prior_mean: float,
    prior_strength: float,
    outcomes: list[bool],
) -> BetaPosterior:
    if not 0 < prior_mean < 1:
        raise ValueError("prior_mean must be strictly between zero and one")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    successes = sum(bool(outcome) for outcome in outcomes)
    failures = len(outcomes) - successes
    return BetaPosterior(
        alpha=prior_strength * prior_mean + successes,
        beta=prior_strength * (1.0 - prior_mean) + failures,
        natural_successes=successes,
        natural_failures=failures,
    )
