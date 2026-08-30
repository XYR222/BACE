# Pairwise Threshold Sensitivity（Step 1）

## Result

This exact CPU-only audit evaluates `4520` task-threshold states from the completed 150-step BACE trace.  It recomputes local designs, threshold capacity, P2 K=2 selection, Beta-Binomial outcomes, C=1 single-branch execution, and C=0 one-way fallback for every threshold.

Recommended offline candidates under the mechanical retention screen are:

- conservative `tau_low* = 0.005` (retain >=97%);
- aggressive `tau_high* = 0.0075` (retain >=90%).

These are candidates for Step-100 continuation controls, not claims about validation improvement.

## Overall threshold table

|   threshold |   initial_pair_feasible |   initial_full_feasible |   eager_capacity_gap |   eager_correction_rounds_lower_bound |   pairwise_replan_changed_realized |   post_pair_zero_capacity_probability |   post_pair_shortfall_probability |   expected_executed_branches |   expected_unexecuted_branches |   expected_fallback_roots |   expected_pair_rounds |   p2_expected_berv_value |   retained_berv_value |   weak_branch_ratio |   weak_value_ratio |
|------------:|------------------------:|------------------------:|---------------------:|--------------------------------------:|-----------------------------------:|--------------------------------------:|----------------------------------:|-----------------------------:|-------------------------------:|--------------------------:|-----------------------:|-------------------------:|----------------------:|--------------------:|-------------------:|
|      0.005  |                1        |                1        |             0        |                              0        |                           0.455752 |                           0.000273526 |                         0.0106023 |                      4.69072 |                      0.0145933 |                 0.0145933 |                2.61892 |                 0.144718 |              1.01685  |            0        |           0        |
|      0.0075 |                0.953097 |                0.79469  |             0.439823 |                              0.439823 |                           0.455752 |                           0.0856912   |                         0.192332  |                      4.29691 |                      0.408399  |                 0.408399  |                2.41933 |                 0.142217 |              0.947561 |            0.230528 |           0.140043 |
|      0.01   |                0.913274 |                0.70708  |             0.758407 |                              0.758407 |                           0.455752 |                           0.141192    |                         0.284122  |                      3.98731 |                      0.718002  |                 0.718002  |                2.26504 |                 0.139502 |              0.895398 |            0.346603 |           0.220555 |
|      0.015  |                0.762832 |                0.514159 |             1.60531  |                              1.60531  |                           0.455752 |                           0.313316    |                         0.457565  |                      3.16532 |                      1.53999   |                 1.53999   |                1.8217  |                 0.129318 |              0.742954 |            0.600772 |           0.430359 |

## Important boundary

`eager_correction_rounds_lower_bound = max(Q-C_tau, 0)` is an exact lower bound on the number of slot conversions an eager Full-Batch controller must make after the observed root support.  Counterfactual *physical* root-wave counts above this bound require fresh roots not present in the historical trace at higher thresholds; this analysis deliberately does not invent those outcomes.  The Step 4 online P2 implementation measures actual correction/fallback waves.

## Outputs

- `threshold_task_audit.parquet`: task-level exact calculations.
- `threshold_summary.csv`, `threshold_phase_summary.csv`, `threshold_family_summary.csv`, `threshold_q_summary.csv`.
- `online_candidates.json`: mechanical candidate screen.
