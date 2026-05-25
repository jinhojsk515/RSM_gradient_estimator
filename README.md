# Reward Score Matching: Value Gradient Experiments in a Toy Diffusion Model

This directory contains three Python scripts for studying value-gradient estimation in a
two-dimensional diffusion model. The experiments use a simple Gaussian-mixture setting
where the reference distribution, the reward-tilted target distribution, and the
ground-truth score difference can be written down analytically. This makes it possible
to compare approximate value-gradient estimators against a known target.

The three main scripts are:

- `value_gradient_estimate.py`
- `value_gradient_lookahead.py`
- `value_gradient_branch.py`

All three scripts assume CUDA is available and load pretrained checkpoints from
`model_base.pth` and `model_ft.pth`.



---
## Shared Setup

The scripts define and reuse a small DDPM-style diffusion model over 2D points.

- `Model` wraps a time-conditioned MLP that predicts diffusion noise.
- `DiffusionProcess` implements reverse sampling from an intermediate noisy state.
- `GaussianDataset` samples from a three-component Gaussian mixture.
- `reward_function(x)` is a linear reward, `-x[:, 0] / 2 + 3`.

The base distribution is an evenly weighted mixture with component centers arranged as
a triangle. The reward-tilted target distribution shifts the mixture in the negative x-direction
and reweights the modes according to the linear reward. The scripts compare gradients
at noisy intermediate points `x_t`, using the analytic quantity

```text
grad_x_t log p_tilted(x_t) - grad_x_t log p_reference(x_t)
```

as the ground-truth value gradient. The experiments focus on a fixed diffusion time
near `alpha_bar_t = 0.01`, using `T = 500`, `beta_1 = 1e-4`, `beta_T = 0.03`, and a
sampling interval of 10 reverse steps.



---
## value_gradient_estimate.py

This compares several ways to estimate the value gradient at an intermediate noisy data $x_t$.
The experiment compares:

- **Zeroth-order estimator(w/o control variate)**: samples many reverse trajectories from the same
  `x_t`, evaluates the final reward, and estimates the gradient with the first reverse
  sampling noise.
- **Zeroth-order estimator(w/ control variate)**: the same Monte Carlo estimator, but with a
  reward baseline/control variate obtained by subtracting the batch mean reward.
- **DPS-style first-order estimator**: predicts `x_0` directly from `x_t` and
  backpropagates the reward through that prediction.
- **SQDF-2 first-order estimator**: performs a two-step deterministic lookahead before
  backpropagating the reward.

The Monte Carlo estimators are evaluated for sample counts `2^4` through `2^16`.
The script averages RMSE over 50 fixed points and saves the plot as:

```text
value_gradient_estimate_plot.png
```

The main question of this experiment is how zeroth-order reward-weighted estimators,
with and without a control variate, compare to direct first-order approximations when
the true value gradient is known.


---
## value_gradient_lookahead.py

This script studies whether stopping the reverse process early and evaluating a
lookahead prediction can improve value-gradient estimation.

It uses the same fixed noisy states and analytic ground truth as
`value_gradient_estimate.py`, but introduces a lookahead stopping time `t_j` between
the current time `t_i` and the final sample. The tested ratios are:

```text
t_j / t_i = 0.90, 0.80, 0.50
```

At each ratio, the script samples reverse trajectories from `x_t` down to the stopping
time. From the stopped state, it predicts `x_0` and evaluates the reward. Two modes are
available:

```bash
python value_gradient_lookahead.py --mode first
python value_gradient_lookahead.py --mode zeroth
```

- In **first** mode, the estimator backpropagates through the partial reverse process
  and the final reward prediction.
- In **zeroth** mode, the estimator uses the reward-weighted first-step noise, like the
  GRPO-style estimator.

The script also includes comparison baselines:

- In `zeroth` mode, it plots the full-trajectory GRPO estimator with the reward control variate, corresponding to `t_j / t_i = 0.00`.
- In `first` mode, it plots the DPS-style estimator corresponding to `t_j / t_i = 1.00`.

It evaluates sample counts `2^4` through `2^16`, averages RMSE over 100 fixed points,
and saves:

```text
lookahead_first.png
lookahead_zeroth.png
```



---
## value_gradient_branch.py

This script studies two extensions of the zeroth-order estimator: branching reverse
trajectories and mixing SDE/ODE sampling.

It again uses the same toy distribution, fixed noisy points, and analytic value-gradient
ground truth. The branch experiment starts from one noisy state and shares early parts
of the reverse trajectory before splitting into multiple stochastic continuations. This
tests whether a tree of correlated samples can use computation more efficiently than
independent full trajectories.

Run modes:

```bash
python value_gradient_branch.py --mode branchgrpo
python value_gradient_branch.py --mode tempflowgrpo
```

In **branchgrpo** mode, the script evaluates several branching configurations:

```text
(#split, #branch) = (1, 1), (2, 2), (3, 2), (4, 2), (2, 4)
```

The total number of endpoints is `#branch ** #split`. The plot x-axis is adjusted by
an approximate number-of-function-evaluations factor, so the comparison is closer to a
compute-normalized RMSE comparison.

In **tempflowgrpo** mode, the script compares:

- an ordinary stochastic SDE reverse sampler, and
- an SDE/ODE mixed sampler where only the first step injects stochastic noise and later
  steps are deterministic.

For the mixed sampler, a high-sample estimate is used as a reference target, and lower
sample counts are compared against it.

The script evaluates sample counts `2^4` through `2^16`, averages over 100 fixed
points, and saves:

```text
branch_zeroth_branchgrpo.png
branch_zeroth_tempflowgrpo.png
```

