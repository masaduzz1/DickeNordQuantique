# TMS Optimization (ECD+R decomposition of two-mode squeezing)

`TMS_optimization_v3.ipynb` numerically searches for the parameters of an
**ECD + qubit-rotation (ECD+R)** circuit that reproduces the **two-mode
squeezing (TMS) gate**

$$
S_{AB}(r) = \exp\bigl[r\,(a_A^\dagger a_B^\dagger - a_A a_B)\bigr]
$$

on two bosonic cavity modes `A` and `B`, using a single dispersively coupled
two-level ancilla ("qubit") to mediate the non-Gaussian control. This is the
standard hybrid CV–DV (continuous-variable / discrete-variable) setting used
in circuit-QED experiments.

The ansatz stacks `Nlayers` repetitions of:

1. **ECD on mode A**: `ECD_A(betaA)` — echoed conditional displacement,
   displaces mode A by `+betaA/2` or `-betaA/2` conditioned on the qubit
   being in `|g>` or `|e>`.
2. **ECD on mode B**: `ECD_B(betaB)` — same, applied to mode B.
3. **Qubit rotation**: `R(theta, phi)` — a rotation of the qubit about an
   axis in the XY-plane.

Unlike the earlier heralded (`v2`) version which only kept the
qubit-measured-`|g>` branch, `v3` uses the full ancilla-traced channel (both
Kraus operators `U_gg` and `U_eg`) and optimizes the **entanglement (process)
fidelity** of that channel to the target unitary. This makes the channel
CPTP (no post-selection) and makes the optimization landscape more
symmetric/robust to initialization.

## Inputs / parameters

Set in the "Params" cell:

| Parameter  | Meaning                                                        |
|------------|-----------------------------------------------------------------|
| `cutoffA`  | Fock-space truncation (Hilbert space dimension) for mode A      |
| `cutoffB`  | Fock-space truncation for mode B                                |
| `Nlayers`  | Number of ECD+R layers in the variational ansatz                |
| `r_target` | Squeezing parameter `r` of the target TMS gate `S_AB(r)`        |

Each layer contributes 6 real variational parameters (optimized numerically,
not set by hand): `Re(betaA)`, `Im(betaA)`, `Re(betaB)`, `Im(betaB)`,
`theta`, `phi` — so the optimizer runs over `Nlayers * 6` parameters. The
initial guess is drawn from a seeded random normal distribution
(`rng.default_rng(seed=0)`), and optimization uses `scipy.optimize.minimize`
with the BFGS method (`maxiter=500`).

## Output

For the optimized parameter vector, the notebook prints:

- **`F_e`** — final entanglement/process fidelity of the compiled channel to
  the target TMS unitary (1.0 = exact match).
- **Branch weights** `w0`, `w1` — average probability the ancilla ends in
  `|g>` vs `|e>` (should converge to `(1, 0)` or `(0, 1)` for a good,
  deterministic solution; `w0 + w1 = 1` always).
- **Per-layer circuit parameters** — `betaA`, `betaB` (complex displacement
  amplitudes) and `theta`, `phi` (qubit rotation angle/axis) for each of the
  `Nlayers` layers, i.e. the compiled pulse sequence that implements the TMS
  gate.

## Requirements

- Python 3
- `numpy`, `scipy`

## Note

There is a stray `$` character in the `Nlayers = 6 $` line of the "Params"
cell — remove it before running, or it will raise a `SyntaxError`.
