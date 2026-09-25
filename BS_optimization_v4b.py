#!/usr/bin/env python3
"""Variational ECD+R compilation of the beam-splitter gate B_AB(theta).

Beam-splitter version of TMS_optimization_v4b.py: identical ansatz, cost and
layer-growth loop, with the target S_AB(r) replaced by
B_AB(theta) = exp[theta (a_A^dag a_B - a_A a_B^dag)].

The ansatz is grown one ECD+R layer at a time, warm-started from the previous
solution, until the entanglement fidelity F_e reaches the strictest requested
threshold or the layer budget runs out.  For each theta the script writes

  data/BS_theta<theta>/nlayer<N>-iteration-cost   two columns: iteration, cost
  data/BS_theta<theta>/nlayer-iteration-cost      combined: nlayer, iteration, cost
  data/BS_theta<theta>/nlayer<N>-params           gate parameters found at N layers
  data/BS_theta<theta>/optimized-params           parameters of the final circuit
  data/BS_nlayers_for_F0.99                       two columns: theta, layers needed
  data/BS_nlayers_for_F0.9999                     two columns: theta, layers needed

The two summary files are updated in place: the row for this theta is replaced
if it is already there, so re-running one theta does not duplicate or clobber
the others.
"""

import argparse
import os

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# thresholds whose "layers needed" we report, and the summary file for each
THRESHOLDS = [(0.99, "BS_nlayers_for_F0.99"), (0.9999, "BS_nlayers_for_F0.9999")]


# --------------------------------------------------------------------------
# circuit blocks  (identical to the notebook; they read the module-level
# operators set up in build_operators())
# --------------------------------------------------------------------------
def destroy(dim):
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
    return a


def displacement(alpha, mode):
    if mode == "A":
        return expm(alpha * adagA - np.conjugate(alpha) * aA)
    elif mode == "B":
        return expm(alpha * adagB - np.conjugate(alpha) * aB)
    raise ValueError("mode must be 'A' or 'B'")


def Rphi(theta, phi):
    sigma = np.cos(phi) * X + np.sin(phi) * Y
    return expm(-1j * theta * sigma / 2)


def ECD_A(beta):
    Dp = displacement(beta / 2, "A")
    Dm = displacement(-beta / 2, "A")
    return np.kron(Pg, np.kron(Dp, IB)) + np.kron(Pe, np.kron(Dm, IB))


def ECD_B(beta):
    Dp = displacement(beta / 2, "B")
    Dm = displacement(-beta / 2, "B")
    return np.kron(Pg, np.kron(IA, Dp)) + np.kron(Pe, np.kron(IA, Dm))


def embedded_rotation(theta, phi, d_cav):
    return np.kron(Rphi(theta, phi), np.eye(d_cav))


def beam_splitter(theta):
    a1 = np.kron(aA, IB)
    a2 = np.kron(IA, aB)
    ad1, ad2 = a1.conj().T, a2.conj().T
    G = ad1 @ a2 - a1 @ ad2
    return expm(theta * G)


def build_operators(cutoffA, cutoffB):
    """Populate the module-level operators the circuit blocks close over."""
    global aA, IA, adagA, aB, IB, adagB, X, Y, Pg, Pe
    aA, IA = destroy(cutoffA), np.eye(cutoffA)
    adagA = aA.conj().T
    aB, IB = destroy(cutoffB), np.eye(cutoffB)
    adagB = aB.conj().T
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Pg = np.array([[1, 0], [0, 0]], dtype=complex)
    Pe = np.array([[0, 0], [0, 1]], dtype=complex)


# --------------------------------------------------------------------------
# ansatz and cost
# --------------------------------------------------------------------------
def ansatz(params, d_cav):
    n_layers = len(params) // 6
    U = np.eye(2 * d_cav, dtype=complex)
    idx = 0
    for _ in range(n_layers):
        betaA = params[idx] + 1j * params[idx + 1]; idx += 2
        betaB = params[idx] + 1j * params[idx + 1]; idx += 2
        theta = params[idx]
        phi = params[idx + 1]; idx += 2

        U = ECD_A(betaA) @ U
        U = ECD_B(betaB) @ U
        U = embedded_rotation(theta, phi, d_cav) @ U
    return U


def kraus_ops(params, d_cav):
    """(K0, K1) = (U_gg, U_eg): the ancilla-traced channel starting from |g>."""
    U = ansatz(params, d_cav)
    return U[:d_cav, :d_cav], U[d_cav:2 * d_cav, :d_cav]


def process_fidelity(params, d_cav, V_target):
    K0, K1 = kraus_ops(params, d_cav)
    Vd = V_target.conj().T
    return (np.abs(np.trace(Vd @ K0)) ** 2
            + np.abs(np.trace(Vd @ K1)) ** 2) / (d_cav ** 2)


def cost(params, d_cav, V_target):
    return 1.0 - process_fidelity(params, d_cav, V_target)


def branch_weights(params, d_cav):
    K0, K1 = kraus_ops(params, d_cav)
    return (np.real(np.trace(K0.conj().T @ K0)) / d_cav,
            np.real(np.trace(K1.conj().T @ K1)) / d_cav)


def format_layers(params):
    """Per-layer `Layer k / betaA / betaB / theta / phi` listing, as in the notebook."""
    lines = []
    for k, (bAr, bAi, bBr, bBi, theta, phi) in enumerate(np.reshape(params, (-1, 6))):
        lines += [f"Layer {k}",
                  f"  betaA = {complex(bAr, bAi)}",
                  f"  betaB = {complex(bBr, bBi)}",
                  f"  theta = {theta}",
                  f"  phi   = {phi}",
                  ""]
    return "\n".join(lines)


def write_params(path, params, header):
    """Parameter table (one row per layer) followed by the readable listing."""
    with open(path, "w") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        fh.write("# layer   Re(betaA)   Im(betaA)   Re(betaB)   Im(betaB)"
                 "   theta   phi\n")
        for k, row in enumerate(np.reshape(params, (-1, 6))):
            fh.write(f"{k:5d}  " + "  ".join(f"{v: .15e}" for v in row) + "\n")
        fh.write("\n# " + format_layers(params).replace("\n", "\n# ") + "\n")


# --------------------------------------------------------------------------
# summary-file bookkeeping
# --------------------------------------------------------------------------
def update_summary(path, r, nlayers, r_fmt="%.3f"):
    """Insert/replace the row for this theta in a two-column `theta  nlayers` file."""
    rows = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                key, val = line.split()[:2]
                rows[float(key)] = val
    rows[float(r_fmt % r)] = "nan" if nlayers is None else str(nlayers)

    with open(path, "w") as fh:
        fh.write("# theta    nlayers   (nan = threshold not reached "
                 "within the layer budget)\n")
        for key in sorted(rows):
            fh.write(f"{r_fmt % key}   {rows[key]:>7}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--theta", type=float, required=True,
                   help="beam-splitter angle theta of the target gate B_AB(theta)")
    p.add_argument("--cutoffA", type=int, default=10, help="Fock cutoff of mode A")
    p.add_argument("--cutoffB", type=int, default=2, help="Fock cutoff of mode B")
    p.add_argument("--nlayers-start", type=int, default=1)
    p.add_argument("--nlayers-max", type=int, default=12,
                   help="layer budget; the sweep stops here even if unconverged")
    p.add_argument("--maxiter", type=int, default=500, help="BFGS iterations per stage")
    p.add_argument("--restarts", type=int, default=1,
                   help="tries per layer count: 1 = warm start only, "
                        ">1 adds that many fresh random initializations")
    p.add_argument("--tuning-factor", type=float, default=0.3,
                   help="scale of the random initial parameters")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    build_operators(args.cutoffA, args.cutoffB)
    d_cav = args.cutoffA * args.cutoffB
    V_target = beam_splitter(args.theta)

    tag = f"BS_theta{args.theta:.3f}"
    out_dir = os.path.join(args.data_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed=args.seed)
    f_strictest = max(t for t, _ in THRESHOLDS)
    reached = {t: None for t, _ in THRESHOLDS}

    print(f"=== theta = {args.theta}  (d_cav = {d_cav}, cutoffA = {args.cutoffA}, "
          f"cutoffB = {args.cutoffB}, layer budget = {args.nlayers_max}) ===")

    params = args.tuning_factor * rng.standard_normal(args.nlayers_start * 6)
    best_F = 0.0

    for Nlayers in range(args.nlayers_start, args.nlayers_max + 1):
        # candidate starting points for this layer count
        starts = [params] if len(params) == 6 * Nlayers else [
            np.concatenate([params, args.tuning_factor * rng.standard_normal(6)])]
        for _ in range(args.restarts - 1):
            starts.append(args.tuning_factor * rng.standard_normal(6 * Nlayers))

        best = None
        for x0 in starts:
            history = []

            def callback(xk):
                history.append(cost(xk, d_cav, V_target))

            res = minimize(cost, x0, args=(d_cav, V_target), method="BFGS",
                           callback=callback,
                           options={"maxiter": args.maxiter, "disp": False})
            curve = [(0, cost(x0, d_cav, V_target))]
            curve += [(i + 1, c) for i, c in enumerate(history)]
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x, curve, res)

        final_cost, params, curve, res = best
        F_e = process_fidelity(params, d_cav, V_target)
        best_F = max(best_F, F_e)

        fname = os.path.join(out_dir, f"nlayer{Nlayers}-iteration-cost")
        with open(fname, "w") as fh:
            fh.write("# iteration    cost = 1 - F_e\n")
            fh.write(f"# Nlayers = {Nlayers}, theta_target = {args.theta}, "
                     f"cutoffA = {args.cutoffA}, cutoffB = {args.cutoffB}\n")
            for it, c in curve:
                fh.write(f"{it:8d}   {c:.12e}\n")

        w0, w1 = branch_weights(params, d_cav)
        write_params(os.path.join(out_dir, f"nlayer{Nlayers}-params"), params, [
            f"Nlayers = {Nlayers}, theta_target = {args.theta}, "
            f"cutoffA = {args.cutoffA}, cutoffB = {args.cutoffB}",
            f"F_e = {F_e:.12f}, branch weights = ({w0:.6f}, {w1:.6f})",
            f"BFGS: nit = {res.nit}, |grad| = {np.linalg.norm(res.jac):.3e}, "
            f"message = {res.message}"])

        for thr, _ in THRESHOLDS:
            if reached[thr] is None and F_e >= thr:
                reached[thr] = Nlayers

        flags = " ".join(f"F>={thr:g}@{reached[thr]}" for thr, _ in THRESHOLDS
                         if reached[thr] is not None)
        print(f"  Nlayers = {Nlayers:2d}   F_e = {F_e:.8f}   "
              f"cost = {final_cost:.3e}   {flags}")

        if F_e >= f_strictest:
            break

    # combined curve dump for this theta
    combined = os.path.join(out_dir, "nlayer-iteration-cost")
    with open(combined, "w") as fh:
        fh.write("# nlayer   iteration    cost = 1 - F_e\n")
        fh.write(f"# theta_target = {args.theta}, cutoffA = {args.cutoffA}, "
                 f"cutoffB = {args.cutoffB}\n")
        for nl in range(args.nlayers_start, Nlayers + 1):
            src = os.path.join(out_dir, f"nlayer{nl}-iteration-cost")
            if not os.path.exists(src):
                continue
            for it, c in np.atleast_2d(np.loadtxt(src)):
                fh.write(f"{nl:6d}   {int(it):8d}   {c:.12e}\n")
            fh.write("\n")

    for thr, name in THRESHOLDS:
        update_summary(os.path.join(args.data_dir, name), args.theta, reached[thr])

    w0, w1 = branch_weights(params, d_cav)
    print(f"  best F_e = {best_F:.8f}   branch weights = ({w0:.4f}, {w1:.4f})")
    for thr, name in THRESHOLDS:
        got = reached[thr]
        print(f"  layers for F_e >= {thr:<7g} : "
              + (f"{got}" if got is not None
                 else f"not reached within {args.nlayers_max}"))
    print(f"  curves -> {out_dir}/")

    final = os.path.join(out_dir, "optimized-params")
    with open(os.path.join(out_dir, f"nlayer{Nlayers}-params")) as src, \
            open(final, "w") as dst:
        dst.write(src.read())
    print(f"\n  optimized parameters ({Nlayers} layers, F_e = {F_e:.8f}) -> {final}\n")
    print(format_layers(params))


if __name__ == "__main__":
    main()
