# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Robust / tube-CBF margin estimator for the decPLM obstacle safety filter (CPU-only, no Isaac).

Motivation
----------
The CBF QP guarantees the *commanded* box twist keeps every body outside the inflated keep-out, but the
*realized* trajectory lags: the learned policy tracks imperfectly and (because of the constellation
reward) the agents swing **wider** than the rigid-constellation model assumes. So ``cbf_margin`` is
load-bearing -- it must absorb that excursion.

This script turns the hand-tuned margin into a **data-grounded (tube-CBF) margin**. Treat the realized
deviation from the commanded/rigid prediction as a bounded position disturbance ``delta``. Inflating
every keep-out radius by the margin ``m`` means the realized physical clearance at the worst point is
roughly ``c(m) = m - delta(m)``. Physical safety with buffer ``b`` needs ``c(m) >= b``.

Estimation
----------
For one run at margin ``m`` with per-step margin-free clearance series ``min_clearance_t`` (saved by
``play_cbf.py`` as ``cbf_series_*.npz``):

    c       = min_t  min_clearance_t            # realized worst physical clearance (< 0 == contact)
    delta_t = max(0, m - min_clearance_t)       # per-step realized excursion past the commanded margin
    delta   = max_t delta_t  (and p99)          # worst-case excursion bound at this margin

Because deflecting earlier (larger ``m``) reduces overshoot, ``delta`` shrinks as ``m`` grows, so a
**margin sweep** (several runs at different ``m``) lets us fit ``c(m)`` and solve ``c(m*) = b`` for the
smallest safe margin. With a single run we report a first-iterate ``m* = b + delta`` (re-measure at
``m*`` since ``delta`` depends on ``m``).

Usage
-----
    # from saved run series (one or more; ideally a sweep over cbf_margin):
    python cbf_robust_margin.py path/to/cbf_series_on_ep0.npz [more.npz ...] --buffer 0.05

    # or directly from measured (margin, realized_min_clearance) points (no files needed):
    python cbf_robust_margin.py --points "0.12,-0.031 0.25,0.228" --buffer 0.05
"""

from __future__ import annotations

import argparse
import glob

import numpy as np


def _analyze_run(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    margin = float(d["margin"]) if "margin" in d else float("nan")
    cbf = str(d["cbf"]) if "cbf" in d else "?"
    clr = np.asarray(d["min_clearance"], dtype=float)
    c = float(clr.min())
    delta_t = np.clip(margin - clr, 0.0, None)
    return {
        "path": npz_path,
        "cbf": cbf,
        "margin": margin,
        "min_clearance": c,
        "delta_max": float(delta_t.max()),
        "delta_p99": float(np.percentile(delta_t, 99)),
        "phys_safe": c >= 0.0,
    }


def _fit_and_recommend(points: list[tuple[float, float]], buffer: float):
    """points: list of (margin, realized_min_clearance). Linear-fit c(m) and solve c(m*) = buffer."""
    pts = sorted(set(points))
    print("\nMeasured clearance vs margin:")
    for m, c in pts:
        print(f"    margin={m:.3f}  ->  realized min_clearance={c:+.3f} m  "
              f"({'SAFE' if c >= 0 else 'CONTACT'})")
    if len({m for m, _ in pts}) < 2:
        m, c = pts[0]
        delta = max(0.0, m - c)
        m_star = buffer + delta
        print(f"\nSingle margin only -> first-iterate estimate (re-measure at the new margin):")
        print(f"    excursion delta ~= margin - clearance = {m:.3f} - ({c:+.3f}) = {delta:.3f} m")
        print(f"    recommended cbf_margin ~= buffer + delta = {buffer:.3f} + {delta:.3f} = {m_star:.3f} m")
        return
    # linear fit c(m) = a*m + b0
    ms = np.array([m for m, _ in pts])
    cs = np.array([c for _, c in pts])
    a, b0 = np.polyfit(ms, cs, 1)
    if a <= 1e-6:
        print("\n[WARN] clearance does not increase with margin in this data; need more / cleaner points.")
        return
    m_zero = (0.0 - b0) / a       # just-physically-safe margin
    m_buf = (buffer - b0) / a     # margin for the requested physical buffer
    print(f"\nLinear fit:  clearance(m) = {a:.2f}*m + ({b0:+.3f})")
    print(f"    -> just physically safe (clearance=0)  at cbf_margin ~= {m_zero:.3f} m")
    print(f"    -> {buffer*100:.0f} cm physical buffer        at cbf_margin ~= {m_buf:.3f} m   <== recommended")
    print("\nNote: delta(m) shrinks as m grows (earlier, gentler deflection), so this is conservative; "
          "verify by re-running at the recommended margin.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="*", help="cbf_series_*.npz file(s) saved by play_cbf.py.")
    ap.add_argument("--points", type=str, default=None,
                    help='Direct "margin,clearance" pairs, space-separated, e.g. "0.12,-0.03 0.25,0.23".')
    ap.add_argument("--buffer", type=float, default=0.05, help="Target physical clearance buffer (m).")
    args = ap.parse_args()

    points: list[tuple[float, float]] = []

    files = []
    for p in args.npz:
        files.extend(sorted(glob.glob(p)))
    if files:
        print("Per-run analysis:")
        print(f"  {'cbf':4} {'margin':>7} {'min_clear':>10} {'delta_max':>10} {'delta_p99':>10}  safe  file")
        for f in files:
            r = _analyze_run(f)
            print(f"  {r['cbf']:4} {r['margin']:7.3f} {r['min_clearance']:+10.3f} "
                  f"{r['delta_max']:10.3f} {r['delta_p99']:10.3f}  "
                  f"{'Y' if r['phys_safe'] else 'n':>4}  {r['path'].split('/')[-1]}")
            if r["cbf"].lower() == "on" and np.isfinite(r["margin"]):
                points.append((r["margin"], r["min_clearance"]))

    if args.points:
        for tok in args.points.split():
            m, c = tok.split(",")
            points.append((float(m), float(c)))

    if not points:
        ap.error("No data. Provide cbf_series_*.npz file(s) (with --cbf on) or --points.")

    _fit_and_recommend(points, args.buffer)


if __name__ == "__main__":
    main()
