"""flecc_with_gurobi.py – Solve the Fixed-Length Error-Correcting Code (FLECC)
problem as an Integer Linear Program (ILP) using Gurobi.

Goal: Maximise the number of selected codewords M from M_ub candidate slots,
      subject to a minimum Hamming or Lee distance d between every pair of
      active codewords.

Formulation overview
────────────────────
Decision variables
  y[i]        ∈ {0,1}   1 iff codeword slot i is selected           (i = 0…M_ub-1)
  x[i,j,k]   ∈ {0,1}   1 iff codeword i has symbol k at position j (j = 0…n-1, k = 0…q-1)

Symmetry-breaking auxiliary variables
  f[i,j,k] ∈ {0,1}   = x[i,j,k] · x[i+1,j,k]   (symbol match at position j)
  eq[i,j]  ∈ {0,1}   = 1 iff y[i+1]=1 and codewords i,i+1 are equal in positions 0…j-1

Constraints (static, built upfront)
  R1   sum_k x[i,j,k] == y[i]          (one symbol per active position)
  R2   sum_i y[i] >= M_lb              (at least M_lb codewords)
  Pack y[i] >= y[i+1]                  (active codewords at lower indices)
  Lex  val[i,j] <= val[i+1,j] + (1-eq[i,j])·q   (lexicographic order)

Distance constraints (lazy, added on-the-fly via branch-and-cut callback)
  For every integer-feasible incumbent that violates the pairwise distance
  requirement, a McCormick-linearised cut is injected for the offending pair.
  This avoids pre-building all C(M_ub,2) pairs − the dominant bottleneck for
  large instances − while still guaranteeing correctness.

  Hamming cut for pair (i1,i2):
    Introduce z_h[j,k] = x[i1,j,k] · x[i2,j,k]  (McCormick)
    sum_{j,k} z_h[j,k]  <=  (n−d) + (2−y[i1]−y[i2])·n

  Lee cut for pair (i1,i2):
    Introduce z_l[j,k1,k2] = x[i1,j,k1] · x[i2,j,k2]  (k1≠k2, McCormick)
    sum_{j,k1,k2} δ_L(k1,k2) · z_l[j,k1,k2]  >=  d − (2−y[i1]−y[i2])·d

Objective  maximise sum_i y[i]

Usage (standalone):
    python flecc_with_gurobi.py --n 5 --q 4 --d 3 [options]
"""

import argparse
import math
import os
import timeit

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_FILE  = "FLECC_Gurobi.xlsx"
DEFAULT_SUMMARY_FILE = "FLECC_Gurobi_Summary.xlsx"


# ─── Coding-theoretic bound helpers ───────────────────────────────────────────

def lee_delta(a: int, b: int, q: int) -> int:
    """Lee distance between symbols a and b in Z_q."""
    diff = abs(a - b)
    return min(diff, q - diff)


def _hamming_ball_volume(n: int, q: int, t: int) -> int:
    """Number of length-n q-ary vectors within Hamming distance t."""
    vol = 0
    for i in range(min(t, n) + 1):
        vol += math.comb(n, i) * (q - 1) ** i
    return vol


def _lee_ball_volume(n: int, q: int, t: int) -> int:
    """
    Number of length-n q-ary vectors within Lee distance t of the zero vector.
    Computed with dynamic programming over per-coordinate Lee weight.
    """
    if t < 0:
        return 0
    max_w = q // 2
    # Multiplicity: how many symbols in Z_q have each Lee weight
    mults = {0: 1}
    for w in range(1, max_w + 1):
        mults[w] = 1 if (q % 2 == 0 and w == max_w) else 2
    # dp[total_weight] = number of partial words with that cumulative Lee weight
    dp = [0] * (t + 1)
    dp[0] = 1
    for _ in range(n):
        ndp = [0] * (t + 1)
        for cw in range(t + 1):
            if dp[cw] == 0:
                continue
            for sw, mult in mults.items():
                nw = cw + sw
                if nw <= t:
                    ndp[nw] += dp[cw] * mult
        dp = ndp
    return sum(dp)


def estimate_M_ub(n: int, q: int, d: int, metric: str) -> int:
    """
    Estimate an upper bound on the maximum code size A_q(n, d).

    Hamming metric: min(Singleton, Hamming-sphere-packing, Plotkin).
    Lee metric:     min(translated-Singleton, translated-Hamming-sphere-packing,
                        Lee-sphere-packing).

    All bounds are capped at q^n (the total alphabet).
    """
    q_n = q ** n

    if metric == "hamming":
        # Singleton bound: q^(n-d+1)
        exp = n - d + 1
        sb = q ** max(exp, 0)
        # Hamming sphere-packing bound: floor(q^n / V(n,t))
        t = max(0, (d - 1) // 2)
        sp = q_n // max(_hamming_ball_volume(n, q, t), 1)
        # Plotkin bound (valid only when d > (q-1)*n/q)
        threshold = (q - 1) * n / q
        denom = q * d - (q - 1) * n
        pb = (q * d) // denom if d > threshold and denom > 0 else q_n
        return int(min(sb, sp, pb, q_n))

    # Lee metric
    half_q = q // 2
    if half_q == 0:
        return q_n
    # Lee sphere-packing bound
    t = max(0, (d - 1) // 2)
    lee_sp = q_n // max(_lee_ball_volume(n, q, t), 1)
    # Translated Hamming bounds (Hamming distance = ceil(d / floor(q/2)))
    d_h = math.ceil(d / half_q)
    sb = q ** max(n - d_h + 1, 0)
    t_h = max(0, (d_h - 1) // 2)
    ham_sp = q_n // max(_hamming_ball_volume(n, q, t_h), 1)
    return int(min(sb, ham_sp, lee_sp, q_n))


# ─── Excel helpers ─────────────────────────────────────────────────────────────

def _sheet_name(metric: str) -> str:
    m = str(metric).strip().lower()
    return m if m in {"hamming", "lee"} else "other"


def _append_df_to_excel_sheet(excel_file: str, df, metric: str) -> str:
    """
    Append *df* rows to the per-metric sheet of *excel_file*.
    Creates the file / sheet on first call; appends on subsequent calls.
    """
    if df is None or df.empty:
        return _sheet_name(metric)
    sheet = _sheet_name(metric)

    existing = pd.DataFrame()
    if os.path.exists(excel_file):
        try:
            existing = pd.read_excel(excel_file, sheet_name=sheet)
        except ValueError:
            # Sheet does not exist yet – start fresh.
            existing = pd.DataFrame()

    combined = df if existing.empty else pd.concat([existing, df], ignore_index=True)
    mode = "a" if os.path.exists(excel_file) else "w"
    extra_kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(excel_file, mode=mode, engine="openpyxl", **extra_kwargs) as writer:
        combined.to_excel(writer, sheet_name=sheet, index=False)
    return sheet


# ─── Lazy constraint callback ─────────────────────────────────────────────────

def _make_distance_callback(y, x, z_h, z_l, n: int, q: int, d: int, M_ub: int, metric: str):
    """
    Return a Gurobi callback function that injects distance constraints lazily.

    Strategy: z variables are pre-allocated in the model but have no constraints.
    When an incumbent violates the distance requirement for a pair (i1, i2),
    the callback adds the full set of McCormick and distance constraints for
    that pair as lazy cuts.  Pairs that are never simultaneously active never
    receive any constraints, keeping the model small.

    Hamming: for violating pair (i1,i2), add:
        z_h[i1,i2,j,k] <= x[i1,j,k]                   (Mc1)
        z_h[i1,i2,j,k] <= x[i2,j,k]                   (Mc2)
        z_h[i1,i2,j,k] >= x[i1,j,k] + x[i2,j,k] - 1  (Mc3)
        sum_{j,k} z_h[i1,i2,j,k] <= (n-d) + (2-y[i1]-y[i2])*n  (Ham)

    Lee: for violating pair (i1,i2), add:
        z_l[i1,i2,j,k1,k2] <= x[i1,j,k1]                      (Mc1)
        z_l[i1,i2,j,k1,k2] <= x[i2,j,k2]                      (Mc2)
        z_l[i1,i2,j,k1,k2] >= x[i1,j,k1] + x[i2,j,k2] - 1    (Mc3)
        sum_{j,k1,k2} delta*z_l >= d - (2-y[i1]-y[i2])*d      (Lee)

    Once added, these constraints persist in the B&C tree and will prevent the
    same violation from being revisited.  The set of injected pairs is tracked
    in _added_pairs to avoid duplicate cuts.
    """
    symbols = list(range(q))
    nz_pairs = [(k1, k2) for k1 in symbols for k2 in symbols if k1 != k2] if metric == "lee" else []
    delta_cache = {(k1, k2): lee_delta(k1, k2, q) for k1, k2 in nz_pairs}
    # Adjacent pairs already have full constraints from build(); skip them.
    added_pairs = {(i, i + 1) for i in range(M_ub - 1)}

    def _callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        y_vals = model.cbGetSolution(list(y.values()))
        active = [i for i in range(M_ub) if y_vals[i] > 0.5]
        if len(active) < 2:
            return

        # Batch-fetch x values for active codewords only
        x_keys = [(i, j, k) for i in active for j in range(n) for k in symbols]
        x_flat = model.cbGetSolution([x[key] for key in x_keys])
        x_val  = dict(zip(x_keys, x_flat))

        # Decode chosen symbol at each (codeword, position)
        sym = {}
        for i in active:
            for j in range(n):
                for k in symbols:
                    if x_val[i, j, k] > 0.5:
                        sym[i, j] = k
                        break

        for idx1 in range(len(active)):
            for idx2 in range(idx1 + 1, len(active)):
                i1, i2 = active[idx1], active[idx2]

                # Compute current distance
                if metric == "hamming":
                    dist = sum(1 for j in range(n) if sym.get((i1, j)) != sym.get((i2, j)))
                else:
                    dist = sum(lee_delta(sym[i1, j], sym[i2, j], q) for j in range(n))

                if dist >= d:
                    continue  # fine

                if (i1, i2) in added_pairs:
                    # Full McCormick cuts already added for this pair.
                    # The incumbent still violates — add a tighter no-good cut
                    # to cut off this specific point immediately.
                    lhs = gp.quicksum(
                        x[i1, j, sym[i1, j]] + x[i2, j, sym[i2, j]]
                        for j in range(n)
                    )
                    rhs = 2 * n - 1 + (2 - y[i1] - y[i2]) * 2 * n
                    model.cbLazy(lhs <= rhs)
                    continue

                # First violation for this pair: inject full McCormick + distance cuts
                added_pairs.add((i1, i2))

                if metric == "hamming":
                    for j in range(n):
                        for k in symbols:
                            z = z_h[i1, i2, j, k]
                            model.cbLazy(z <= x[i1, j, k])
                            model.cbLazy(z <= x[i2, j, k])
                            model.cbLazy(z >= x[i1, j, k] + x[i2, j, k] - 1)
                    model.cbLazy(
                        gp.quicksum(z_h[i1, i2, j, k] for j in range(n) for k in symbols)
                        <= (n - d) + (2 - y[i1] - y[i2]) * n
                    )
                else:
                    for j in range(n):
                        for k1, k2 in nz_pairs:
                            z = z_l[i1, i2, j, k1, k2]
                            model.cbLazy(z <= x[i1, j, k1])
                            model.cbLazy(z <= x[i2, j, k2])
                            model.cbLazy(z >= x[i1, j, k1] + x[i2, j, k2] - 1)
                    model.cbLazy(
                        gp.quicksum(
                            delta_cache[k1, k2] * z_l[i1, i2, j, k1, k2]
                            for j in range(n)
                            for k1, k2 in nz_pairs
                        )
                        >= d - (2 - y[i1] - y[i2]) * d
                    )

    return _callback



# ─── ILP model ────────────────────────────────────────────────────────────────

class FleccWithGurobi:
    """
    FLECC ILP solver backed by Gurobi.

    Call :meth:`build` to construct the model, then :meth:`solve` to optimise.
    The best codewords (if any) are available via :attr:`solution` after solving.
    """

    def __init__(
        self,
        n: int,
        q: int,
        d: int,
        M_ub: int,
        M_lb: int = 2,
        metric: str = "hamming",
    ):
        """
        Parameters
        ----------
        n      : codeword length
        q      : alphabet size (symbols 0 … q-1)
        d      : minimum required distance
        M_ub   : number of candidate codeword slots (upper bound on M)
        M_lb   : required lower bound on selected codewords (default 2)
        metric : 'hamming' or 'lee'
        """
        if q < 2:
            raise ValueError("q must be >= 2")
        if n < 1:
            raise ValueError("n must be >= 1")
        if d < 1:
            raise ValueError("d must be >= 1")
        if M_lb < 0:
            raise ValueError("M_lb must be >= 0")
        if M_ub < M_lb:
            raise ValueError("M_ub must be >= M_lb")
        if metric not in {"hamming", "lee"}:
            raise ValueError("metric must be 'hamming' or 'lee'")

        self.n      = n
        self.q      = q
        self.d      = d
        self.M_ub   = M_ub
        self.M_lb   = M_lb
        self.metric = metric

        # Populated by build() / solve()
        self.model    = None
        self.y        = None      # {i: Var}
        self.x        = None      # {(i,j,k): Var}
        self.z_h      = None      # {(i1,i2,j,k): Var}  Hamming aux, or None
        self.z_l      = None      # {(i1,i2,j,k1,k2): Var}  Lee aux, or None
        self.solution = None      # list of codeword strings, or None
        self.obj_val  = None      # best M found (int), or None
        self.status   = None      # "Optimal" | "Feasible" | "Infeasible" | "Unknown"
        self.runtime  = None      # wall-clock seconds (float)

    # ── Public interface ───────────────────────────────────────────────────────

    def build(self):
        """Construct the Gurobi ILP model without solving it."""
        n, q, d, M_ub, M_lb = self.n, self.q, self.d, self.M_ub, self.M_lb
        symbols = list(range(q))

        m = gp.Model("FLECC")
        m.setParam("OutputFlag", 1)

        # ── Decision variables ─────────────────────────────────────────────
        # y[i] = 1 iff codeword slot i is selected
        y = m.addVars(M_ub, vtype=GRB.BINARY, name="y")

        # x[i,j,k] = 1 iff codeword i has symbol k at position j
        x = m.addVars(
            [(i, j, k) for i in range(M_ub) for j in range(n) for k in symbols],
            vtype=GRB.BINARY,
            name="x",
        )

        # ── R1: each active codeword has exactly one symbol per position ───
        # sum_k x[i,j,k] == y[i]  for all i, j
        m.addConstrs(
            (gp.quicksum(x[i, j, k] for k in symbols) == y[i]
             for i in range(M_ub) for j in range(n)),
            name="R1",
        )

        # ── R2: enforce lower bound on selected codewords ──────────────────
        m.addConstr(gp.quicksum(y[i] for i in range(M_ub)) >= M_lb, name="R2")

        # ── Distance constraints: seed pairs upfront, rest lazily ──────────
        # Adjacent pairs (i, i+1) are added immediately so the LP relaxation
        # has enough distance information to produce useful bounds.  All other
        # pairs are handled on-demand by the branch-and-cut callback.
        if self.metric == "hamming":
            symbols_local = list(range(q))
            z_h_idx = [
                (i1, i2, j, k)
                for i1 in range(M_ub)
                for i2 in range(i1 + 1, M_ub)
                for j in range(n)
                for k in symbols_local
            ]
            z_h = m.addVars(z_h_idx, vtype=GRB.BINARY, name="zh")
            # Add full McCormick + distance constraints for adjacent pairs
            for i1 in range(M_ub - 1):
                i2 = i1 + 1
                for j in range(n):
                    for k in symbols_local:
                        m.addConstr(z_h[i1, i2, j, k] <= x[i1, j, k])
                        m.addConstr(z_h[i1, i2, j, k] <= x[i2, j, k])
                        m.addConstr(z_h[i1, i2, j, k] >= x[i1, j, k] + x[i2, j, k] - 1)
                m.addConstr(
                    gp.quicksum(z_h[i1, i2, j, k] for j in range(n) for k in symbols_local)
                    <= (n - d) + (2 - y[i1] - y[i2]) * n,
                    name=f"Ham_{i1}_{i2}",
                )
            z_l = None
        else:
            symbols_local = list(range(q))
            nz_pairs = [(k1, k2) for k1 in symbols_local for k2 in symbols_local if k1 != k2]
            delta_local = {(k1, k2): lee_delta(k1, k2, q) for k1, k2 in nz_pairs}
            z_h = None
            z_l_idx = [
                (i1, i2, j, k1, k2)
                for i1 in range(M_ub)
                for i2 in range(i1 + 1, M_ub)
                for j in range(n)
                for k1, k2 in nz_pairs
            ]
            z_l = m.addVars(z_l_idx, vtype=GRB.BINARY, name="zl")
            # Add full McCormick + distance constraints for adjacent pairs
            for i1 in range(M_ub - 1):
                i2 = i1 + 1
                for j in range(n):
                    for k1, k2 in nz_pairs:
                        m.addConstr(z_l[i1, i2, j, k1, k2] <= x[i1, j, k1])
                        m.addConstr(z_l[i1, i2, j, k1, k2] <= x[i2, j, k2])
                        m.addConstr(z_l[i1, i2, j, k1, k2] >= x[i1, j, k1] + x[i2, j, k2] - 1)
                m.addConstr(
                    gp.quicksum(
                        delta_local[k1, k2] * z_l[i1, i2, j, k1, k2]
                        for j in range(n) for k1, k2 in nz_pairs
                    )
                    >= d - (2 - y[i1] - y[i2]) * d,
                    name=f"Lee_{i1}_{i2}",
                )

        # ── Symmetry breaking ──────────────────────────────────────────────
        self._add_symmetry_breaking(m, y, x)

        # ── Objective: maximise number of active codewords ─────────────────
        m.setObjective(gp.quicksum(y[i] for i in range(M_ub)), GRB.MAXIMIZE)

        self.model = m
        self.y     = y
        self.x     = x
        if self.metric == "hamming":
            self.z_h = z_h
            self.z_l = None
        else:
            self.z_h = None
            self.z_l = z_l

    def solve(self, time_limit=None):
        """
        Run Gurobi on the built model and extract the solution.

        Distance constraints are enforced lazily via a branch-and-cut callback:
        for every integer-feasible incumbent, each active pair is checked and
        a McCormick-linearised cut is added if the distance is violated.

        Parameters
        ----------
        time_limit : float or None
            Wall-clock seconds given to Gurobi.  None = no limit.

        Returns
        -------
        status   : str   – "Optimal", "Feasible", "Infeasible", or "Unknown"
        obj_val  : int   – best M found (None if no feasible solution found)
        solution : list  – list of codeword strings (None if no solution found)
        runtime  : float – elapsed seconds
        """
        if self.model is None:
            raise RuntimeError("Call build() before solve().")

        if time_limit is not None:
            self.model.setParam("TimeLimit", float(time_limit))

        # Lazy constraints require LazyConstraints parameter = 1
        self.model.setParam("LazyConstraints", 1)

        cb = _make_distance_callback(
            y=self.y,
            x=self.x,
            z_h=self.z_h,
            z_l=self.z_l,
            n=self.n,
            q=self.q,
            d=self.d,
            M_ub=self.M_ub,
            metric=self.metric,
        )

        t0 = timeit.default_timer()
        self.model.optimize(cb)
        self.runtime = timeit.default_timer() - t0

        gstat = self.model.Status
        if gstat == GRB.OPTIMAL:
            self.status = "Optimal"
        elif gstat == GRB.TIME_LIMIT and self.model.SolCount > 0:
            self.status = "Feasible"
        elif gstat == GRB.INFEASIBLE:
            self.status = "Infeasible"
        else:
            self.status = "Unknown"

        if self.model.SolCount > 0:
            self.obj_val  = int(round(self.model.ObjVal))
            self.solution = self._extract_solution()
        else:
            self.obj_val  = None
            self.solution = None

        return self.status, self.obj_val, self.solution, self.runtime

    # ── Private model-building helpers ────────────────────────────────────────

    def _add_symmetry_breaking(self, m, y, x):
        """
        Symmetry-breaking constraints to reduce the search space.

        (1) Packing:  y[i] >= y[i+1]
            Forces all selected codewords into the lowest-indexed slots.
            If slot i+1 can be active, slot i must be too.

        (2) Lexicographic order:  w[i] <=_lex w[i+1]  for every i.

            Auxiliary variables:
              eq[i,j]  ∈ {0,1} = 1 iff y[i+1]=1 AND codewords i, i+1 agree
                                  in all positions 0…j-1.

            Initialisation:  eq[i,0] = y[i+1]
              (When y[i+1] = 0 the lex chain is deactivated, preventing the
               constraint from forcing the active codeword i to start with 0.)

            At each position j:
              val[i,j]  = sum_k k * x[i,j,k]    (symbol value, in 0…q-1)
              Constraint:  val[i,j] - val[i+1,j]  <=  (1 - eq[i,j]) * q

            Equality update (j = 0…n-2):
              f[i,j,k]  = x[i,j,k] AND x[i+1,j,k]   (match on symbol k)
              match[i,j] = sum_k f[i,j,k]             (1 iff positions equal)
              eq[i,j+1]  = eq[i,j] AND match[i,j]    (AND of two binary vars)
        """
        n, q, M_ub = self.n, self.q, self.M_ub
        symbols = list(range(q))

        for i in range(M_ub - 1):
            # ── (1) Packing: slot i must be active if slot i+1 is ─────────
            m.addConstr(y[i] >= y[i + 1], name=f"Pack_{i}")

            # ── (2) Lexicographic ordering ─────────────────────────────────
            # eq_prev starts as y[i+1]: if next slot inactive, entire lex
            # chain is inactive (constraint trivially satisfied at every j).
            eq_prev = y[i + 1]

            for j in range(n):
                # Weighted symbol values for codewords i and i+1 at position j
                val_i  = gp.quicksum(k * x[i,     j, k] for k in symbols)
                val_i1 = gp.quicksum(k * x[i + 1, j, k] for k in symbols)

                # Lex constraint at position j:
                #   If eq_prev = 1 (equal so far AND next codeword active):
                #     val[i,j] <= val[i+1,j]
                #   If eq_prev = 0 (differ earlier OR next inactive):
                #     val[i,j] <= val[i+1,j] + q  (trivially satisfied)
                m.addConstr(
                    val_i - val_i1 <= (1 - eq_prev) * q,
                    name=f"Lex_{i}_{j}",
                )

                # Update the equality indicator for the next position
                if j < n - 1:
                    # f[i,j,k] = x[i,j,k] AND x[i+1,j,k]  (McCormick)
                    f_vars = []
                    for k in symbols:
                        fv = m.addVar(vtype=GRB.BINARY, name=f"f_{i}_{j}_{k}")
                        m.addConstr(fv <= x[i,     j, k])
                        m.addConstr(fv <= x[i + 1, j, k])
                        m.addConstr(fv >= x[i, j, k] + x[i + 1, j, k] - 1)
                        f_vars.append(fv)

                    # match[i,j] = 1 iff codewords i and i+1 have the same
                    # symbol at position j  (sum is in {0,1} by exactly-one)
                    match_v = m.addVar(vtype=GRB.BINARY, name=f"match_{i}_{j}")
                    m.addConstr(
                        match_v == gp.quicksum(f_vars),
                        name=f"Match_{i}_{j}",
                    )

                    # eq[i,j+1] = eq_prev AND match_v
                    # Encoded as:  eq <= eq_prev,  eq <= match_v,
                    #              eq >= eq_prev + match_v - 1
                    eq_next = m.addVar(vtype=GRB.BINARY, name=f"eq_{i}_{j+1}")
                    m.addConstr(eq_next <= eq_prev,              name=f"Eq1_{i}_{j}")
                    m.addConstr(eq_next <= match_v,              name=f"Eq2_{i}_{j}")
                    m.addConstr(eq_next >= eq_prev + match_v - 1, name=f"Eq3_{i}_{j}")

                    eq_prev = eq_next

    def _extract_solution(self):
        """Read selected codeword strings from the solved model."""
        solution = []
        for i in range(self.M_ub):
            if self.y[i].X > 0.5:                  # slot i is active
                word = []
                for j in range(self.n):
                    for k in range(self.q):
                        if self.x[i, j, k].X > 0.5:
                            word.append(str(k))
                            break
                solution.append("".join(word))
        return solution


# ─── Top-level solver interface ────────────────────────────────────────────────

def solve_flecc_gurobi(
    n: int,
    q: int,
    d: int,
    M_lb: int = 2,
    M_ub: int = None,
    metric: str = "hamming",
    time_limit: float = 600.0,
    output_file: str = DEFAULT_OUTPUT_FILE,
    test: bool = False,
):
    """
    Solve one FLECC instance with Gurobi and optionally write results to Excel.

    Parameters
    ----------
    n          : codeword length
    q          : alphabet size (symbols 0…q-1)
    d          : minimum required distance
    M_lb       : lower bound on # codewords  (default 2)
    M_ub       : upper bound / candidate slots  (auto-estimated when None)
    metric     : 'hamming' or 'lee'
    time_limit : solver wall-clock budget in seconds  (None = unlimited)
    output_file: path to Excel output file
    test       : if True, skip all Excel I/O

    Returns
    -------
    (status, obj_val, solution, result_row)
      status     : str  – "Optimal", "Feasible", "Infeasible", "Unknown"
      obj_val    : int  – best M found (or None)
      solution   : list – list of codeword strings (or None)
      result_row : dict – full result record
    """
    # ── Estimate M_ub if not provided ──────────────────────────────────────
    if M_ub is None:
        M_ub = estimate_M_ub(n, q, d, metric)
    M_ub = max(M_ub, M_lb)       # ensure at least M_lb slots are available

    print(f"\n{'='*70}")
    print("FLECC ILP Solver (Gurobi)")
    print(f"{'='*70}")
    print(f"  n={n}, q={q}, d={d}, metric={metric}")
    print(f"  M_lb={M_lb}, M_ub (candidate slots)={M_ub}")
    if time_limit is not None:
        print(f"  Time limit: {time_limit}s")
    print(f"{'='*70}\n")

    # ── Build model ────────────────────────────────────────────────────────
    solver = FleccWithGurobi(n=n, q=q, d=d, M_ub=M_ub, M_lb=M_lb, metric=metric)
    solver.build()

    # ── Solve ──────────────────────────────────────────────────────────────
    status, obj_val, solution, runtime = solver.solve(time_limit=time_limit)

    # ── Console report ─────────────────────────────────────────────────────
    print(f"\nStatus  : {status}")
    print(f"Best M  : {obj_val}")
    print(f"Runtime : {runtime:.4f}s")
    if solution:
        print(f"Codewords ({len(solution)}):")
        for cw in solution:
            print(f"  {cw}")

    # ── Build result row ───────────────────────────────────────────────────
    instance_name = f"FLECC_{n}_{d}_{metric}_q{q}_gurobi"
    result_row = {
        "Instance":  instance_name,
        "n":         n,
        "q":         q,
        "d":         d,
        "Metric":    metric,
        "M_lb":      M_lb,
        "M_ub":      M_ub,
        "M_best":    obj_val,
        "Status":    status,
        "Runtime(s)": round(runtime, 4),
        "Codewords": str(solution) if solution else "None",
        "Method":    "ILP-Gurobi",
    }

    # ── Save to Excel ──────────────────────────────────────────────────────
    if not test:
        results_df = pd.DataFrame([result_row])
        sheet = _append_df_to_excel_sheet(output_file, results_df, metric)
        print(f"\nResults saved  → {output_file}  (sheet: {sheet})")

        summary_row = {
            "Instance": instance_name,
            "n":        n,
            "q":        q,
            "d":        d,
            "M_best":   obj_val,
            "UB":       M_ub,
            "Time(s)":  round(runtime, 4),
            "Status":   status,
            "Method":   "ILP-Gurobi",
        }
        s_sheet = _append_df_to_excel_sheet(
            DEFAULT_SUMMARY_FILE, pd.DataFrame([summary_row]), metric
        )
        print(f"Summary saved  → {DEFAULT_SUMMARY_FILE}  (sheet: {s_sheet})")

    return status, obj_val, solution, result_row


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Solve the FLECC problem as an ILP using Gurobi (maximises codewords).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n",          type=int,   required=True,
                        help="Codeword length")
    parser.add_argument("--q",          type=int,   default=2,
                        help="Alphabet size (symbols 0…q-1)")
    parser.add_argument("--d",          type=int,   required=True,
                        help="Minimum required distance")
    parser.add_argument("--M_lb",       type=int,   default=2,
                        help="Lower bound on number of codewords")
    parser.add_argument("--M_ub",       type=int,   default=None,
                        help="Upper bound / candidate slots (auto-estimated if omitted)")
    parser.add_argument("--metric",     type=str,   default="hamming",
                        choices=["hamming", "lee"],
                        help="Distance metric")
    parser.add_argument("--time_limit", type=float, default=600.0,
                        help="Solver time limit in seconds (0 = unlimited)")
    parser.add_argument("--output",     type=str,   default=DEFAULT_OUTPUT_FILE,
                        help="Output Excel file")
    parser.add_argument("--test",       action="store_true",
                        help="Dry-run: skip all Excel I/O")

    args = parser.parse_args()

    time_limit = args.time_limit if args.time_limit > 0 else None

    solve_flecc_gurobi(
        n          = args.n,
        q          = args.q,
        d          = args.d,
        M_lb       = args.M_lb,
        M_ub       = args.M_ub,
        metric     = args.metric,
        time_limit = time_limit,
        output_file= args.output,
        test       = args.test,
    )


if __name__ == "__main__":
    main()
