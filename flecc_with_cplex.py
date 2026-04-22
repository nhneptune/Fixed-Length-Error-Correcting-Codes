"""flecc_with_cplex.py – Solve the Fixed-Length Error-Correcting Code (FLECC)
problem as an Integer Linear Program (ILP) using IBM CPLEX (via cplex Python API).

Goal: Maximise the number of selected codewords M from M_ub candidate slots,
      subject to a minimum Hamming or Lee distance d between every pair of
      active codewords.

Formulation overview
--------------------
Decision variables
  y[i]        ∈ {0,1}   1 iff codeword slot i is selected           (i = 0...M_ub-1)
  x[i,j,k]   ∈ {0,1}   1 iff codeword i has symbol k at position j (j = 0...n-1, k = 0...q-1)

Hamming metric auxiliary variables
  z_h[i1,i2,j,k] ∈ {0,1}  = x[i1,j,k] · x[i2,j,k]  (McCormick linearisation)

Lee metric auxiliary variables
  z_l[i1,i2,j,k1,k2] ∈ {0,1}  = x[i1,j,k1] · x[i2,j,k2]  (k1 ≠ k2 only, McCormick)

Symmetry-breaking auxiliary variables
  f[i,j,k]  ∈ {0,1}  = x[i,j,k] · x[i+1,j,k]   (symbol match indicator)
  match[i,j] ∈ {0,1}  = 1 iff codewords i, i+1 agree at position j
  eq[i,j]   ∈ {0,1}  = 1 iff y[i+1]=1 AND codewords i,i+1 agree in positions 0...j-1

Constraints
  R1   sum_k x[i,j,k] == y[i]              (one symbol per active position)
  R2   sum_i y[i] >= M_lb                  (at least M_lb codewords)
  Ham  sum_{j,k} z_h[i1,i2,j,k] <= (n-d) + (2-y[i1]-y[i2])·n
  Lee  sum_{j,k1≠k2} δ·z_l[...]   >= d - (2-y[i1]-y[i2])·d
  Pack y[i] >= y[i+1]               (active slots at lower indices)
  Lex  val[i,j] - val[i+1,j] <= (1-eq[i,j])·q  (lexicographic ordering)

Objective  maximise sum_i y[i]

Usage (standalone):
    python flecc_with_cplex.py --n 5 --q 4 --d 3 [options]
"""

import argparse
import math
import os
import timeit

import cplex
import pandas as pd


# --- Constants ----------------------------------------------------------------

DEFAULT_OUTPUT_FILE  = "FLECC_CPLEX.xlsx"
DEFAULT_SUMMARY_FILE = "FLECC_CPLEX_Summary.xlsx"


# --- Coding-theoretic bound helpers -------------------------------------------

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
    Computed with DP over per-coordinate Lee weight.
    """
    if t < 0:
        return 0
    max_w = q // 2
    mults = {0: 1}
    for w in range(1, max_w + 1):
        mults[w] = 1 if (q % 2 == 0 and w == max_w) else 2
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
    All bounds are capped at q^n.
    """
    q_n = q ** n

    if metric == "hamming":
        exp = n - d + 1
        sb = q ** max(exp, 0)
        t = max(0, (d - 1) // 2)
        sp = q_n // max(_hamming_ball_volume(n, q, t), 1)
        threshold = (q - 1) * n / q
        denom = q * d - (q - 1) * n
        pb = (q * d) // denom if d > threshold and denom > 0 else q_n
        return int(min(sb, sp, pb, q_n))

    half_q = q // 2
    if half_q == 0:
        return q_n
    t = max(0, (d - 1) // 2)
    lee_sp = q_n // max(_lee_ball_volume(n, q, t), 1)
    d_h = math.ceil(d / half_q)
    sb = q ** max(n - d_h + 1, 0)
    t_h = max(0, (d_h - 1) // 2)
    ham_sp = q_n // max(_hamming_ball_volume(n, q, t_h), 1)
    return int(min(sb, ham_sp, lee_sp, q_n))


# --- Excel helpers -------------------------------------------------------------

def _sheet_name(metric: str) -> str:
    m = str(metric).strip().lower()
    return m if m in {"hamming", "lee"} else "other"


def _append_df_to_excel_sheet(excel_file: str, df, metric: str) -> str:
    """
    Append *df* rows to the per-metric sheet of *excel_file*.
    Creates the file / sheet the first time; appends on subsequent calls.
    """
    if df is None or df.empty:
        return _sheet_name(metric)
    sheet = _sheet_name(metric)

    existing = pd.DataFrame()
    if os.path.exists(excel_file):
        try:
            existing = pd.read_excel(excel_file, sheet_name=sheet)
        except ValueError:
            existing = pd.DataFrame()

    combined = df if existing.empty else pd.concat([existing, df], ignore_index=True)
    mode = "a" if os.path.exists(excel_file) else "w"
    extra_kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(excel_file, mode=mode, engine="openpyxl", **extra_kwargs) as writer:
        combined.to_excel(writer, sheet_name=sheet, index=False)
    return sheet


# --- Lazy constraint callback -------------------------------------------------

class _DistanceLazyCallback(cplex.callbacks.LazyConstraintCallback):
    """
    CPLEX lazy constraint callback that injects distance constraints on demand.

    All z auxiliary variables (McCormick) are pre-allocated by build() but have
    NO constraints.  When an incumbent is found and a pair (i1, i2) violates the
    minimum distance, this callback adds the full McCormick (Mc1/Mc2/Mc3) for
    every (j,k) plus the distance constraint for that pair.

    Hamming distance constraint (lazy, injected per-pair):
      sum_{j,k} z_h[i1,i2,j,k]  <=  (n-d) + (2-y[i1]-y[i2])·n
      With McCormick:  z <= x[i1,j,k],  z <= x[i2,j,k],  z >= x[i1]+x[i2]-1

    Lee distance constraint (lazy, injected per-pair):
      sum_{j,k1≠k2} δ_L(k1,k2)·z_l[i1,i2,j,k1,k2] >= d - (2-y[i1]-y[i2])·d
      With McCormick:  z <= x[i1,j,k1],  z <= x[i2,j,k2],  z >= x[i1]+x[i2]-1

    Seed pairs (adjacent + slot-0 anchored) already have constraints from build(),
    so the callback skips pairs in _added_pairs.
    """

    def __call__(self):
        # Lazy-initialise _added_pairs on first invocation.
        if not hasattr(self, '_added_pairs'):
            self._added_pairs = {(i, i + 1) for i in range(self._M_ub - 1)}
            for i2 in range(1, self._M_ub):
                self._added_pairs.add((0, i2))

        y_vals = self.get_values(self._y_names)
        active = [i for i, v in enumerate(y_vals) if v > 0.5]
        if len(active) < 2:
            return

        x_names_needed = [
            f"x_{i}_{j}_{k}"
            for i in active
            for j in range(self._n)
            for k in self._symbols
        ]
        x_vals_flat = self.get_values(x_names_needed)
        x_val = dict(zip(x_names_needed, x_vals_flat))

        sym = {}
        for i in active:
            for j in range(self._n):
                for k in self._symbols:
                    if x_val[f"x_{i}_{j}_{k}"] > 0.5:
                        sym[i, j] = k
                        break

        n, q, d = self._n, self._q, self._d

        for idx1 in range(len(active)):
            for idx2 in range(idx1 + 1, len(active)):
                i1, i2 = active[idx1], active[idx2]

                if (i1, i2) in self._added_pairs:
                    continue

                if self._metric == "hamming":
                    dist = sum(
                        1 for j in range(n)
                        if sym.get((i1, j)) != sym.get((i2, j))
                    )
                else:  # lee
                    dist = sum(
                        lee_delta(sym[i1, j], sym[i2, j], q)
                        for j in range(n)
                    )

                if dist >= d:
                    continue

                # Mark pair as constrained
                self._added_pairs.add((i1, i2))

                lin_exprs = []
                senses    = []
                rhs_vals  = []

                if self._metric == "hamming":
                    pair_z = []
                    for j in range(n):
                        for k in self._symbols:
                            z  = f"zh_{i1}_{i2}_{j}_{k}"
                            x1 = f"x_{i1}_{j}_{k}"
                            x2 = f"x_{i2}_{j}_{k}"
                            pair_z.append(z)
                            # Mc1: z - x1 <= 0
                            lin_exprs.append(cplex.SparsePair(ind=[z, x1], val=[1.0, -1.0]))
                            senses.append("L"); rhs_vals.append(0.0)
                            # Mc2: z - x2 <= 0
                            lin_exprs.append(cplex.SparsePair(ind=[z, x2], val=[1.0, -1.0]))
                            senses.append("L"); rhs_vals.append(0.0)
                            # Mc3: z - x1 - x2 >= -1
                            lin_exprs.append(cplex.SparsePair(ind=[z, x1, x2], val=[1.0, -1.0, -1.0]))
                            senses.append("G"); rhs_vals.append(-1.0)
                    # Ham distance:
                    y1, y2 = f"y_{i1}", f"y_{i2}"
                    lin_exprs.append(cplex.SparsePair(
                        ind=pair_z + [y1, y2],
                        val=[1.0] * len(pair_z) + [float(n), float(n)],
                    ))
                    senses.append("L"); rhs_vals.append(float(3 * n - d))

                else:  # lee
                    dist_vars   = []
                    dist_coeffs = []
                    for j in range(n):
                        for k1, k2 in self._nonzero_pairs:
                            z  = f"zl_{i1}_{i2}_{j}_{k1}_{k2}"
                            x1 = f"x_{i1}_{j}_{k1}"
                            x2 = f"x_{i2}_{j}_{k2}"
                            d_coef = float(self._delta_cache[k1, k2])
                            dist_vars.append(z)
                            dist_coeffs.append(d_coef)
                            # Mc1
                            lin_exprs.append(cplex.SparsePair(ind=[z, x1], val=[1.0, -1.0]))
                            senses.append("L"); rhs_vals.append(0.0)
                            # Mc2
                            lin_exprs.append(cplex.SparsePair(ind=[z, x2], val=[1.0, -1.0]))
                            senses.append("L"); rhs_vals.append(0.0)
                            # Mc3
                            lin_exprs.append(cplex.SparsePair(ind=[z, x1, x2], val=[1.0, -1.0, -1.0]))
                            senses.append("G"); rhs_vals.append(-1.0)
                    # Lee distance:
                    y1, y2 = f"y_{i1}", f"y_{i2}"
                    lin_exprs.append(cplex.SparsePair(
                        ind=dist_vars + [y1, y2],
                        val=dist_coeffs + [-float(d), -float(d)],
                    ))
                    senses.append("G"); rhs_vals.append(-float(d))

                for sp, sense, rhs in zip(lin_exprs, senses, rhs_vals):
                    self.add(sp, sense, rhs)


# --- ILP model ----------------------------------------------------------------

class FleccWithCplex:
    """
    FLECC ILP solver backed by IBM CPLEX.

    Call :meth:`build` to construct the model, then :meth:`solve` to optimise.
    The best codewords (if any) are available via :attr:`solution` after solving.

    Variable naming convention used internally (all strings fed to CPLEX):
      y_i              – slot-active indicator
      x_i_j_k          – symbol assignment
      zh_i1_i2_j_k     – Hamming McCormick auxiliary
      zl_i1_i2_j_k1_k2 – Lee McCormick auxiliary
      f_i_j_k          – symbol-match auxiliary (lex)
      match_i_j        – position-equal auxiliary (lex)
      eq_i_j           – lex equality-so-far auxiliary
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
        q      : alphabet size (symbols 0 ... q-1)
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
        self._prob    = None   # cplex.Cplex instance
        self.solution = None   # list of codeword strings, or None
        self.obj_val  = None   # best M found (int), or None
        self.status   = None   # "Optimal" | "Feasible" | "Infeasible" | "Unknown"
        self.runtime  = None   # wall-clock seconds (float)

    # -- Variable name helpers --------------------------------------------------

    @staticmethod
    def _yn(i):                           return f"y_{i}"
    @staticmethod
    def _xn(i, j, k):                    return f"x_{i}_{j}_{k}"
    @staticmethod
    def _zhn(i1, i2, j, k):              return f"zh_{i1}_{i2}_{j}_{k}"
    @staticmethod
    def _zln(i1, i2, j, k1, k2):         return f"zl_{i1}_{i2}_{j}_{k1}_{k2}"
    @staticmethod
    def _fn(i, j, k):                    return f"f_{i}_{j}_{k}"
    @staticmethod
    def _matchn(i, j):                   return f"match_{i}_{j}"
    @staticmethod
    def _eqn(i, j):                      return f"eq_{i}_{j}"

    # -- Public interface -------------------------------------------------------

    def build(self):
        """Construct the CPLEX ILP model without solving it."""
        n, q, d, M_ub, M_lb = self.n, self.q, self.d, self.M_ub, self.M_lb
        symbols = list(range(q))

        prob = cplex.Cplex()
        prob.set_log_stream(None)       # suppress CPLEX banner during build
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.set_error_stream(None)

        # Restore output streams before solving (re-enabled in solve())
        self._prob = prob

        # -- Objective: maximise sum_i y[i] --------------------------------
        prob.objective.set_sense(prob.objective.sense.maximize)

        # -- Declare all binary variables in one batch ----------------------
        # y[i]
        y_names = [self._yn(i) for i in range(M_ub)]
        prob.variables.add(
            names=y_names,
            types=[prob.variables.type.binary] * M_ub,
            obj=[1.0] * M_ub,           # coefficient in maximisation objective
        )

        # x[i,j,k]
        x_names = [self._xn(i, j, k)
                   for i in range(M_ub) for j in range(n) for k in symbols]
        prob.variables.add(
            names=x_names,
            types=[prob.variables.type.binary] * len(x_names),
        )

        # -- R1: sum_k x[i,j,k] == y[i]  for all i, j --------------------
        # Rearranged: sum_k x[i,j,k] - y[i] == 0
        for i in range(M_ub):
            for j in range(n):
                vars_  = [self._xn(i, j, k) for k in symbols] + [self._yn(i)]
                coeffs = [1.0] * q + [-1.0]
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=vars_, val=coeffs)],
                    senses=["E"],
                    rhs=[0.0],
                    names=[f"R1_{i}_{j}"],
                )

        # -- R2: sum_i y[i] >= M_lb ----------------------------------------
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self._yn(i) for i in range(M_ub)],
                val=[1.0] * M_ub,
            )],
            senses=["G"],
            rhs=[float(M_lb)],
            names=["R2"],
        )

        # -- Distance constraints -------------------------------------------
        # When M_lb == M_ub every slot is forced active; add ALL pair constraints
        # upfront so the LP relaxation is as tight as possible (no lazy callback).
        all_pairs = (self.M_lb == self.M_ub)
        if self.metric == "hamming":
            self._add_hamming_constraints(prob, symbols, all_pairs=all_pairs)
        else:
            self._add_lee_constraints(prob, symbols, all_pairs=all_pairs)

        # -- Symmetry breaking ----------------------------------------------
        self._add_symmetry_breaking(prob, symbols)

    def solve(self, time_limit=None):
        """
        Run CPLEX on the built model and extract the solution.

        Parameters
        ----------
        time_limit : float or None
            Wall-clock seconds given to CPLEX.  None = no limit.

        Returns
        -------
        status   : str   – "Optimal", "Feasible", "Infeasible", or "Unknown"
        obj_val  : int   – best M found (None if no feasible solution found)
        solution : list  – list of codeword strings (None if no solution found)
        runtime  : float – elapsed seconds
        """
        if self._prob is None:
            raise RuntimeError("Call build() before solve().")

        prob = self._prob

        # Re-enable output for the solve phase
        prob.set_log_stream(None)      # keep solver log quiet; remove to see B&B log
        prob.set_results_stream(None)

        if time_limit is not None:
            prob.parameters.timelimit.set(float(time_limit))

        # Register lazy constraint callback only when not all pairs are pre-constrained.
        # (When M_lb == M_ub the build step already added every pair upfront.)
        if self.M_lb != self.M_ub:
            cb = prob.register_callback(_DistanceLazyCallback)
            cb._M_ub    = self.M_ub
            cb._n       = self.n
            cb._q       = self.q
            cb._d       = self.d
            cb._metric  = self.metric
            cb._symbols = list(range(self.q))
            cb._y_names = [FleccWithCplex._yn(i) for i in range(self.M_ub)]
            if self.metric == "lee":
                _nz = [(k1, k2)
                       for k1 in range(self.q) for k2 in range(self.q) if k1 != k2]
                cb._nonzero_pairs = _nz
                cb._delta_cache   = {(k1, k2): lee_delta(k1, k2, self.q) for k1, k2 in _nz}

        t0 = timeit.default_timer()
        prob.solve()
        self.runtime = timeit.default_timer() - t0

        # Map CPLEX solution status to our status strings
        sol   = prob.solution
        stype = sol.get_status()
        # CPLEX MIP status codes:
        #   101/102/115/129/130 – optimal
        #   107/131/132         – time limit, feasible solution was found
        #   108                 – time limit, NO integer feasible solution found
        #                         (NOT proven infeasible – CPLEX just gave up)
        #   103/119/120         – truly proved infeasible
        OPTIMAL_CODES         = {101, 102, 115, 129, 130}
        FEASIBLE_CODES        = {107, 131, 132}
        TRUE_INFEASIBLE_CODES = {103, 119, 120}
        # 108 = time limit without any feasible solution found → "Unknown" for MIP
        # (the problem may still be feasible; CPLEX just didn't find it in time)

        if stype in OPTIMAL_CODES:
            self.status = "Optimal"
        elif stype in FEASIBLE_CODES:
            self.status = "Feasible"
        elif stype in TRUE_INFEASIBLE_CODES:
            self.status = "Infeasible"
        else:
            self.status = "Unknown"   # includes 108 (time limit, no sol)

        if self.status in {"Optimal", "Feasible"}:
            self.obj_val  = int(round(sol.get_objective_value()))
            self.solution = self._extract_solution()
        else:
            self.obj_val  = None
            self.solution = None

        return self.status, self.obj_val, self.solution, self.runtime

    # -- Private model-building helpers ----------------------------------------

    def _add_hamming_constraints(self, prob, symbols, all_pairs=False):
        """
        Hamming distance ILP constraints via McCormick linearisation.

        For i1 < i2 and each (j, k): introduce z_h[i1,i2,j,k] = x[i1,j,k]·x[i2,j,k].

        McCormick (three constraints):
          z <= x[i1,j,k]                         (Mc1)
          z <= x[i2,j,k]                         (Mc2)
          z >= x[i1,j,k] + x[i2,j,k] - 1        (Mc3)

        Distance constraint for every active pair (i1, i2):
          sum_{j,k} z_h[i1,i2,j,k]  <=  (n-d) + (2 - y[i1] - y[i2])·n

        When both active (y=1,1): RHS = n-d  ->  identical positions <= n-d  ->  distance >= d.
        When at least one inactive: z=0 by McCormick, constraint trivially satisfied.

        all_pairs=True : Add full McCormick + distance for every C(M_ub,2) pair upfront.
                          Used by the incremental strategy (M_lb==M_ub) where all slots
                          are active and the LP must be fully tight.
        all_pairs=False: Only seed pairs (adjacent + slot-0 anchored) get constraints;
                          remaining pairs handled lazily by _DistanceLazyCallback.
                          ALL z_h variables are still pre-allocated for the callback.
        """
        n, q, M_ub, d = self.n, self.q, self.M_ub, self.d

        # Batch-add ALL z_h variables (needed for both seed and lazy pairs).
        z_names = [
            self._zhn(i1, i2, j, k)
            for i1 in range(M_ub)
            for i2 in range(i1 + 1, M_ub)
            for j in range(n)
            for k in symbols
        ]
        if z_names:
            prob.variables.add(
                names=z_names,
                types=[prob.variables.type.binary] * len(z_names),
            )

        # Pairs that receive McCormick + distance constraints now.
        if all_pairs:
            constrained_pairs = sorted(
                (i1, i2)
                for i1 in range(M_ub) for i2 in range(i1 + 1, M_ub)
            )
        else:
            # Seed pairs: adjacent (i, i+1) PLUS all pairs anchored at slot 0.
            seed = set()
            for i in range(M_ub - 1):
                seed.add((i, i + 1))
            for i2 in range(1, M_ub):
                seed.add((0, i2))
            constrained_pairs = sorted(seed)

        lin_exprs, senses, rhs_vals, names = [], [], [], []

        for (i1, i2) in constrained_pairs:
            pair_z = []
            for j in range(n):
                for k in symbols:
                    z  = self._zhn(i1, i2, j, k)
                    x1 = self._xn(i1, j, k)
                    x2 = self._xn(i2, j, k)
                    pair_z.append(z)

                    # Mc1: z - x[i1,j,k] <= 0
                    lin_exprs.append(cplex.SparsePair(ind=[z, x1], val=[1.0, -1.0]))
                    senses.append("L"); rhs_vals.append(0.0)
                    names.append(f"Mc1h_{i1}_{i2}_{j}_{k}")

                    # Mc2: z - x[i2,j,k] <= 0
                    lin_exprs.append(cplex.SparsePair(ind=[z, x2], val=[1.0, -1.0]))
                    senses.append("L"); rhs_vals.append(0.0)
                    names.append(f"Mc2h_{i1}_{i2}_{j}_{k}")

                    # Mc3: z - x[i1,j,k] - x[i2,j,k] >= -1
                    lin_exprs.append(cplex.SparsePair(ind=[z, x1, x2], val=[1.0, -1.0, -1.0]))
                    senses.append("G"); rhs_vals.append(-1.0)
                    names.append(f"Mc3h_{i1}_{i2}_{j}_{k}")

            # Hamming distance constraint
            y1, y2 = self._yn(i1), self._yn(i2)
            lin_exprs.append(cplex.SparsePair(
                ind=pair_z + [y1, y2],
                val=[1.0] * len(pair_z) + [float(n), float(n)],
            ))
            senses.append("L")
            rhs_vals.append(float(3 * n - d))
            names.append(f"Ham_{i1}_{i2}")

        if lin_exprs:
            prob.linear_constraints.add(
                lin_expr=lin_exprs,
                senses=senses,
                rhs=rhs_vals,
                names=names,
            )

    def _add_lee_constraints(self, prob, symbols, all_pairs=False):
        """
        Lee distance ILP constraints via McCormick linearisation.

        For i1 < i2 and each (j, k1≠k2): introduce z_l = x[i1,j,k1]·x[i2,j,k2].
        Only k1 ≠ k2 pairs are created (delta_L(k,k)=0 contributes nothing).

        McCormick (three constraints per z_l):
          z_l <= x[i1,j,k1]
          z_l <= x[i2,j,k2]
          z_l >= x[i1,j,k1] + x[i2,j,k2] - 1

        Lee distance constraint for every active pair (i1, i2):
          sum_{j,k1≠k2} δ_L(k1,k2)·z_l[i1,i2,j,k1,k2]
              >= d - (2 - y[i1] - y[i2])·d
          i.e.  sum δ·z  -  d·y[i1]  -  d·y[i2]  >=  d - 2d  =  -d

        When both active: RHS = d  ->  Lee distance >= d.
        When at least one inactive: LHS = 0 >= -d (always satisfied for d > 0).

        all_pairs=True : Add full McCormick + distance for every C(M_ub,2) pair upfront.
        all_pairs=False: Only seed pairs (adjacent + slot-0 anchored) get constraints;
                          ALL z_l variables are still pre-allocated for the callback.
        """
        n, q, M_ub, d = self.n, self.q, self.M_ub, self.d

        # Symbol pairs with positive Lee distance only
        nonzero_pairs = [(k1, k2) for k1 in symbols for k2 in symbols if k1 != k2]
        delta = {(k1, k2): lee_delta(k1, k2, q) for k1, k2 in nonzero_pairs}

        # Batch-add ALL z_l variables (needed for both seed and lazy pairs).
        z_names = [
            self._zln(i1, i2, j, k1, k2)
            for i1 in range(M_ub)
            for i2 in range(i1 + 1, M_ub)
            for j in range(n)
            for k1, k2 in nonzero_pairs
        ]
        if z_names:
            prob.variables.add(
                names=z_names,
                types=[prob.variables.type.binary] * len(z_names),
            )

        # Pairs that receive McCormick + distance constraints now.
        if all_pairs:
            constrained_pairs = sorted(
                (i1, i2)
                for i1 in range(M_ub) for i2 in range(i1 + 1, M_ub)
            )
        else:
            # Seed pairs: adjacent (i, i+1) PLUS all pairs anchored at slot 0.
            seed = set()
            for i in range(M_ub - 1):
                seed.add((i, i + 1))
            for i2 in range(1, M_ub):
                seed.add((0, i2))
            constrained_pairs = sorted(seed)

        lin_exprs, senses, rhs_vals, names = [], [], [], []

        for (i1, i2) in constrained_pairs:
            dist_vars   = []
            dist_coeffs = []
            for j in range(n):
                for k1, k2 in nonzero_pairs:
                    z  = self._zln(i1, i2, j, k1, k2)
                    x1 = self._xn(i1, j, k1)
                    x2 = self._xn(i2, j, k2)
                    d_coef = float(delta[k1, k2])

                    # Mc1: z - x[i1,j,k1] <= 0
                    lin_exprs.append(cplex.SparsePair(ind=[z, x1], val=[1.0, -1.0]))
                    senses.append("L"); rhs_vals.append(0.0)
                    names.append(f"Mc1l_{i1}_{i2}_{j}_{k1}_{k2}")

                    # Mc2: z - x[i2,j,k2] <= 0
                    lin_exprs.append(cplex.SparsePair(ind=[z, x2], val=[1.0, -1.0]))
                    senses.append("L"); rhs_vals.append(0.0)
                    names.append(f"Mc2l_{i1}_{i2}_{j}_{k1}_{k2}")

                    # Mc3: z - x[i1,j,k1] - x[i2,j,k2] >= -1
                    lin_exprs.append(cplex.SparsePair(ind=[z, x1, x2], val=[1.0, -1.0, -1.0]))
                    senses.append("G"); rhs_vals.append(-1.0)
                    names.append(f"Mc3l_{i1}_{i2}_{j}_{k1}_{k2}")

                    dist_vars.append(z)
                    dist_coeffs.append(d_coef)

            # Lee distance constraint
            y1, y2 = self._yn(i1), self._yn(i2)
            lin_exprs.append(cplex.SparsePair(
                ind=dist_vars + [y1, y2],
                val=dist_coeffs + [-float(d), -float(d)],
            ))
            senses.append("G")
            rhs_vals.append(-float(d))
            names.append(f"Lee_{i1}_{i2}")

        if lin_exprs:
            prob.linear_constraints.add(
                lin_expr=lin_exprs,
                senses=senses,
                rhs=rhs_vals,
                names=names,
            )

    def _add_symmetry_breaking(self, prob, symbols):
        """
        Symmetry-breaking constraints to reduce the search space.

        (1) Packing:  y[i] >= y[i+1]
            Forces selected codewords into the lowest-indexed slots, eliminating
            permutation symmetry of inactive slots.

        (2) Lexicographic order:  w[i] <=_lex w[i+1]  for every consecutive pair.

            Auxiliary variables:
              eq[i,j]  ∈ {0,1}  = 1 iff y[i+1]=1 AND codewords i,i+1 are identical
                                   in all positions 0...j-1.
              eq[i,0]  = y[i+1]  (the chain is active only when slot i+1 is active)

            At each position j:
              val[i,j]   = sum_k k·x[i,j,k]   (symbol value, 0...q-1)
              val[i+1,j] = sum_k k·x[i+1,j,k]
              Constraint: val[i,j] - val[i+1,j]  <=  (1 - eq[i,j]) · q

            When eq[i,j] = 1: must have val[i,j] <= val[i+1,j]  ->  codeword i ≤ codeword i+1 here.
            When eq[i,j] = 0: RHS = q (max possible symbol value is q-1), trivially satisfied.

            Equality update (j = 0...n-2):
              f[i,j,k]   = x[i,j,k] AND x[i+1,j,k]      (McCormick)
              match[i,j] = sum_k f[i,j,k]                (1 iff symbols equal at j)
              eq[i,j+1]  = eq[i,j] AND match[i,j]        (McCormick AND of two binaries)
        """
        n, q, M_ub = self.n, self.q, self.M_ub

        lin_exprs, senses, rhs_vals, cnames = [], [], [], []

        # Pre-declare all lex auxiliary variables before adding constraints
        aux_var_names = []
        aux_var_types = []
        for i in range(M_ub - 1):
            # eq[i,j] for j = 0...n-1
            for j in range(n):
                aux_var_names.append(self._eqn(i, j))
                aux_var_types.append(prob.variables.type.binary)
            # For positions 0...n-2: f[i,j,k] and match[i,j]
            for j in range(n - 1):
                for k in symbols:
                    aux_var_names.append(self._fn(i, j, k))
                    aux_var_types.append(prob.variables.type.binary)
                aux_var_names.append(self._matchn(i, j))
                aux_var_types.append(prob.variables.type.binary)

        if aux_var_names:
            prob.variables.add(names=aux_var_names, types=aux_var_types)

        for i in range(M_ub - 1):
            y_i  = self._yn(i)
            y_i1 = self._yn(i + 1)

            # -- (1) Packing: y[i] - y[i+1] >= 0 --------------------------
            lin_exprs.append(cplex.SparsePair(ind=[y_i, y_i1], val=[1.0, -1.0]))
            senses.append("G"); rhs_vals.append(0.0)
            cnames.append(f"Pack_{i}")

            # -- (2) Lex initialisation: eq[i,0] = y[i+1] ------------------
            # Encoded as two inequalities:
            #   eq[i,0] - y[i+1] <= 0   (eq <= y_i1)
            #   eq[i,0] - y[i+1] >= 0   (eq >= y_i1)
            eq0 = self._eqn(i, 0)
            lin_exprs.append(cplex.SparsePair(ind=[eq0, y_i1], val=[1.0, -1.0]))
            senses.append("E"); rhs_vals.append(0.0)
            cnames.append(f"EqInit_{i}")

            for j in range(n):
                eq_j = self._eqn(i, j)

                # Lex ordering at position j:
                #   val[i,j] - val[i+1,j] - q·eq[i,j] <= q - (something)
                # Simplified by moving eq to RHS:
                #   sum_k k·x[i,j,k]  -  sum_k k·x[i+1,j,k]  -  q·eq[i,j]  <=  0
                # (When eq=1: val_i - val_{i+1} <= 0, i.e. val_i <= val_{i+1}.)
                # (When eq=0: val_i - val_{i+1} <= q, trivially true since max diff = q-1.)
                # Rewritten:  val_i - val_{i+1} <= q·(1 - eq[i,j])
                # i.e.        val_i - val_{i+1} + q·eq[i,j] <= q
                x_i_j_vars    = [self._xn(i,     j, k) for k in symbols]
                x_i1_j_vars   = [self._xn(i + 1, j, k) for k in symbols]
                coeffs_i      = [float(k) for k in symbols]
                coeffs_i1     = [-float(k) for k in symbols]
                lin_exprs.append(cplex.SparsePair(
                    ind=x_i_j_vars + x_i1_j_vars + [eq_j],
                    val=coeffs_i   + coeffs_i1   + [float(q)],
                ))
                senses.append("L"); rhs_vals.append(float(q))
                cnames.append(f"Lex_{i}_{j}")

                # Equality propagation (only needed for positions 0...n-2)
                if j < n - 1:
                    eq_next = self._eqn(i, j + 1)

                    # f[i,j,k] = x[i,j,k] AND x[i+1,j,k]  (McCormick)
                    f_vars = []
                    for k in symbols:
                        fv = self._fn(i, j, k)
                        x1 = self._xn(i,     j, k)
                        x2 = self._xn(i + 1, j, k)
                        f_vars.append(fv)

                        # fv <= x1
                        lin_exprs.append(cplex.SparsePair(ind=[fv, x1], val=[1.0, -1.0]))
                        senses.append("L"); rhs_vals.append(0.0)
                        cnames.append(f"Mc1f_{i}_{j}_{k}")

                        # fv <= x2
                        lin_exprs.append(cplex.SparsePair(ind=[fv, x2], val=[1.0, -1.0]))
                        senses.append("L"); rhs_vals.append(0.0)
                        cnames.append(f"Mc2f_{i}_{j}_{k}")

                        # fv >= x1 + x2 - 1
                        lin_exprs.append(cplex.SparsePair(ind=[fv, x1, x2], val=[1.0, -1.0, -1.0]))
                        senses.append("G"); rhs_vals.append(-1.0)
                        cnames.append(f"Mc3f_{i}_{j}_{k}")

                    # match[i,j] = sum_k f[i,j,k]
                    # Encoded as equality: match - sum f = 0
                    match_v = self._matchn(i, j)
                    lin_exprs.append(cplex.SparsePair(
                        ind=[match_v] + f_vars,
                        val=[1.0]     + [-1.0] * len(f_vars),
                    ))
                    senses.append("E"); rhs_vals.append(0.0)
                    cnames.append(f"Match_{i}_{j}")

                    # eq[i,j+1] = eq[i,j] AND match[i,j]  – McCormick AND
                    # eq_next <= eq_j
                    lin_exprs.append(cplex.SparsePair(ind=[eq_next, eq_j], val=[1.0, -1.0]))
                    senses.append("L"); rhs_vals.append(0.0)
                    cnames.append(f"Eq1_{i}_{j}")

                    # eq_next <= match_v
                    lin_exprs.append(cplex.SparsePair(ind=[eq_next, match_v], val=[1.0, -1.0]))
                    senses.append("L"); rhs_vals.append(0.0)
                    cnames.append(f"Eq2_{i}_{j}")

                    # eq_next >= eq_j + match_v - 1
                    lin_exprs.append(cplex.SparsePair(
                        ind=[eq_next, eq_j, match_v],
                        val=[1.0, -1.0, -1.0],
                    ))
                    senses.append("G"); rhs_vals.append(-1.0)
                    cnames.append(f"Eq3_{i}_{j}")

        if lin_exprs:
            prob.linear_constraints.add(
                lin_expr=lin_exprs,
                senses=senses,
                rhs=rhs_vals,
                names=cnames,
            )

    def _extract_solution(self):
        """Read selected codeword strings from the solved model."""
        prob = self._prob
        sol  = prob.solution
        n, M_ub, q = self.n, self.M_ub, self.q
        solution = []
        for i in range(M_ub):
            if sol.get_values(self._yn(i)) > 0.5:   # slot i is active
                word = []
                for j in range(n):
                    for k in range(q):
                        if sol.get_values(self._xn(i, j, k)) > 0.5:
                            word.append(str(k))
                            break
                solution.append("".join(word))
        return solution


# --- Top-level solver interface ------------------------------------------------

def solve_flecc_cplex(
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
    Solve one FLECC instance with CPLEX and optionally write results to Excel.

    Parameters
    ----------
    n          : codeword length
    q          : alphabet size (symbols 0...q-1)
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
    # -- Estimate M_ub when not provided ------------------------------------
    if M_ub is None:
        M_ub = estimate_M_ub(n, q, d, metric)
    M_ub = max(M_ub, M_lb)

    print(f"\n{'='*70}")
    print("FLECC ILP Solver (CPLEX)")
    print(f"{'='*70}")
    print(f"  n={n}, q={q}, d={d}, metric={metric}")
    print(f"  M_lb={M_lb}, M_ub (candidate slots)={M_ub}")
    if time_limit is not None:
        print(f"  Time limit: {time_limit}s")
    print(f"{'='*70}\n")

    # -- Build model --------------------------------------------------------
    solver = FleccWithCplex(n=n, q=q, d=d, M_ub=M_ub, M_lb=M_lb, metric=metric)
    solver.build()

    # Allow CPLEX log output during solve
    solver._prob.set_log_stream(None)   # set to None to suppress, or remove line for verbose

    # -- Solve --------------------------------------------------------------
    status, obj_val, solution, runtime = solver.solve(time_limit=time_limit)

    # -- Console report -----------------------------------------------------
    print(f"\nStatus  : {status}")
    print(f"Best M  : {obj_val}")
    print(f"Runtime : {runtime:.4f}s")
    if solution:
        print(f"Codewords ({len(solution)}):")
        for cw in solution:
            print(f"  {cw}")

    # -- Build result row ---------------------------------------------------
    instance_name = f"FLECC_{n}_{d}_{metric}_q{q}_cplex"
    result_row = {
        "Instance":   instance_name,
        "n":          n,
        "q":          q,
        "d":          d,
        "Metric":     metric,
        "M_lb":       M_lb,
        "M_ub":       M_ub,
        "M_best":     obj_val,
        "Status":     status,
        "Runtime(s)": round(runtime, 4),
        "Codewords":  str(solution) if solution else "None",
        "Method":     "ILP-CPLEX",
    }

    # -- Save to Excel ------------------------------------------------------
    if not test:
        results_df = pd.DataFrame([result_row])
        sheet = _append_df_to_excel_sheet(output_file, results_df, metric)
        print(f"\nResults saved  -> {output_file}  (sheet: {sheet})")

        summary_row = {
            "Instance": instance_name,
            "n":        n,
            "q":        q,
            "d":        d,
            "M_best":   obj_val,
            "UB":       M_ub,
            "Time(s)":  round(runtime, 4),
            "Status":   status,
            "Method":   "ILP-CPLEX",
        }
        s_sheet = _append_df_to_excel_sheet(
            DEFAULT_SUMMARY_FILE, pd.DataFrame([summary_row]), metric
        )
        print(f"Summary saved  -> {DEFAULT_SUMMARY_FILE}  (sheet: {s_sheet})")

    return status, obj_val, solution, result_row


def solve_flecc_cplex_incremental(
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
    Find the maximum feasible M by iterative search: test M = M_lb, M_lb+1, ...
    until the sub-problem becomes infeasible or the time budget is exhausted.

    Each sub-problem is :class:`FleccWithCplex` with ``M_ub = M_lb = M``, i.e.
    *exactly* M codeword slots all forced active.  This is far more efficient
    than the direct-maximise approach with a large M_ub because:

    * Each model is smaller  – C(M, 2) pairs vs C(M_ub, 2).
    * The LP relaxation is tighter – all y[i] = 1 removes big-M slack from
      every distance constraint, so the LP is essentially the fractional version
      of the exact M-codeword problem.
    * CPLEX finds incumbents much more quickly, triggering the lazy callback
      enough times to prove feasibility or infeasibility.

    Termination:
    * ``Infeasible`` for step M  ->  M-1 is optimal, stop.
    * Time budget exhausted      ->  last feasible M is **lower bound** (Feasible).
    * All steps up to M_ub pass  ->  M_ub is optimal.

    Parameters
    ----------
    n, q, d, M_lb, M_ub, metric : same as :func:`solve_flecc_cplex`
    time_limit : total wall-clock budget across ALL steps (None = unlimited)
    output_file, test            : Excel output control

    Returns
    -------
    (status, obj_val, solution, result_row)
      status  : "Optimal" | "Feasible" | "Infeasible"
      obj_val : best M found (None if nothing feasible)
      solution, result_row : same structure as :func:`solve_flecc_cplex`
    """
    if M_ub is None:
        M_ub = estimate_M_ub(n, q, d, metric)
    M_ub = max(M_ub, M_lb)

    print(f"\n{'='*70}")
    print("FLECC ILP Solver (CPLEX) — Incremental strategy")
    print(f"{'='*70}")
    print(f"  n={n}, q={q}, d={d}, metric={metric}")
    print(f"  Search range: M = {M_lb} ... {M_ub}")
    if time_limit is not None:
        print(f"  Total time budget: {time_limit}s")
    print(f"{'='*70}\n")

    global_start  = timeit.default_timer()
    best_M        = None
    best_solution = None
    final_status  = "Infeasible"
    step_statuses = []

    for M in range(M_lb, M_ub + 1):
        elapsed   = timeit.default_timer() - global_start
        remaining = (time_limit - elapsed) if time_limit is not None else None
        if remaining is not None and remaining <= 0:
            print(f"  [M={M}] Time budget exhausted before this step.")
            final_status = "Feasible" if best_M is not None else "Infeasible"
            break

        print(f"  [M={M}] Building model ({M} slots, all active)...", end=" ", flush=True)
        solver = FleccWithCplex(n=n, q=q, d=d, M_ub=M, M_lb=M, metric=metric)
        solver.build()
        status, obj_val, solution, rt = solver.solve(time_limit=remaining)
        print(f"=> {status}  M_found={obj_val}  ({rt:.2f}s)")
        step_statuses.append((M, status, obj_val, rt))

        if status == "Infeasible":
            # CPLEX proved M is truly infeasible -> M-1 is the optimal answer
            final_status = "Optimal" if best_M is not None else "Infeasible"
            break
        elif status in ("Optimal", "Feasible") and obj_val is not None and obj_val >= M:
            best_M        = M
            best_solution = solution
            final_status  = "Optimal" if status == "Optimal" else "Feasible"
        elif status == "Unknown":
            # Time limit hit before finding any solution for this M
            # (CPLEX status 108: gave up, not proven infeasible)
            # Conservative: report best_M so far as a lower bound
            print(f"  [M={M}] Time limit hit without finding a solution — "
                  f"cannot determine if M={M} is feasible.")
            final_status = "Feasible" if best_M is not None else "Infeasible"
            break
        else:
            # Unexpected: solver found fewer codewords than M
            final_status = "Feasible" if best_M is not None else "Infeasible"
            break
    else:
        # Completed all M values without proving M_ub+1 infeasible
        final_status = "Optimal" if best_M is not None else "Infeasible"

    total_runtime = timeit.default_timer() - global_start

    # -- Console report -----------------------------------------------------
    print(f"\n{'-'*70}")
    print(f"Status  : {final_status}")
    print(f"Best M  : {best_M}")
    print(f"Runtime : {total_runtime:.4f}s (total)")
    if best_solution:
        print(f"Codewords ({len(best_solution)}):")
        for cw in best_solution:
            print(f"  {cw}")

    # -- Build result row ---------------------------------------------------
    instance_name = f"FLECC_{n}_{d}_{metric}_q{q}_cplex_incr"
    result_row = {
        "Instance":   instance_name,
        "n":          n,
        "q":          q,
        "d":          d,
        "Metric":     metric,
        "M_lb":       M_lb,
        "M_ub":       M_ub,
        "M_best":     best_M,
        "Status":     final_status,
        "Runtime(s)": round(total_runtime, 4),
        "Codewords":  str(best_solution) if best_solution else "None",
        "Method":     "ILP-CPLEX-Incremental",
    }

    # -- Save to Excel ------------------------------------------------------
    if not test:
        results_df = pd.DataFrame([result_row])
        sheet = _append_df_to_excel_sheet(output_file, results_df, metric)
        print(f"\nResults saved  -> {output_file}  (sheet: {sheet})")

        summary_row = {
            "Instance": instance_name,
            "n":        n,
            "q":        q,
            "d":        d,
            "M_best":   best_M,
            "UB":       M_ub,
            "Time(s)":  round(total_runtime, 4),
            "Status":   final_status,
            "Method":   "ILP-CPLEX-Incremental",
        }
        s_sheet = _append_df_to_excel_sheet(
            DEFAULT_SUMMARY_FILE, pd.DataFrame([summary_row]), metric
        )
        print(f"Summary saved  -> {DEFAULT_SUMMARY_FILE}  (sheet: {s_sheet})")

    return final_status, best_M, best_solution, result_row


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Solve the FLECC problem as an ILP using CPLEX (maximises codewords).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n",          type=int,   required=True,
                        help="Codeword length")
    parser.add_argument("--q",          type=int,   default=2,
                        help="Alphabet size (symbols 0...q-1)")
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
    parser.add_argument(
        "--strategy",
        type=str,
        default="incremental",
        choices=["incremental", "direct"],
        help=(
            "incremental: linear search M=M_lb...M_ub, each sub-problem has exactly M "
            "slots (recommended, much faster for Lee).  "
            "direct: single maximisation solve over M_ub candidate slots."
        ),
    )

    args = parser.parse_args()

    time_limit = args.time_limit if args.time_limit > 0 else None

    common = dict(
        n           = args.n,
        q           = args.q,
        d           = args.d,
        M_lb        = args.M_lb,
        M_ub        = args.M_ub,
        metric      = args.metric,
        time_limit  = time_limit,
        output_file = args.output,
        test        = args.test,
    )

    if args.strategy == "incremental":
        solve_flecc_cplex_incremental(**common)
    else:
        solve_flecc_cplex(**common)



if __name__ == "__main__":
    main()
