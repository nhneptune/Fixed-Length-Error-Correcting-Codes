"""
flecc_with_sat_multisetlex.py – Multiset-Row + Lex-Column Symmetry Breaking
           for FleccWithSatIncremental

Combines two families of symmetry-breaking predicates:

  ROW multiset ordering  (>=_m R):
    multiset(codeword[i]) >=_multiset multiset(codeword[i+1])
    for every consecutive row pair i < i+1.

    Multiset comparison is defined by sorting each row's symbols in
    non-increasing (descending) order and comparing the resulting
    sequences lexicographically:

        sorted_desc(row_i) >=_lex sorted_desc(row_{i+1})

    Equivalently, comparing the frequency vectors of the two rows from
    the largest symbol downward: symbol q-1 first, then q-2, ..., 0.

    SAT encoding uses ORDER ENCODING of per-symbol frequencies:
        cnt_geq[i, s, v]  =  True  iff  #{j : row_i[j] = s} >= v

    The sorted-descending sequence of row i is:
        v=1,2,...,n for s=q-1 (up to frequency of q-1)
        v=1,2,...,n for s=q-2
        ...
        v=1,2,...,n for s=0

    All values cnt_geq[i, s, v] form a binary sequence representing
    sorted_desc(row_i).  We then enforce:

        sorted_desc(row_{i+1}) <=_lex sorted_desc(row_i)

    using the standard equality-prefix chain encoding.

    The cnt_geq variables are encoded using a Sinz sequential counter
    (clausal, no PBLib) over the per-row one-hot symbol variables.

    The row SB is guarded by p_{i+2} (activation of row i+1), so the
    constraint is vacuously satisfied when row i+1 is inactive.

  COLUMN lex ordering  (<=_lex C):
    col[j] <=_lex col[j+1]   for every consecutive column pair.
    (Identical to double-lex / snake-lex column SB.)

  FIX FIRST CODEWORD (Lee metric only):
    codeword[0] = (0, ..., 0) -- valid by Lee distance translation invariance.

Usage
-----
    solver = FleccWithSatMultiSetLex(alphabet_size=4)
    solver.initialize_base_constraints(n, d, "hamming", max_possible_codewords=20)
    # incremental solving ...

Or via the top-level function:

    solve_flecc_multi_sat_incremental_multisetlex(n, d, m, metric, ...)

CLI quick-test:

    python -m flecc_with_sat_multisetlex --n 6 --d 3 --m 2 --q 2 \
           --metric hamming --timeout 30 --test --validate
"""

from flecc_with_sat import (
    FleccWithSatIncremental,
    get_cp_mip_c_multiplier_placeholder,
    compute_upper_bound_max_possible_codewords,
    append_results_to_excel_by_metric_sheet,
    append_summary_to_excel,
    validate_codewords,
)
import timeit
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: guarded clause addition
# ---------------------------------------------------------------------------

def _add_guarded(solver, clause, guard_var):
    """Add *clause* to solver; if guard_var is set, prepend -guard_var."""
    if guard_var is not None:
        solver._add_solver_clause([-guard_var] + list(clause))
    else:
        solver._add_solver_clause(list(clause))


# ---------------------------------------------------------------------------
# Helper: standard lex-<=  CNF encoding (equality-prefix chain)
# ---------------------------------------------------------------------------

def _encode_lex_leq(solver, alphabet, seq_a_vars, seq_b_vars, guard_var=None):
    """Add clauses enforcing seq_A <=_lex seq_B to *solver*.

    Parameters
    ----------
    solver : FleccWithSatMultiSetLex
        Must expose allocate_variables() and _add_solver_clause().
    alphabet : tuple of str
        Ordered alphabet symbols.
    seq_a_vars : list[dict[str, int]]
        One-hot variable dicts for each element of sequence A.
    seq_b_vars : list[dict[str, int]]
        One-hot variable dicts for each element of sequence B.
    guard_var : int or None
        When not None every clause is guarded by -guard_var.
    """
    assert len(seq_a_vars) == len(seq_b_vars)
    n = len(seq_a_vars)
    if n == 0:
        return

    eq_prev = solver.allocate_variables(1)
    _add_guarded(solver, [eq_prev], guard_var)          # eq_prev is True initially

    for j in range(n):
        a_vars = seq_a_vars[j]
        b_vars = seq_b_vars[j]

        # Forbid: eq_prev AND A[j] > B[j].
        for idx_a, sym_a in enumerate(alphabet):
            for sym_b in alphabet[:idx_a]:              # sym_b < sym_a
                _add_guarded(
                    solver,
                    [-eq_prev, -a_vars[sym_a], -b_vars[sym_b]],
                    guard_var,
                )

        if j < n - 1:
            eq_next = solver.allocate_variables(1)
            _add_guarded(solver, [-eq_next, eq_prev], guard_var)    # eq_next -> eq_prev

            # eq_pos: True iff A[j] == B[j]  (same symbol chosen in both)
            eq_pos = solver.allocate_variables(1)
            pair_vars = []
            for sym in alphabet:
                _add_guarded(solver, [-a_vars[sym], -b_vars[sym], eq_pos], guard_var)
                pv = solver.allocate_variables(1)
                _add_guarded(solver, [-pv, a_vars[sym]], guard_var)
                _add_guarded(solver, [-pv, b_vars[sym]], guard_var)
                _add_guarded(solver, [-a_vars[sym], -b_vars[sym], pv], guard_var)
                pair_vars.append(pv)
            _add_guarded(solver, [-eq_pos] + pair_vars, guard_var)  # eq_pos -> OR(pair_vars)

            _add_guarded(solver, [-eq_next, eq_pos], guard_var)             # eq_next -> eq_pos
            _add_guarded(solver, [-eq_prev, -eq_pos, eq_next], guard_var)   # both -> eq_next
            eq_prev = eq_next


# ---------------------------------------------------------------------------
# Helper: encode cnt_geq variables for one row's symbol counts (Sinz counter)
# ---------------------------------------------------------------------------

def _encode_row_count_geq_vars(solver, row_idx, alphabet, n):
    """Create cnt_geq[s][v-1] variables for row_idx using a Sinz sequential counter.

    cnt_geq[s][v-1] = True  iff  #{j : codeword[row_idx][j] == s} >= v
    for v = 1..n, for each symbol s in alphabet.

    Encoding uses a sequential counter (Sinz 2005):
      r[j][k] = True iff among x_0..x_{j-1} at least k are True.

    CNF rules for symbol s:
      (1) r[j-1][k] -> r[j][k]                        (pass-through)
      (2) r[j-1][k-1] AND x_{j-1} -> r[j][k]          (new count)
      (3) r[j][k] -> r[j-1][k] OR x_{j-1}             (backward, k < j)
          r[j][j] -> x_{j-1}  AND  r[j][j] -> r[j-1][j-1]  (backward, k==j)

    Returns
    -------
    cnt_geq : dict  symbol -> list[int]
        cnt_geq[s][v-1] is the SAT variable for count(s) >= v, for v=1..n.
    """
    codeword_vars = solver.codeword_vars
    cnt_geq = {}

    for s in alphabet:
        x_vars = [codeword_vars[(row_idx, j, s)] for j in range(n)]

        # Allocate r_var[(j, k)] for 1 <= k <= j <= n
        r_var = {}
        for j in range(1, n + 1):
            for k in range(1, j + 1):
                r_var[(j, k)] = solver.allocate_variables(1)

        for j in range(1, n + 1):
            x_prev = x_vars[j - 1]     # x_{j-1} (0-indexed)

            for k in range(1, j + 1):
                r_jk = r_var[(j, k)]

                # (1) Pass-through: r[j-1][k] -> r[j][k]
                if k < j:
                    solver._add_solver_clause([-r_var[(j - 1, k)], r_jk])

                # (2) New count
                if k == 1:
                    # r[j-1][0] = True (constant): x_{j-1} -> r[j][k]
                    solver._add_solver_clause([-x_prev, r_jk])
                else:
                    solver._add_solver_clause([-r_var[(j - 1, k - 1)], -x_prev, r_jk])

                # (3) Backward sourcing
                if k < j:
                    solver._add_solver_clause([-r_jk, r_var[(j - 1, k)], x_prev])
                else:
                    # k == j: can only come from (r[j-1][j-1] AND x_{j-1})
                    solver._add_solver_clause([-r_jk, x_prev])
                    if k >= 2:
                        solver._add_solver_clause([-r_jk, r_var[(j - 1, k - 1)]])

        cnt_geq[s] = [r_var[(n, v)] for v in range(1, n + 1)]

    return cnt_geq


# ---------------------------------------------------------------------------
# Subclass with multiset-row + lex-column symmetry breaking
# ---------------------------------------------------------------------------

class FleccWithSatMultiSetLex(FleccWithSatIncremental):
    """FleccWithSatIncremental with multiset-row + lex-column SB.

    ROW multiset ordering:
        sorted_desc(codeword[i]) >=_lex sorted_desc(codeword[i+1])
        encoded using per-symbol Sinz sequential counter variables.

    COLUMN lex ordering:
        col[j] <=_lex col[j+1]   (standard lex column SB).

    FIX FIRST CODEWORD (Lee metric only):
        codeword[0] = (0, ..., 0).
    """

    def __init__(
        self,
        alphabet_size=2,
        row_symmetry_breaking=True,
        col_symmetry_breaking=True,
        fix_first_codeword=True,
    ):
        super().__init__(alphabet_size=alphabet_size)
        self.row_symmetry_breaking = row_symmetry_breaking
        self.col_symmetry_breaking = col_symmetry_breaking
        self.fix_first_codeword = fix_first_codeword
        # _cnt_geq[(row_idx, s)][v-1] = SAT var for "count(s, row_idx) >= v"
        self._cnt_geq = {}

    # ------------------------------------------------------------------
    # Override: build cnt_geq vars + column SB + optional Lee fix
    # ------------------------------------------------------------------

    def initialize_base_constraints(
        self,
        length_of_codeword,
        distance_threshold,
        distance_metric,
        max_possible_codewords=None,
    ):
        super().initialize_base_constraints(
            length_of_codeword,
            distance_threshold,
            distance_metric,
            max_possible_codewords=max_possible_codewords,
        )
        if self.row_symmetry_breaking:
            n = self.length_of_codeword
            for row in range(self.max_possible_codewords):
                cnt = _encode_row_count_geq_vars(self, row, self.alphabet, n)
                for s in self.alphabet:
                    self._cnt_geq[(row, s)] = cnt[s]

        if self.col_symmetry_breaking:
            self._add_all_column_lex_constraints()

        if self.fix_first_codeword and distance_metric == "lee":
            self._add_fix_first_codeword_zeros()

    # ------------------------------------------------------------------
    # Lee symmetry: fix codeword[0] = (0, ..., 0)
    # ------------------------------------------------------------------

    def _add_fix_first_codeword_zeros(self):
        """Force codeword[0] = all-zeros when active (guarded by p_1)."""
        p1 = self.count_activation_vars[1]
        first_sym = self.alphabet[0]
        for j in range(self.length_of_codeword):
            var = self.codeword_vars[(0, j, first_sym)]
            self._add_solver_clause([-p1, var])

    # ------------------------------------------------------------------
    # Override: add row multiset SB when a new codeword slot is introduced
    # ------------------------------------------------------------------

    def add_distance_constraints_for_codeword(self, codeword_index):
        super().add_distance_constraints_for_codeword(codeword_index)
        if self.row_symmetry_breaking and codeword_index > 0:
            self._add_row_multiset_constraint(codeword_index - 1, codeword_index)

    # ------------------------------------------------------------------
    # Row multiset ordering: sorted_desc(row_a) >=_lex sorted_desc(row_b)
    # i.e.  sorted_desc(row_b) <=_lex sorted_desc(row_a)
    # ------------------------------------------------------------------

    def _multiset_seq(self, row_idx):
        """Binary one-hot sequence representing sorted_desc(row_idx).

        The sequence is cnt_geq[s, v] for s in reversed(alphabet), v = 1..n.
        Each element is a one-hot dict over binary alphabet ('0', '1'):
            '1' -> cnt_geq variable (True iff count >= v)
            '0' -> complement variable
        """
        n = self.length_of_codeword
        seq = []
        for s in reversed(self.alphabet):       # q-1 down to 0
            cnt_vars = self._cnt_geq[(row_idx, s)]
            for v in range(n):                  # v = 1..n (stored at index v-1..n-1)
                bit_var = cnt_vars[v]
                not_bit_var = self.allocate_variables(1)
                # not_bit_var <-> NOT bit_var (explicit complement)
                self._add_solver_clause([bit_var, not_bit_var])
                self._add_solver_clause([-bit_var, -not_bit_var])
                seq.append({'0': not_bit_var, '1': bit_var})
        return seq

    def _add_row_multiset_constraint(self, row_a, row_b):
        """Enforce multiset(row_a) >=_m multiset(row_b).

        Encoded as: sorted_desc(row_b) <=_lex sorted_desc(row_a).
        Guarded by p_{row_b+1} so the constraint is vacuous for inactive rows.
        """
        guard = self.count_activation_vars[row_b + 1]
        seq_a = self._multiset_seq(row_a)
        seq_b = self._multiset_seq(row_b)
        # Enforce seq_b <=_lex seq_a
        _encode_lex_leq(self, ('0', '1'), seq_b, seq_a, guard_var=guard)

    # ------------------------------------------------------------------
    # Column lex ordering: col[j] <=_lex col[j+1]
    # ------------------------------------------------------------------

    def _add_all_column_lex_constraints(self):
        """Enforce col[j] <=_lex col[j+1] for every consecutive column pair."""
        M = self.max_possible_codewords
        n = self.length_of_codeword
        for j in range(n - 1):
            seq_a = [
                {sym: self.codeword_vars[(row, j, sym)] for sym in self.alphabet}
                for row in range(M)
            ]
            seq_b = [
                {sym: self.codeword_vars[(row, j + 1, sym)] for sym in self.alphabet}
                for row in range(M)
            ]
            _encode_lex_leq(self, self.alphabet, seq_a, seq_b, guard_var=None)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def solve_flecc_multi_sat_incremental_multisetlex(
    length_of_codeword,
    distance_threshold,
    number_of_codewords,
    distance_metric="hamming",
    alphabet_size=2,
    test=False,
    validate=False,
    timeout=600,
    max_timeout_retries=1,
    row_symmetry_breaking=True,
    col_symmetry_breaking=True,
    fix_first_codeword=True,
    max_vars=1_000_000,
    instance_name=None,
):
    """Solve FLECC with incremental SAT + multiset-row + lex-column SB.

    Parameters
    ----------
    length_of_codeword : int
    distance_threshold : int
    number_of_codewords : int
        Lower bound on the number of codewords to start from.
    distance_metric : str
        'hamming' or 'lee'.
    alphabet_size : int
    test : bool
        If True, skip all Excel writes.
    validate : bool
        If True, verify each solution's pairwise distances.
    timeout : float
        Per-run global timeout in seconds.
    max_timeout_retries : int
    row_symmetry_breaking : bool
        Enforce multiset row ordering.
    col_symmetry_breaking : bool
        Enforce lex column ordering.
    fix_first_codeword : bool
        (Lee only) Fix codeword[0] = (0, ..., 0).
    max_vars : int or None
        Cap on SAT variables to prevent OOM.
    instance_name : str or None
        Label used in Excel output.

    Returns
    -------
    (max_codewords_found, best_solution, all_results)
    """
    print(f"\n{'='*70}")
    print("Starting Incremental Multi-SAT Solver with MultiSet-Lex Symmetry Breaking")
    print(f"{'='*70}")
    print(f"Lower bound (initial number of codewords): {number_of_codewords}")
    print(f"Codeword length: {length_of_codeword}, Distance threshold: {distance_threshold}")
    print(f"Distance metric: {distance_metric}")
    print(f"Row multiset SB: {row_symmetry_breaking}, Column lex SB: {col_symmetry_breaking}")
    fix_fc_active = fix_first_codeword and distance_metric == "lee"
    print(f"Fix first codeword = zeros (Lee only): {fix_fc_active}")
    if timeout is not None:
        print(f"Global timeout: {timeout}s (shared across all iterations)")
    print("Learned clauses and SB constraints preserved between iterations\n")

    flecc_solver = FleccWithSatMultiSetLex(
        alphabet_size=alphabet_size,
        row_symmetry_breaking=row_symmetry_breaking,
        col_symmetry_breaking=col_symmetry_breaking,
        fix_first_codeword=fix_first_codeword,
    )

    c_multiplier = get_cp_mip_c_multiplier_placeholder()
    ub_max_possible_codewords = compute_upper_bound_max_possible_codewords(
        alphabet_size=len(flecc_solver.alphabet),
        length_of_codeword=length_of_codeword,
        distance_threshold=distance_threshold,
        requested_codewords=number_of_codewords,
        distance_metric=distance_metric,
        c_multiplier=c_multiplier,
    )
    if number_of_codewords > ub_max_possible_codewords:
        print(f"M_lb={number_of_codewords} > UB={ub_max_possible_codewords}, resetting to start from M=2")
        number_of_codewords = 2

    if timeout is not None and timeout > 0:
        iteration_budget = max(20, int(timeout))
    else:
        iteration_budget = 100
    trajectory_cap = number_of_codewords + iteration_budget
    vars_per_codeword = length_of_codeword * alphabet_size
    mem_cap = (
        max(number_of_codewords, max_vars // max(1, vars_per_codeword * 10))
        if max_vars is not None
        else ub_max_possible_codewords
    )
    max_possible_codewords = max(
        number_of_codewords,
        min(ub_max_possible_codewords, trajectory_cap, mem_cap),
    )
    memory_limited = max_possible_codewords < ub_max_possible_codewords
    print(
        f"Upper bound estimate: {ub_max_possible_codewords}, "
        f"trajectory cap: {trajectory_cap}, using: {max_possible_codewords}"
    )
    if memory_limited:
        print(
            f"Memory cap applied (max_vars={max_vars}): "
            f"max M limited to {max_possible_codewords} (UB was {ub_max_possible_codewords})"
        )

    global_start = timeit.default_timer()
    flecc_solver.initialize_base_constraints(
        length_of_codeword,
        distance_threshold,
        distance_metric,
        max_possible_codewords=max_possible_codewords,
    )

    current_num_codewords = number_of_codewords
    max_codewords_found = None
    best_solution = None
    all_results = []
    iteration = 0
    summary_status = "Optimal"

    while True:
        if current_num_codewords > max_possible_codewords:
            if memory_limited:
                summary_status = "MemoryLimit"
            print(f"\n{'='*70}")
            if memory_limited:
                print(
                    f"Search Complete -- memory limit (max_vars={max_vars}): "
                    f"M capped at {max_possible_codewords}."
                )
            else:
                print("Search Complete -- reached upper bound estimate.")
            print(f"Maximum codewords found: {max_codewords_found}")
            if best_solution:
                print(f"Best solution: {best_solution['codewords']}")
            print(f"Total iterations: {iteration}")
            print(f"{'='*70}\n")
            break

        iteration += 1
        current_timeout_retries = 0
        timeout_occurred = False

        while current_timeout_retries <= max_timeout_retries:
            print(
                f"[Iteration {iteration}] Trying {current_num_codewords} codewords...",
                end=" ", flush=True,
            )

            start = timeit.default_timer()
            flecc_solver.solve_incremental(
                current_num_codewords, timeout, _global_start=global_start
            )
            stop = timeit.default_timer()
            runtime = stop - start

            is_sat = flecc_solver.solution is not None
            status = "SAT" if is_sat else "UNSAT"

            if flecc_solver.timeout_occurred:
                timeout_occurred = True
                current_timeout_retries += 1
                if current_timeout_retries <= max_timeout_retries:
                    print(
                        f"TIMEOUT ({runtime:.2f}s) -- retrying "
                        f"({current_timeout_retries}/{max_timeout_retries})"
                    )
                    continue
                else:
                    print(f"TIMEOUT ({runtime:.2f}s) -- max retries reached.")
                    status = "TIMEOUT"
                    is_sat = False
            else:
                print(f"{status} ({runtime:.2f}s)")
                break

        if timeout_occurred and current_timeout_retries > max_timeout_retries:
            status = "TIMEOUT"

        _instance_label = instance_name if instance_name else (
            f"FLECC_{length_of_codeword}_{distance_threshold}_"
            f"{current_num_codewords}_{distance_metric}_incremental_multisetlex"
        )
        codewords_str = str(flecc_solver.solution) if flecc_solver.solution else "None"

        result = {
            "Iteration": iteration,
            "Instance": _instance_label,
            "Num_Codewords": current_num_codewords,
            "Variables": flecc_solver.variables_count,
            "Clauses": flecc_solver.clauses_count,
            "Runtime": round(timeit.default_timer() - global_start, 4),
            "Codewords": codewords_str,
            "Status": status,
        }
        all_results.append(result)

        if not test:
            excel_file = "FLECC_MultiSAT_Incremental_MultiSetLex.xlsx"
            results_df = pd.DataFrame([result])
            sheet_name = append_results_to_excel_by_metric_sheet(
                excel_file, results_df, distance_metric
            )
            print(f"Iteration {iteration} saved to {excel_file} (sheet: {sheet_name})")

        if is_sat:
            max_codewords_found = current_num_codewords
            best_solution = {
                "num_codewords": current_num_codewords,
                "codewords": flecc_solver.solution,
                "variables_count": flecc_solver.variables_count,
                "clauses_count": flecc_solver.clauses_count,
                "result": result,
            }
            if validate:
                validate_codewords(
                    flecc_solver.solution,
                    distance_threshold,
                    distance_metric,
                    alphabet_size,
                )
            current_num_codewords += 1
        else:
            if status == "TIMEOUT":
                summary_status = "Timeout"
            print(f"\n{'='*70}")
            print("Search Complete!")
            print(f"Maximum codewords found: {max_codewords_found}")
            if best_solution:
                print(f"Best solution: {best_solution['codewords']}")
            print(f"Total iterations: {iteration}")
            print(f"{'='*70}\n")
            break

    if not test:
        if max_codewords_found is None and summary_status == "Optimal":
            summary_status = "UNSAT"
        summary_row = {
            "Instance": instance_name if instance_name else (
                f"FLECC_{length_of_codeword}_{distance_threshold}_"
                f"{distance_metric}_q{alphabet_size}"
            ),
            "n": length_of_codeword,
            "q": alphabet_size,
            "d": distance_threshold,
            "M_best": max_codewords_found,
            "UB": ub_max_possible_codewords,
            "Time(s)": round(timeit.default_timer() - global_start, 4),
            "Status": summary_status,
            "Vars": best_solution["variables_count"] if best_solution else None,
            "Method": "Incremental_MultiSetLex",
        }
        summary_file, sheet_name = append_summary_to_excel(
            summary_row, distance_metric
        )
        print(f"Summary saved to {summary_file} (sheet: {sheet_name})")

    return max_codewords_found, best_solution, all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FLECC Incremental SAT with MultiSet-Row + Lex-Column Symmetry Breaking"
    )
    parser.add_argument("--n", type=int, default=6, help="Codeword length")
    parser.add_argument("--d", type=int, default=3, help="Minimum distance")
    parser.add_argument("--m", type=int, default=2, help="Lower bound on # codewords")
    parser.add_argument("--q", type=int, default=2, help="Alphabet size")
    parser.add_argument("--metric", choices=["hamming", "lee"], default="hamming")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-row-sb", action="store_true", help="Disable row multiset SB")
    parser.add_argument("--no-col-sb", action="store_true", help="Disable column lex SB")
    parser.add_argument(
        "--no-fix-first-codeword",
        action="store_true",
        help="Disable fix-first-codeword=zeros for Lee metric",
    )
    parser.add_argument(
        "--max-vars", type=int, default=1_000_000,
        help="Max SAT variables (memory guard, default: 1000000)",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true", help="Don't save to Excel")
    args = parser.parse_args()

    solve_flecc_multi_sat_incremental_multisetlex(
        length_of_codeword=args.n,
        distance_threshold=args.d,
        number_of_codewords=args.m,
        distance_metric=args.metric,
        alphabet_size=args.q,
        test=args.test,
        validate=args.validate,
        timeout=args.timeout,
        row_symmetry_breaking=not args.no_row_sb,
        col_symmetry_breaking=not args.no_col_sb,
        fix_first_codeword=not args.no_fix_first_codeword,
        max_vars=args.max_vars,
    )
