"""
flecc_multisat_multisetlex.py – Multi-SAT with Multiset-Row + Lex-Column Symmetry Breaking
                                 (non-incremental)

Solves the FLECC problem by iterating M = M_lb, M_lb+1, ... until UNSAT.
Each iteration builds a *fresh* SAT formula (no learned clauses preserved)
and augments it with:

  ROW multiset ordering:
    multiset(codeword[i]) >=_multiset multiset(codeword[i+1])
    for every consecutive row pair.

    Encoded by sorting each row's symbol frequencies in non-increasing
    (descending) order and comparing lexicographically:

        sorted_desc(row_i) >=_lex sorted_desc(row_{i+1})

    Per-symbol counts are encoded via a Sinz sequential counter:
        cnt_geq[s][v-1] = True iff #{j : row[j] == s} >= v

    The sorted-descending binary sequence for row i is:
        cnt_geq[q-1][0], cnt_geq[q-1][1], ..., cnt_geq[q-1][n-1],
        cnt_geq[q-2][0], ..., cnt_geq[0][n-1]

    Row SB is unguarded (no activation variables in the non-incremental model).

  COLUMN lex ordering:
    col[j] <=_lex col[j+1]  for every consecutive column pair.
    (Standard equality-prefix chain encoding.)

  FIX FIRST CODEWORD (Lee metric only):
    codeword[0] = (0, ..., 0) — valid by Lee distance translation invariance.

Unlike flecc_with_sat_multisetlex.py (incremental), this module uses the base
FleccWithSat class and rebuilds the formula from scratch for each M.
No activation variables, no assumption-based solving.

Usage
-----
    from flecc_multisat_multisetlex import solve_flecc_multi_sat_multisetlex

    max_m, best, all_results = solve_flecc_multi_sat_multisetlex(
        length_of_codeword=6,
        distance_threshold=3,
        number_of_codewords=2,
        distance_metric="hamming",
        alphabet_size=2,
        timeout=120,
    )

CLI
---
    python flecc_multisat_multisetlex.py --n 6 --d 3 --m 2 --q 2 \\
           --metric hamming --timeout 30 --test --validate
"""

from flecc_with_sat import (
    FleccWithSat,
    solve_with_timeout,
    get_cp_mip_c_multiplier_placeholder,
    compute_upper_bound_max_possible_codewords,
    append_results_to_excel_by_metric_sheet,
    append_summary_to_excel,
    validate_codewords,
)
import timeit
import pandas as pd


# ---------------------------------------------------------------------------
# CNF-level lex-order helper
# ---------------------------------------------------------------------------

def _encode_lex_leq_cnf(cnf_list, allocate_fn, alphabet, seq_a_vars, seq_b_vars):
    """Append clauses to *cnf_list* enforcing seq_A <=_lex seq_B.

    Parameters
    ----------
    cnf_list : list
        Target clause list (e.g. solver.cnf.clauses).
    allocate_fn : callable
        ``allocate_fn(n) -> int`` allocates n fresh variables, returns first.
    alphabet : tuple of str
        Ordered alphabet, e.g. ('0','1') or ('0','1','0','1') for binary.
    seq_a_vars : list[dict[str, int]]
        One-hot variable dicts for each position of sequence A.
    seq_b_vars : list[dict[str, int]]
        One-hot variable dicts for each position of sequence B.
    """
    assert len(seq_a_vars) == len(seq_b_vars), "sequences must have equal length"
    n = len(seq_a_vars)
    if n == 0:
        return

    # eq_prev: empty-prefix equality, forced True.
    eq_prev = allocate_fn(1)
    cnf_list.append([eq_prev])

    for j in range(n):
        a_vars = seq_a_vars[j]
        b_vars = seq_b_vars[j]

        # Forbid: eq_prev AND A[j] > B[j]
        for idx_a, sym_a in enumerate(alphabet):
            for sym_b in alphabet[:idx_a]:   # sym_b < sym_a
                cnf_list.append([-eq_prev, -a_vars[sym_a], -b_vars[sym_b]])

        if j < n - 1:
            eq_next = allocate_fn(1)
            eq_pos  = allocate_fn(1)

            # eq_next -> eq_prev
            cnf_list.append([-eq_next, eq_prev])

            # eq_pos <-> (A[j] == B[j])
            for sym in alphabet:
                cnf_list.append([-a_vars[sym], -b_vars[sym], eq_pos])  # backward
            pair_vars = []
            for sym in alphabet:
                pv = allocate_fn(1)
                cnf_list.append([-pv, a_vars[sym]])
                cnf_list.append([-pv, b_vars[sym]])
                cnf_list.append([-a_vars[sym], -b_vars[sym], pv])
                pair_vars.append(pv)
            cnf_list.append([-eq_pos] + pair_vars)                      # forward

            # eq_next = eq_prev AND eq_pos
            cnf_list.append([-eq_next, eq_pos])
            cnf_list.append([-eq_prev, -eq_pos, eq_next])

            eq_prev = eq_next


# ---------------------------------------------------------------------------
# Sinz sequential counter for per-row symbol frequencies
# ---------------------------------------------------------------------------

def _encode_row_count_geq_vars_cnf(cnf_list, allocate_fn, codeword_vars,
                                    row_idx, alphabet, n):
    """Build cnt_geq variables for one row via a Sinz sequential counter.

    cnt_geq[s][v-1] = True iff #{j : codeword[row_idx][j] == s} >= v,
    for v = 1..n, for each symbol s in *alphabet*.

    All clauses are appended directly to *cnf_list*.

    Returns
    -------
    cnt_geq : dict  symbol -> list[int]
        cnt_geq[s][v-1] is the SAT variable for count(s, row_idx) >= v.
    """
    cnt_geq = {}

    for s in alphabet:
        x_vars = [codeword_vars[(row_idx, j, s)] for j in range(n)]

        # Allocate r_var[(j, k)] for 1 <= k <= j <= n
        r_var = {}
        for j in range(1, n + 1):
            for k in range(1, j + 1):
                r_var[(j, k)] = allocate_fn(1)

        for j in range(1, n + 1):
            x_prev = x_vars[j - 1]   # x_{j-1} (0-indexed input)

            for k in range(1, j + 1):
                r_jk = r_var[(j, k)]

                # (1) Pass-through: r[j-1][k] -> r[j][k]
                if k < j:
                    cnf_list.append([-r_var[(j - 1, k)], r_jk])

                # (2) New count propagation
                if k == 1:
                    # r[j-1][0] = True (constant): x_{j-1} -> r[j][1]
                    cnf_list.append([-x_prev, r_jk])
                else:
                    cnf_list.append([-r_var[(j - 1, k - 1)], -x_prev, r_jk])

                # (3) Backward sourcing
                if k < j:
                    cnf_list.append([-r_jk, r_var[(j - 1, k)], x_prev])
                else:
                    # k == j: must come from r[j-1][j-1] AND x_{j-1}
                    cnf_list.append([-r_jk, x_prev])
                    if k >= 2:
                        cnf_list.append([-r_jk, r_var[(j - 1, k - 1)]])

        cnt_geq[s] = [r_var[(n, v)] for v in range(1, n + 1)]

    return cnt_geq


# ---------------------------------------------------------------------------
# Subclass: FleccWithSat + multiset-row + lex-column symmetry breaking
# ---------------------------------------------------------------------------

class FleccWithSatMultiSatMultiSetLex(FleccWithSat):
    """FleccWithSat augmented with multiset-row + lex-column SB (non-incremental).

    Builds a fresh formula for each fixed M.  No activation variables.

    ROW multiset ordering:
        sorted_desc(codeword[i]) >=_lex sorted_desc(codeword[i+1])
        encoded using per-symbol Sinz sequential counter variables.

    COLUMN lex ordering:
        col[j] <=_lex col[j+1]   (standard lex column SB).

    FIX FIRST CODEWORD (Lee only): codeword[0] = (0, ..., 0).
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
        # cnt_geq[(row_idx, s)][v-1] = SAT var for count(s, row_idx) >= v
        self._cnt_geq = {}

    def _build_all_count_geq_vars(self):
        """Build Sinz counter variables for every row."""
        n = self.length_of_codeword
        for row in range(self.number_of_codewords):
            cnt = _encode_row_count_geq_vars_cnf(
                self.cnf.clauses,
                self.allocate_variables,
                self.codeword_vars,
                row,
                self.alphabet,
                n,
            )
            for s in self.alphabet:
                self._cnt_geq[(row, s)] = cnt[s]

    def _multiset_seq(self, row_idx):
        """Binary one-hot sequence representing sorted_desc(row_idx).

        Sequence elements: cnt_geq[s, v] for s in reversed(alphabet), v = 1..n.
        Each element is a one-hot dict over binary alphabet ('0', '1'):
            '1' -> cnt_geq variable   (True iff count >= v)
            '0' -> complement variable
        """
        n = self.length_of_codeword
        seq = []
        for s in reversed(self.alphabet):       # q-1 down to 0
            cnt_vars = self._cnt_geq[(row_idx, s)]
            for v in range(n):                  # v index 0..n-1  (count >= v+1)
                bit_var = cnt_vars[v]
                not_bit_var = self.allocate_variables(1)
                # not_bit_var <-> NOT bit_var
                self.cnf.append([bit_var, not_bit_var])
                self.cnf.append([-bit_var, -not_bit_var])
                seq.append({'0': not_bit_var, '1': bit_var})
        return seq

    def _add_row_multiset_constraints(self):
        """Add sorted_desc(row_b) <=_lex sorted_desc(row_a) for all consecutive pairs."""
        for i in range(self.number_of_codewords - 1):
            seq_a = self._multiset_seq(i)
            seq_b = self._multiset_seq(i + 1)
            # sorted_desc(row_{i+1}) <=_lex sorted_desc(row_i)
            _encode_lex_leq_cnf(
                self.cnf.clauses,
                self.allocate_variables,
                ('0', '1'),
                seq_b,
                seq_a,
            )

    def _add_column_lex_constraints(self):
        """Add col[j] <=_lex col[j+1] for all consecutive column pairs."""
        M = self.number_of_codewords
        for j in range(self.length_of_codeword - 1):
            seq_a = [
                {sym: self.codeword_vars[(row, j, sym)] for sym in self.alphabet}
                for row in range(M)
            ]
            seq_b = [
                {sym: self.codeword_vars[(row, j + 1, sym)] for sym in self.alphabet}
                for row in range(M)
            ]
            _encode_lex_leq_cnf(
                self.cnf.clauses,
                self.allocate_variables,
                self.alphabet,
                seq_a,
                seq_b,
            )

    def _add_fix_first_codeword_zeros(self):
        """Force codeword[0] = (0,...,0) as a Lee-metric symmetry break."""
        first_sym = self.alphabet[0]  # '0'
        for j in range(self.length_of_codeword):
            var = self.codeword_vars[(0, j, first_sym)]
            self.cnf.append([var])

    # ------------------------------------------------------------------
    # Override solve to inject SB after constraints are built
    # ------------------------------------------------------------------

    def solve(
        self,
        length_of_codeword,
        distance_threshold,
        number_of_codewords,
        distance_metric="hamming",
        timeout=None,
        _global_start=None,
    ):
        """Build formula, add multiset-row + lex-column SB, then solve."""
        # --- setup ---
        self.set_next_aux_var(1)
        self.length_of_codeword = length_of_codeword
        self.distance_threshold = distance_threshold
        self.number_of_codewords = number_of_codewords
        self._cnt_geq = {}

        self.create_variables()
        self.create_exactly_one_symbol_per_position_constraints()

        if distance_metric == "hamming":
            self.create_hamming_distance_constraints()
        elif distance_metric == "lee":
            self.create_lee_distance_constraints()
        else:
            raise ValueError(f"unknown distance metric '{distance_metric}'")

        # --- symmetry breaking ---
        # Build count variables first (needed by row multiset SB)
        if self.row_symmetry_breaking and number_of_codewords > 1:
            self._build_all_count_geq_vars()
            self._add_row_multiset_constraints()
        if self.col_symmetry_breaking and length_of_codeword > 1:
            self._add_column_lex_constraints()
        if self.fix_first_codeword and distance_metric == "lee":
            self._add_fix_first_codeword_zeros()

        # --- record counts ---
        self.variables_count = self.next_aux_var - 1
        self.clauses_count = len(self.cnf.clauses)

        # --- load formula into solver ---
        self.append_formula()

        # Deduct formula-build time from global budget
        if timeout is not None and _global_start is not None:
            effective_timeout = max(0.0, timeout - (timeit.default_timer() - _global_start))
        else:
            effective_timeout = timeout

        if effective_timeout is not None:
            print(f"Solving with timeout: {effective_timeout:.1f}s remaining")
            sat, model = solve_with_timeout(self, effective_timeout)
            if sat == "timeout":
                print("Timeout reached")
                self.timeout_occurred = True
                sat = False
                model = None
        else:
            sat = self.solver.solve()
            model = self.solver.get_model() if sat else None

        if sat:
            print("Solution found!")
            model_set = set(model) if model else set()
            self.solution = []
            for i in range(self.number_of_codewords):
                codeword = []
                for j in range(self.length_of_codeword):
                    for symbol in self.alphabet:
                        var_id = self.codeword_vars[(i, j, symbol)]
                        if var_id in model_set:
                            codeword.append(symbol)
                            break
                self.solution.append("".join(codeword))
        else:
            print("No solution exists.")
            self.solution = None


# ---------------------------------------------------------------------------
# Top-level multi-SAT loop with multiset-row + lex-column symmetry breaking
# ---------------------------------------------------------------------------

def solve_flecc_multi_sat_multisetlex(
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
    """Solve FLECC with multi-SAT (non-incremental) + multiset-row + lex-column SB.

    Iterates M = M_lb, M_lb+1, ... building a fresh SAT formula each time,
    augmented with multiset-row and lex-column symmetry-breaking predicates.
    Stops when the solver returns UNSAT, TIMEOUT (after retries), or the
    upper bound is reached.

    Parameters
    ----------
    length_of_codeword : int
    distance_threshold : int
    number_of_codewords : int
        Lower bound; search starts from this M.
    distance_metric : 'hamming' | 'lee'
    alphabet_size : int
    test : bool
        If True, do not write Excel output.
    validate : bool
        If True, validate pairwise distances of found solutions.
    timeout : float or None
        Total wall-clock budget in seconds shared across all iterations.
    max_timeout_retries : int
        How many times to retry a timed-out iteration before stopping.
    row_symmetry_breaking : bool
        Enforce multiset(row_i) >=_m multiset(row_{i+1}).
    col_symmetry_breaking : bool
        Enforce col[j] <=_lex col[j+1].
    fix_first_codeword : bool
        (Lee only) Fix codeword[0] = (0,...,0).
    max_vars : int or None
        Cap on SAT variables; used to estimate memory safety.
    instance_name : str or None
        Custom label for Excel rows.

    Returns
    -------
    (max_codewords_found, best_solution, all_results)
        max_codewords_found : int or None
        best_solution : dict or None
        all_results : list of dict
    """
    print(f"\n{'='*70}")
    print("Starting Multi-SAT (non-incremental) Solver with MultiSet-Lex Symmetry Breaking")
    print(f"{'='*70}")
    print(f"Lower bound (initial number of codewords): {number_of_codewords}")
    print(f"Codeword length: {length_of_codeword}, Distance threshold: {distance_threshold}")
    print(f"Distance metric: {distance_metric}, Alphabet size: {alphabet_size}")
    print(f"Row multiset SB: {row_symmetry_breaking}, Column lex SB: {col_symmetry_breaking}")
    fix_fc_active = fix_first_codeword and distance_metric == "lee"
    print(f"Fix first codeword = zeros (Lee only): {fix_fc_active}")
    if timeout is not None:
        print(f"Global timeout: {timeout}s (shared across all iterations)")
    else:
        print("No timeout set")
    print("Fresh formula built for each M (non-incremental)\n")

    # Compute upper bound
    c_multiplier = get_cp_mip_c_multiplier_placeholder()
    ub_max_possible_codewords = compute_upper_bound_max_possible_codewords(
        alphabet_size=alphabet_size,
        length_of_codeword=length_of_codeword,
        distance_threshold=distance_threshold,
        requested_codewords=number_of_codewords,
        distance_metric=distance_metric,
        c_multiplier=c_multiplier,
    )
    print(f"Upper bound estimate: {ub_max_possible_codewords}")

    if number_of_codewords > ub_max_possible_codewords:
        print(
            f"M_lb={number_of_codewords} > UB={ub_max_possible_codewords}, "
            f"resetting to start from M=2"
        )
        number_of_codewords = 2

    # Memory cap
    vars_per_codeword = length_of_codeword * alphabet_size
    if max_vars is not None:
        mem_cap = max(number_of_codewords, max_vars // max(1, vars_per_codeword * 10))
    else:
        mem_cap = ub_max_possible_codewords
    effective_ub = min(ub_max_possible_codewords, mem_cap)
    memory_limited = effective_ub < ub_max_possible_codewords
    if memory_limited:
        print(
            f"Memory cap applied (max_vars={max_vars}): "
            f"max M limited to {effective_ub} (UB was {ub_max_possible_codewords})"
        )

    global_start = timeit.default_timer()
    current_num_codewords = number_of_codewords
    max_codewords_found = None
    best_solution = None
    all_results = []
    iteration = 0
    summary_status = "Optimal"

    while current_num_codewords <= effective_ub:
        iteration += 1
        current_timeout_retries = 0
        timeout_occurred = False
        status = "UNSAT"
        is_sat = False
        runtime = 0.0
        flecc_solver = None

        while current_timeout_retries <= max_timeout_retries:
            print(
                f"[Iteration {iteration}] Trying {current_num_codewords} codewords...",
                end=" ",
                flush=True,
            )

            # Build a fresh solver each retry (formula includes SB)
            flecc_solver = FleccWithSatMultiSatMultiSetLex(
                alphabet_size=alphabet_size,
                row_symmetry_breaking=row_symmetry_breaking,
                col_symmetry_breaking=col_symmetry_breaking,
                fix_first_codeword=fix_first_codeword,
            )

            iter_start = timeit.default_timer()
            flecc_solver.solve(
                length_of_codeword=length_of_codeword,
                distance_threshold=distance_threshold,
                number_of_codewords=current_num_codewords,
                distance_metric=distance_metric,
                timeout=timeout,
                _global_start=global_start,
            )
            runtime = timeit.default_timer() - iter_start

            is_sat = flecc_solver.solution is not None
            status = "SAT" if is_sat else "UNSAT"

            if flecc_solver.timeout_occurred:
                timeout_occurred = True
                current_timeout_retries += 1
                if current_timeout_retries <= max_timeout_retries:
                    print(
                        f"TIMEOUT ({runtime:.2f}s) — retrying "
                        f"({current_timeout_retries}/{max_timeout_retries})"
                    )
                    continue
                else:
                    print(f"TIMEOUT ({runtime:.2f}s) — max retries reached.")
                    status = "TIMEOUT"
                    is_sat = False
            else:
                print(f"{status} ({runtime:.2f}s)")
                break

        _instance_label = instance_name if instance_name else (
            f"FLECC_{length_of_codeword}_{distance_threshold}_"
            f"{current_num_codewords}_{distance_metric}_multisat_multisetlex"
        )
        codewords_str = str(flecc_solver.solution) if (flecc_solver and flecc_solver.solution) else "None"

        result = {
            "Iteration": iteration,
            "Instance": _instance_label,
            "Num_Codewords": current_num_codewords,
            "Variables": flecc_solver.variables_count if flecc_solver else None,
            "Clauses": flecc_solver.clauses_count if flecc_solver else None,
            "Runtime": round(timeit.default_timer() - global_start, 4),
            "Codewords": codewords_str,
            "Status": status,
        }
        all_results.append(result)

        if not test:
            excel_file = "FLECC_MultiSAT_MultiSetLex.xlsx"
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
    else:
        # Exited loop because current_num_codewords > effective_ub
        if memory_limited:
            summary_status = "MemoryLimit"
        print(f"\n{'='*70}")
        if memory_limited:
            print(f"Search Complete — reached memory limit (max_vars={max_vars}): M capped at {effective_ub}.")
        else:
            print("Search Complete — reached upper bound estimate.")
        print(f"Maximum codewords found: {max_codewords_found}")
        if best_solution:
            print(f"Best solution: {best_solution['codewords']}")
        print(f"Total iterations: {iteration}")
        print(f"{'='*70}\n")

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
            "Method": "MultiSAT_MultiSetLex",
        }
        summary_file, sheet_name = append_summary_to_excel(summary_row, distance_metric)
        print(f"Summary saved to {summary_file} (sheet: {sheet_name})")

    return max_codewords_found, best_solution, all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FLECC Multi-SAT (non-incremental) with MultiSet-Row + Lex-Column Symmetry Breaking"
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
        "--max-vars",
        type=int,
        default=1_000_000,
        help="Max SAT variables for memory protection (default: 1000000)",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true", help="Don't save to Excel")
    args = parser.parse_args()

    solve_flecc_multi_sat_multisetlex(
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
