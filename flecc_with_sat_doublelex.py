"""
flecc_with_sat_doublelex.py – Double Lex Symmetry Breaking for FleccWithSatIncremental

Adds three families of symmetry-breaking predicates to the incremental
SAT solver:

  ROW symmetry breaking (double-lex):
    codeword[i] <=_lex codeword[i+1]   for every consecutive row pair.

  COLUMN symmetry breaking (double-lex):
    col[j] <=_lex col[j+1]             for every consecutive column pair,
    where col[j] is the vector (codeword[0][j], codeword[1][j], ...).

  FIX FIRST CODEWORD (Lee metric only):
    codeword[0] = (0, ..., 0), exploiting Lee distance translation-invariance.

Both lex predicates are encoded with the standard "chain of equality prefix"
technique using auxiliary Boolean variables.

  For a pair of sequences A = (a_0, ..., a_{k-1}) and
                           B = (b_0, ..., b_{k-1})
  we introduce eq_j  (True iff A[0..j-1] == B[0..j-1]) and encode:

      eq_0  = True  (empty prefix is always equal)
      eq_j  = eq_{j-1} AND (A[j-1] == B[j-1])     for j >= 1

  The constraint A <=_lex B is then:

      for every position j:
          eq_j AND (A[j] > B[j])  must be UNSAT
      i.e.   NOT (eq_j AND A[j] > B[j])
      i.e.   eq_j -> (A[j] <= B[j])

  Comparing one-hot encoded symbols (x[row, pos, symbol]):
      A[j] == B[j]  iff  for each symbol s: x_A[j,s] == x_B[j,s]
      A[j] >  B[j]  iff  there exists s such that x_A[j,s]=1 and
                          x_B[j,s']=1 with s' < s (i.e. s' is strictly
                          smaller symbol).

Because symmetry-breaking constraints interact with the activation
variables p_M (codewords that are inactive have all symbol vars forced
to False), care must be taken:
  - Row SB for the pair (i, i+1) is guarded by p_{i+1}: if slot i+1 is
    not active the constraint is trivially satisfied (inactive slot has
    all-symbol vars = 0, which is lex-smallest).
  - Column SB for the pair (j, j+1) is unguarded (positions are always
    present when any codeword exists).

Usage
-----
Instantiate FleccWithSatDoubleLex and call initialize_base_constraints
and solve_incremental exactly like FleccWithSatIncremental.  The SB
predicates can be toggled separately:

    solver = FleccWithSatDoubleLex(
        alphabet_size=4,
        row_symmetry_breaking=True,
        col_symmetry_breaking=True,
        fix_first_codeword=True,
    )

The top-level function solve_flecc_multi_sat_incremental_doublelex mirrors
solve_flecc_multi_sat_incremental from flecc_with_sat.py.
"""

from flecc_with_sat import (
    FleccWithSatIncremental,
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
# Helper: encode lex <=  for two one-hot encoded sequences
# ---------------------------------------------------------------------------

def _encode_lex_leq(solver, alphabet, seq_a_vars, seq_b_vars, guard_var=None):
    """Add clauses to enforce seq_A <=_lex seq_B to *solver* directly.

    Parameters
    ----------
    solver : FleccWithSatIncrementalSB
        The solver whose _add_solver_clause / allocate_variables are used.
    alphabet : tuple of str
        The ordered alphabet, e.g. ('0','1','2','3').
    seq_a_vars : list of dict {symbol -> var_id}
        One-hot variable dicts for each position of sequence A.
    seq_b_vars : list of dict {symbol -> var_id}
        One-hot variable dicts for each position of sequence B.
    guard_var : int or None
        If provided, every clause is guarded by this literal (the clause
        is only active when guard_var is True in the model).
    """
    assert len(seq_a_vars) == len(seq_b_vars)
    n = len(seq_a_vars)
    if n == 0:
        return

    # eq[j] means A[0..j-1] == B[0..j-1].
    # eq[0] is always True; we represent it as a constant.
    # For j >= 1 we allocate an auxiliary variable.

    # eq_prev starts as a "unit" variable forced to True.
    eq_true_var = solver.allocate_variables(1)
    # Force eq_true_var = True (unit clause).
    _add_guarded(solver, [eq_true_var], guard_var)

    eq_prev = eq_true_var

    for j in range(n):
        a_vars = seq_a_vars[j]   # {symbol -> var_id}
        b_vars = seq_b_vars[j]

        # ------------------------------------------------------------------
        # 1. Forbid: eq_prev AND A[j] > B[j]
        #    A[j] > B[j] happens when A[j] = s_a and B[j] = s_b with s_a > s_b.
        # ------------------------------------------------------------------
        for idx_a, sym_a in enumerate(alphabet):
            for sym_b in alphabet[:idx_a]:   # sym_b < sym_a means A > B
                # Clause: NOT(eq_prev) OR NOT(x_A[j,sym_a]) OR NOT(x_B[j,sym_b])
                clause = [-eq_prev, -a_vars[sym_a], -b_vars[sym_b]]
                _add_guarded(solver, clause, guard_var)

        # ------------------------------------------------------------------
        # 2. Update eq_next = eq_prev AND (A[j] == B[j])
        #    A[j] == B[j]  iff  for all s: x_A[j,s] = x_B[j,s]
        #
        #    eq_next -> eq_prev  (easy: add [-eq_next, eq_prev])
        #    eq_next -> A[j]==B[j]:
        #        for each s: eq_next -> (x_A[j,s] <-> x_B[j,s])
        #
        #    eq_prev AND A[j]==B[j] -> eq_next:
        #        We need: eq_prev AND (for all s, x_A==x_B) -> eq_next
        #        Equivalently: NOT eq_next -> (NOT eq_prev OR A[j]!=B[j])
        #
        #    The cleanest CNF uses an equality literal per position first.
        #
        #    eq_pos_j:  True iff A[j] == B[j]
        # ------------------------------------------------------------------
        if j < n - 1:
            # We only need eq_next for the next iteration.
            eq_next = solver.allocate_variables(1)

            # eq_next -> eq_prev
            _add_guarded(solver, [-eq_next, eq_prev], guard_var)

            # Encode eq_pos_j: True iff A[j] == B[j]
            # eq_pos_j <-> AND_{s} (x_A[j,s] <-> x_B[j,s])
            # With one-hot variables, A[j]==B[j] iff the same symbol is chosen
            # in both A and B.  So eq_pos_j <-> OR_{s}(x_A[j,s] AND x_B[j,s]).
            eq_pos = solver.allocate_variables(1)

            # eq_pos -> OR_{s}(x_A and x_B) is captured by the backward direction
            # eq_pos -> exists s: (x_A[j,s]=1 AND x_B[j,s]=1)
            # Backward: for each s, (x_A[j,s] AND x_B[j,s]) -> eq_pos
            for sym in alphabet:
                _add_guarded(solver, [-a_vars[sym], -b_vars[sym], eq_pos], guard_var)
            # Forward: eq_pos -> (exists s with x_A[j,s] AND x_B[j,s])
            # This is: eq_pos -> big_OR of pair_vars.  We use auxiliary pair vars.
            pair_vars = []
            for sym in alphabet:
                pv = solver.allocate_variables(1)
                _add_guarded(solver, [-pv, a_vars[sym]], guard_var)
                _add_guarded(solver, [-pv, b_vars[sym]], guard_var)
                _add_guarded(solver, [-a_vars[sym], -b_vars[sym], pv], guard_var)
                pair_vars.append(pv)
            _add_guarded(solver, [-eq_pos] + pair_vars, guard_var)

            # eq_next = eq_prev AND eq_pos
            # eq_next -> eq_prev  (already added above)
            # eq_next -> eq_pos
            _add_guarded(solver, [-eq_next, eq_pos], guard_var)
            # eq_prev AND eq_pos -> eq_next
            _add_guarded(solver, [-eq_prev, -eq_pos, eq_next], guard_var)

            eq_prev = eq_next


def _add_guarded(solver, clause, guard_var):
    """Add *clause* to the solver; if guard_var is set, negate it as a
    guard so the clause only activates when guard_var=True."""
    if guard_var is not None:
        full_clause = [-guard_var] + list(clause)
    else:
        full_clause = list(clause)
    solver._add_solver_clause(full_clause)


# ---------------------------------------------------------------------------
# Subclass with double-lex symmetry breaking
# ---------------------------------------------------------------------------

class FleccWithSatDoubleLex(FleccWithSatIncremental):
    """FleccWithSatIncremental augmented with double-lex symmetry breaking.

    Row SB:   codeword[i] <=_lex codeword[i+1]  for each consecutive pair.
    Column SB: col[j] <=_lex col[j+1]            for each consecutive column pair,
               where col[j] = (codeword[0][j], ..., codeword[M-1][j]).
    Fix first codeword (Lee only): codeword[0] = (0, ..., 0).

    Both lex predicates are encoded incrementally: row SB for a new codeword
    index k is added when that row is first introduced (inside
    add_distance_constraints_for_codeword).  Column SB and fix-first-codeword
    are added once during initialize_base_constraints.
    """

    def __init__(self, alphabet_size=2, row_symmetry_breaking=True, col_symmetry_breaking=True, fix_first_codeword=True):
        super().__init__(alphabet_size=alphabet_size)
        self.row_symmetry_breaking = row_symmetry_breaking
        self.col_symmetry_breaking = col_symmetry_breaking
        self.fix_first_codeword = fix_first_codeword

    # ------------------------------------------------------------------
    # Override initialize_base_constraints to add column SB
    # ------------------------------------------------------------------

    def initialize_base_constraints(self, length_of_codeword, distance_threshold, distance_metric, max_possible_codewords=None):
        """Same as parent, then appends column lex-order SB."""
        super().initialize_base_constraints(
            length_of_codeword,
            distance_threshold,
            distance_metric,
            max_possible_codewords=max_possible_codewords,
        )
        if self.col_symmetry_breaking:
            self._add_all_column_lex_constraints()
        if self.fix_first_codeword and distance_metric == "lee":
            self._add_fix_first_codeword_zeros()

    # ------------------------------------------------------------------
    # Lee symmetry breaking: fix codeword[0] = (0, ..., 0)
    # ------------------------------------------------------------------

    def _add_fix_first_codeword_zeros(self):
        """Fix codeword[0] = (0, ..., 0) as a Lee-metric symmetry break.

        Lee distance is translation-invariant over Z_q:
            d_L(a + c, b + c) = d_L(a, b)  for any c in Z_q^n.
        Therefore any Lee code can be translated so that one codeword is
        the all-zeros vector, reducing the search space by a factor of q^n.

        Each clause is guarded by p_1 (count_activation_vars[1]) so the
        constraint is vacuously satisfied when row 0 is inactive.
        """
        p1 = self.count_activation_vars[1]
        first_sym = self.alphabet[0]   # symbol '0'
        for j in range(self.length_of_codeword):
            var = self.codeword_vars[(0, j, first_sym)]
            # p_1 -> x[0, j, '0']  (if row 0 is active, its j-th symbol is '0')
            self._add_solver_clause([-p1, var])

    # ------------------------------------------------------------------
    # Override add_distance_constraints_for_codeword to also add row SB
    # ------------------------------------------------------------------

    def add_distance_constraints_for_codeword(self, codeword_index):
        """Adds distance constraints (parent) and row lex SB for the new row."""
        super().add_distance_constraints_for_codeword(codeword_index)
        if self.row_symmetry_breaking and codeword_index > 0:
            self._add_row_lex_constraint(codeword_index - 1, codeword_index)

    # ------------------------------------------------------------------
    # Row lex: codeword[i] <=_lex codeword[i+1]
    # ------------------------------------------------------------------

    def _add_row_lex_constraint(self, row_a, row_b):
        """Enforce codeword[row_a] <=_lex codeword[row_b].

        The constraint is guarded by p_{row_b+1} (activation of row_b) so
        that it is vacuously satisfied when row_b is inactive.
        """
        guard = self.count_activation_vars[row_b + 1]

        seq_a = [
            {sym: self.codeword_vars[(row_a, j, sym)] for sym in self.alphabet}
            for j in range(self.length_of_codeword)
        ]
        seq_b = [
            {sym: self.codeword_vars[(row_b, j, sym)] for sym in self.alphabet}
            for j in range(self.length_of_codeword)
        ]

        _encode_lex_leq(self, self.alphabet, seq_a, seq_b, guard_var=guard)

    # ------------------------------------------------------------------
    # Column lex: col[j] <=_lex col[j+1]
    # ------------------------------------------------------------------

    def _add_all_column_lex_constraints(self):
        """Enforce col[j] <=_lex col[j+1] for each consecutive column pair.

        col[j] is the vertical vector (codeword[0][j], ..., codeword[M-1][j])
        ordered by row index.  The constraint is unguarded at the column level;
        individual entries for inactive rows have all symbol vars = False (lex
        minimum), so the constraint is naturally satisfied for inactive rows.

        Note: we compare col vectors only using the active rows implied by the
        base activation encoding.  Because inactive rows have all symbol vars
        forced to False (representing the zero/smallest symbol effectively), this
        gives a valid lex ordering over the realised codeword matrix.
        """
        M = self.max_possible_codewords
        n = self.length_of_codeword

        for j in range(n - 1):
            # seq_a = column j, seq_b = column j+1, each of length M
            seq_a = [
                {sym: self.codeword_vars[(row, j, sym)] for sym in self.alphabet}
                for row in range(M)
            ]
            seq_b = [
                {sym: self.codeword_vars[(row, j + 1, sym)] for sym in self.alphabet}
                for row in range(M)
            ]
            # No guard; column ordering is a global property.
            _encode_lex_leq(self, self.alphabet, seq_a, seq_b, guard_var=None)


# ---------------------------------------------------------------------------
# Top-level solver function (mirrors solve_flecc_multi_sat_incremental)
# ---------------------------------------------------------------------------

def solve_flecc_multi_sat_incremental_doublelex(
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
    """Solve FLECC with incremental SAT + double-lex symmetry breaking.

    Parameters are identical to solve_flecc_multi_sat_incremental except for:
        row_symmetry_breaking: enforce codeword[i] <=_lex codeword[i+1]
        col_symmetry_breaking: enforce col[j] <=_lex col[j+1]
        fix_first_codeword: (Lee only) fix codeword[0] = (0,...,0)

    Returns:
        (max_codewords_found, best_solution, all_results)
    """
    print(f"\n{'='*70}")
    print("Starting Incremental Multi-SAT Solver with Double-Lex Symmetry Breaking")
    print(f"{'='*70}")
    print(f"Lower bound (initial number of codewords): {number_of_codewords}")
    print(f"Codeword length: {length_of_codeword}, Distance threshold: {distance_threshold}")
    print(f"Distance metric: {distance_metric}")
    print(f"Row SB: {row_symmetry_breaking}, Column SB: {col_symmetry_breaking}")
    fix_fc_active = fix_first_codeword and distance_metric == "lee"
    print(f"Fix first codeword = zeros (Lee only): {fix_fc_active}")
    if timeout is not None:
        print(f"Global timeout: {timeout}s (shared across all iterations)")
    print("Learned clauses and SB constraints preserved between iterations\n")

    flecc_solver = FleccWithSatDoubleLex(
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
    mem_cap = max(number_of_codewords, max_vars // max(1, vars_per_codeword * 10)) if max_vars is not None else ub_max_possible_codewords
    max_possible_codewords = max(number_of_codewords, min(ub_max_possible_codewords, trajectory_cap, mem_cap))
    memory_limited = max_possible_codewords < ub_max_possible_codewords
    print(
        f"Upper bound estimate: {ub_max_possible_codewords}, "
        f"trajectory cap: {trajectory_cap}, using: {max_possible_codewords}"
    )
    if memory_limited:
        print(f"Memory cap applied (max_vars={max_vars}): max M limited to {max_possible_codewords} (UB was {ub_max_possible_codewords})")

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
                print(f"Search Complete — reached memory limit (max_vars={max_vars}): M capped at {max_possible_codewords}.")
            else:
                print("Search Complete — reached upper bound estimate.")
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
            sat, _ = flecc_solver.solve_incremental(
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

        if timeout_occurred and current_timeout_retries > max_timeout_retries:
            status = "TIMEOUT"

        _instance_label = instance_name if instance_name else (
            f"FLECC_{length_of_codeword}_{distance_threshold}_"
            f"{current_num_codewords}_{distance_metric}_incremental_doublelex"
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
            excel_file = "FLECC_MultiSAT_Incremental_DoubleLex.xlsx"
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
            "Method": "Incremental_DoubleLex",
        }
        summary_file, sheet_name = append_summary_to_excel(
            summary_row, distance_metric
        )
        print(f"Summary saved to {summary_file} (sheet: {sheet_name})")

    return max_codewords_found, best_solution, all_results


# ---------------------------------------------------------------------------
# Quick smoke-test / CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLECC Incremental SAT with Double-Lex Symmetry Breaking")
    parser.add_argument("--n", type=int, default=6, help="Codeword length")
    parser.add_argument("--d", type=int, default=3, help="Minimum distance")
    parser.add_argument("--m", type=int, default=2, help="Lower bound on # codewords")
    parser.add_argument("--q", type=int, default=2, help="Alphabet size")
    parser.add_argument("--metric", choices=["hamming", "lee"], default="hamming")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-row-sb", action="store_true", help="Disable row symmetry breaking")
    parser.add_argument("--no-col-sb", action="store_true", help="Disable column symmetry breaking")
    parser.add_argument("--no-fix-first-codeword", action="store_true", help="Disable fix-first-codeword=zeros for Lee metric")
    parser.add_argument("--max-vars", type=int, default=1_000_000, help="Max SAT variables for memory protection (default: 1000000)")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true", help="Don't save to Excel")
    args = parser.parse_args()

    solve_flecc_multi_sat_incremental_doublelex(
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
