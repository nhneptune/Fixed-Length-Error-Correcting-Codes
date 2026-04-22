"""
flecc_with_sat_snakelex.py – Snake-Lex Symmetry Breaking for FleccWithSatIncremental

Linearizes the codeword matrix in "snake" (boustrophedon) scan order, then
applies lex-order symmetry-breaking constraints on consecutive row pairs and
consecutive column pairs using that scan direction.

  Snake scan of an M×n matrix:
    Row 0  (even):  read left  → right   positions 0, 1, …, n-1
    Row 1  (odd):   read right → left    positions n-1, n-2, …, 0
    Row 2  (even):  read left  → right
    …

  Columns are treated symmetrically:
    Col 0  (even):  read top  → bottom  rows 0, 1, …, M-1
    Col 1  (odd):   read bottom → top   rows M-1, M-2, …, 0
    …

  ROW snake-lex:
    snake_row(i) <=_lex snake_row(i+1)
    where snake_row(i) = row[i] read in its snake direction.

  COLUMN snake-lex:
    snake_col(j) <=_lex snake_col(j+1)
    where snake_col(j) = col[j] read in its snake direction.

  FIX FIRST CODEWORD (Lee metric only):
    codeword[0] = (0, …, 0), exploiting Lee distance translation-invariance.

Snake-lex is strictly stronger than double-lex for breaking combined
row+column permutation symmetries: it eliminates more equivalent solutions
than applying separate row-lex and column-lex constraints.

Both predicates use the "equality-prefix chain" CNF encoding with
auxiliary Boolean variables (same technique as flecc_with_sat_doublelex.py).

Guarding:
  - Row SB for pair (i, i+1) is guarded by p_{i+2} so the constraint is
    vacuously satisfied when row i+1 is inactive.
  - Column SB is unguarded; inactive rows have all symbol vars = False
    (lex-minimum), so no extra guard is needed.
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
# CNF encoding helpers (shared with doublelex / sb variants)
# ---------------------------------------------------------------------------

def _encode_lex_leq(solver, alphabet, seq_a_vars, seq_b_vars, guard_var=None):
    """Add clauses enforcing seq_A <=_lex seq_B directly to *solver*.

    Parameters
    ----------
    solver : FleccWithSatSnakeLex
        Must expose allocate_variables() and _add_solver_clause().
    alphabet : tuple of str
        Ordered alphabet symbols, e.g. ('0','1','2','3').
    seq_a_vars : list[dict[str, int]]
        One-hot variable dicts for each element of sequence A.
    seq_b_vars : list[dict[str, int]]
        One-hot variable dicts for each element of sequence B.
    guard_var : int or None
        When set, every clause receives an extra literal (-guard_var), making
        the constraint active only when guard_var is True in the assignment.
    """
    assert len(seq_a_vars) == len(seq_b_vars)
    n = len(seq_a_vars)
    if n == 0:
        return

    # eq_prev encodes "A[0..j-1] == B[0..j-1]".
    # We start with a fresh variable forced to True (empty prefix is equal).
    eq_prev = solver.allocate_variables(1)
    _add_guarded(solver, [eq_prev], guard_var)

    for j in range(n):
        a_vars = seq_a_vars[j]
        b_vars = seq_b_vars[j]

        # 1. Forbid eq_prev ∧ A[j] > B[j].
        #    A[j] = sym_a > sym_b = B[j]  iff  sym_a comes after sym_b in alphabet.
        for idx_a, sym_a in enumerate(alphabet):
            for sym_b in alphabet[:idx_a]:          # sym_b < sym_a
                _add_guarded(solver, [-eq_prev, -a_vars[sym_a], -b_vars[sym_b]], guard_var)

        # 2. Update eq_next = eq_prev ∧ (A[j] == B[j]).
        #    Only needed when there is a next position.
        if j < n - 1:
            eq_next = solver.allocate_variables(1)
            _add_guarded(solver, [-eq_next, eq_prev], guard_var)   # eq_next -> eq_prev

            # eq_pos: True iff A[j] == B[j]
            # With one-hot vars: A[j]==B[j]  iff  same symbol chosen in both rows.
            # eq_pos <-> OR_s (x_A[j,s] AND x_B[j,s])
            eq_pos = solver.allocate_variables(1)
            pair_vars = []
            for sym in alphabet:
                # Forward: (x_A AND x_B) -> eq_pos
                _add_guarded(solver, [-a_vars[sym], -b_vars[sym], eq_pos], guard_var)
                # Aux pair variable for the backward direction
                pv = solver.allocate_variables(1)
                _add_guarded(solver, [-pv, a_vars[sym]], guard_var)
                _add_guarded(solver, [-pv, b_vars[sym]], guard_var)
                _add_guarded(solver, [-a_vars[sym], -b_vars[sym], pv], guard_var)
                pair_vars.append(pv)
            # Backward: eq_pos -> OR_s (x_A[j,s] AND x_B[j,s])
            _add_guarded(solver, [-eq_pos] + pair_vars, guard_var)

            # eq_next = eq_prev ∧ eq_pos
            _add_guarded(solver, [-eq_next, eq_pos], guard_var)            # eq_next -> eq_pos
            _add_guarded(solver, [-eq_prev, -eq_pos, eq_next], guard_var)  # both -> eq_next

            eq_prev = eq_next


def _add_guarded(solver, clause, guard_var):
    """Prepend -guard_var (if set) and add the clause to solver."""
    if guard_var is not None:
        solver._add_solver_clause([-guard_var] + list(clause))
    else:
        solver._add_solver_clause(list(clause))


# ---------------------------------------------------------------------------
# Subclass with snake-lex symmetry breaking
# ---------------------------------------------------------------------------

class FleccWithSatSnakeLex(FleccWithSatIncremental):
    """FleccWithSatIncremental augmented with snake-lex symmetry breaking.

    ROW snake-lex:
        snake_row(i) <=_lex snake_row(i+1)  for each consecutive row pair,
        where snake_row(i) reads row i left→right for even i, right→left for odd i.

    COLUMN snake-lex:
        snake_col(j) <=_lex snake_col(j+1)  for each consecutive column pair,
        where snake_col(j) reads column j top→bottom for even j, bottom→top for odd j.

    FIX FIRST CODEWORD (Lee only):
        codeword[0] = (0, …, 0) — always valid due to translation invariance.

    Row SB is added incrementally as new codeword slots are introduced.
    Column SB and fix-first-codeword are added once at initialization.
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

    # ------------------------------------------------------------------
    # Override: add column snake-lex (and optional Lee fix) after base init
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
        if self.col_symmetry_breaking:
            self._add_all_column_snake_lex_constraints()
        if self.fix_first_codeword and distance_metric == "lee":
            self._add_fix_first_codeword_zeros()

    # ------------------------------------------------------------------
    # Lee symmetry: fix codeword[0] = (0, …, 0)
    # ------------------------------------------------------------------

    def _add_fix_first_codeword_zeros(self):
        """Force codeword[0] = all-zeros when active (guarded by p_1).

        Valid because Lee distance is translation-invariant over Z_q^n.
        """
        p1 = self.count_activation_vars[1]
        first_sym = self.alphabet[0]  # '0'
        for j in range(self.length_of_codeword):
            var = self.codeword_vars[(0, j, first_sym)]
            self._add_solver_clause([-p1, var])

    # ------------------------------------------------------------------
    # Override: add row snake-lex when a new codeword slot is introduced
    # ------------------------------------------------------------------

    def add_distance_constraints_for_codeword(self, codeword_index):
        super().add_distance_constraints_for_codeword(codeword_index)
        if self.row_symmetry_breaking and codeword_index > 0:
            self._add_row_snake_lex_constraint(codeword_index - 1, codeword_index)

    # ------------------------------------------------------------------
    # Snake-direction helpers
    # ------------------------------------------------------------------

    def _row_seq(self, row_idx):
        """One-hot dicts for row_idx read in its snake direction.

        Even row → left to right (positions 0 … n-1).
        Odd  row → right to left (positions n-1 … 0).
        """
        n = self.length_of_codeword
        positions = range(n) if row_idx % 2 == 0 else range(n - 1, -1, -1)
        return [
            {sym: self.codeword_vars[(row_idx, j, sym)] for sym in self.alphabet}
            for j in positions
        ]

    def _col_seq(self, col_idx):
        """One-hot dicts for col_idx read in its snake direction.

        Even col → top to bottom (rows 0 … M-1).
        Odd  col → bottom to top (rows M-1 … 0).
        """
        M = self.max_possible_codewords
        rows = range(M) if col_idx % 2 == 0 else range(M - 1, -1, -1)
        return [
            {sym: self.codeword_vars[(row, col_idx, sym)] for sym in self.alphabet}
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Row snake-lex: snake_row(i) <=_lex snake_row(i+1)
    # ------------------------------------------------------------------

    def _add_row_snake_lex_constraint(self, row_a, row_b):
        """Enforce snake_row(row_a) <=_lex snake_row(row_b).

        Guarded by p_{row_b+1} so the constraint is vacuous when row_b
        is inactive.
        """
        guard = self.count_activation_vars[row_b + 1]
        _encode_lex_leq(self, self.alphabet, self._row_seq(row_a), self._row_seq(row_b), guard_var=guard)

    # ------------------------------------------------------------------
    # Column snake-lex: snake_col(j) <=_lex snake_col(j+1)
    # ------------------------------------------------------------------

    def _add_all_column_snake_lex_constraints(self):
        """Enforce snake_col(j) <=_lex snake_col(j+1) for every consecutive pair."""
        n = self.length_of_codeword
        for j in range(n - 1):
            _encode_lex_leq(
                self, self.alphabet,
                self._col_seq(j), self._col_seq(j + 1),
                guard_var=None,
            )


# ---------------------------------------------------------------------------
# Top-level driver (mirrors solve_flecc_multi_sat_incremental)
# ---------------------------------------------------------------------------

def solve_flecc_multi_sat_incremental_snakelex(
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
    """Solve FLECC with incremental SAT + snake-lex symmetry breaking.

    Extra parameters vs. solve_flecc_multi_sat_incremental:
        row_symmetry_breaking  – enforce snake_row(i) <=_lex snake_row(i+1)
        col_symmetry_breaking  – enforce snake_col(j) <=_lex snake_col(j+1)
        fix_first_codeword     – (Lee only) fix codeword[0] = (0,…,0)
        max_vars               – cap on SAT variables to prevent OOM
        instance_name          – label used in Excel output

    Returns: (max_codewords_found, best_solution, all_results)
    """
    print(f"\n{'='*70}")
    print("Starting Incremental Multi-SAT Solver with Snake-Lex Symmetry Breaking")
    print(f"{'='*70}")
    print(f"Lower bound (initial number of codewords): {number_of_codewords}")
    print(f"Codeword length: {length_of_codeword}, Distance threshold: {distance_threshold}")
    print(f"Distance metric: {distance_metric}")
    print(f"Row snake-lex: {row_symmetry_breaking}, Column snake-lex: {col_symmetry_breaking}")
    fix_fc_active = fix_first_codeword and distance_metric == "lee"
    print(f"Fix first codeword = zeros (Lee only): {fix_fc_active}")
    if timeout is not None:
        print(f"Global timeout: {timeout}s (shared across all iterations)")
    print("Learned clauses and SB constraints preserved between iterations\n")

    flecc_solver = FleccWithSatSnakeLex(
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
                print(f"Search Complete — memory limit (max_vars={max_vars}): M capped at {max_possible_codewords}.")
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
            f"{current_num_codewords}_{distance_metric}_incremental_snakelex"
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
            excel_file = "FLECC_MultiSAT_Incremental_SnakeLex.xlsx"
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
            "Method": "Incremental_SnakeLex",
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
        description="FLECC Incremental SAT with Snake-Lex Symmetry Breaking"
    )
    parser.add_argument("--n", type=int, default=6, help="Codeword length")
    parser.add_argument("--d", type=int, default=3, help="Minimum distance")
    parser.add_argument("--m", type=int, default=2, help="Lower bound on # codewords")
    parser.add_argument("--q", type=int, default=2, help="Alphabet size")
    parser.add_argument("--metric", choices=["hamming", "lee"], default="hamming")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-row-sb", action="store_true", help="Disable row snake-lex")
    parser.add_argument("--no-col-sb", action="store_true", help="Disable column snake-lex")
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

    solve_flecc_multi_sat_incremental_snakelex(
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
