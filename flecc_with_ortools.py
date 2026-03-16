"""Fixed Length Error Correcting Codes — Google OR-Tools CP-SAT Solver
======================================================================
Implements the CSPLib prob036 Essence model using OR-Tools CP-SAT:

    given Character (new type enum)
          codeWordLength : int(1..)
          numOfCodeWords : int(1..)
    letting Index be domain int(1..codeWordLength)
            String be domain function (total) Index --> Character
    given dist   : function (Character, Character) --> int(0..maxDist)
          minDist: int(0..maxDist * codeWordLength)
    find c : set (size numOfCodeWords) of String
    such that
        forAll s1, s2 in c, s1 != s2
            . (sum i : Index . dist(s1(i), s2(i))) >= minDist

Modelling approach (CP-SAT):
  • Each codeword c[i] is a list of IntVar in 0..|alphabet|-1, one per position.
  • The per-position distance dist(c[i][k], c[j][k]) is captured by an auxiliary
    IntVar d[i][j][k] whose value is enforced via AddAllowedAssignments (table
    constraint) using a pre-computed distance table.
  • The pairwise total-distance constraint becomes a simple linear sum >= minDist.
  • Lexicographic symmetry breaking is added by encoding each codeword as its
    base-|alphabet| numerical value and requiring strict ordered values across
    consecutive codewords.

Supported distance metrics:
  • hamming  — dist(a, b) = 0 if a == b else 1
  • lee      — dist(a, b) = min(|a-b|, m-|a-b|)  (alphabet viewed as a cycle)
"""

from ortools.sat.python import cp_model
import pandas as pd
import timeit
import os
import argparse


# ---------------------------------------------------------------------------
# Validation helper (shared between single-solve and multi-SAT)
# ---------------------------------------------------------------------------

def validate_codewords(codewords, distance_threshold, distance_metric, alphabet):
    """Check all pairwise distances and print a report.

    Returns True if every pair meets *distance_threshold*, False otherwise.
    """
    if not codewords:
        print("No codewords to validate")
        return False

    m = len(alphabet)

    def hamming(a, b):
        return sum(x != y for x, y in zip(a, b))

    def lee(a, b):
        total = 0
        for x, y in zip(a, b):
            ai, bi = int(x), int(y)
            diff = abs(ai - bi)
            total += min(diff, m - diff)
        return total

    dist_fn = lee if distance_metric == "lee" else hamming

    n = len(codewords)
    print(f"Validating {n} codewords with {distance_metric} distance >= {distance_threshold}")
    ok = True
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(codewords[i], codewords[j])
            print(f"  pair ({i},{j}): {codewords[i]} vs {codewords[j]} -> dist = {d}")
            if d < distance_threshold:
                print("    *** VIOLATION ***")
                ok = False
    if ok:
        print("All pairs meet the threshold.")
    else:
        print("Some pairs failed the threshold!")
    return ok


# ---------------------------------------------------------------------------
# Core solver class
# ---------------------------------------------------------------------------

class FleccWithOrTools:
    """Solve Fixed Length Error Correcting Codes with OR-Tools CP-SAT.

    Each codeword is a list of IntVar, each taking values in 0..|alphabet|-1.
    The distance function is encoded as a table constraint so it works for any
    distance metric (Hamming, Lee, or custom).

    Usage::
        solver = FleccWithOrTools()
        solver.alphabet = ['0', '1', '2', '3']   # optional: set alphabet
        solver.solve(length_of_codeword=5, distance_threshold=3,
                     number_of_codewords=4, distance_metric='lee', timeout=60)
        print(solver.solution)
    """

    def __init__(self):
        self.solution = None
        # Default binary alphabet; change before calling solve() for other alphabets
        self.alphabet = ['0', '1']
        self.length_of_codeword = None
        self.distance_threshold = None
        self.number_of_codewords = None
        self.distance_metric = "hamming"
        self.variables_count = 0
        self.timeout_occurred = False
        # Internal OR-Tools objects — reset on each solve()
        self._model = None
        self._solver_obj = None
        self._codeword_vars = None   # list[list[IntVar]], shape [n][L]

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _dist(self, a_idx, b_idx):
        """Distance between alphabet[a_idx] and alphabet[b_idx]."""
        if self.distance_metric == "hamming":
            return 0 if a_idx == b_idx else 1
        # Lee distance
        ai = int(self.alphabet[a_idx])
        bi = int(self.alphabet[b_idx])
        diff = abs(ai - bi)
        m = len(self.alphabet)
        return min(diff, m - diff)

    def _build_dist_table(self):
        """Return dist_table[a][b] = distance between alphabet[a] and alphabet[b]."""
        m = len(self.alphabet)
        return [[self._dist(a, b) for b in range(m)] for a in range(m)]

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _create_variables(self):
        m = len(self.alphabet)
        n = self.number_of_codewords
        L = self.length_of_codeword
        self._codeword_vars = [
            [self._model.NewIntVar(0, m - 1, f'c_{i}_{j}') for j in range(L)]
            for i in range(n)
        ]
        self.variables_count = n * L

    def _create_distance_constraints(self):
        """For every pair (i, j), sum_k dist(c[i][k], c[j][k]) >= distance_threshold."""
        n = self.number_of_codewords
        L = self.length_of_codeword
        m = len(self.alphabet)
        dist_table = self._build_dist_table()
        max_d = max(dist_table[a][b] for a in range(m) for b in range(m))

        # Pre-build the table of allowed (symbol_i, symbol_j, dist_value) triples once.
        # AddAllowedAssignments enforces that the assigned values form an allowed tuple.
        allowed = [(a, b, dist_table[a][b]) for a in range(m) for b in range(m)]

        for i in range(n):
            for j in range(i + 1, n):
                pos_dists = []
                for k in range(L):
                    d = self._model.NewIntVar(0, max_d, f'd_{i}_{j}_{k}')
                    # d = dist_table[c[i][k]][c[j][k]]  (table lookup via allowed tuples)
                    self._model.AddAllowedAssignments(
                        [self._codeword_vars[i][k], self._codeword_vars[j][k], d],
                        allowed,
                    )
                    pos_dists.append(d)
                # Total pairwise distance must meet the threshold
                self._model.Add(sum(pos_dists) >= self.distance_threshold)

    def _create_symmetry_breaking(self):
        """Break symmetry by enforcing strict lexicographic order on codewords.

        Each codeword is interpreted as a base-|alphabet| integer.  Adding
        num[i] < num[i+1] for consecutive codewords eliminates all permutations
        of the same codeword set from the search space.
        """
        m = len(self.alphabet)
        L = self.length_of_codeword
        n = self.number_of_codewords
        max_num = m ** L - 1

        num_vars = [self._model.NewIntVar(0, max_num, f'num_{i}') for i in range(n)]
        for i in range(n):
            # num[i] = sum_j c[i][j] * m^(L-1-j)
            coeffs = [m ** (L - 1 - j) for j in range(L)]
            self._model.Add(
                sum(coeffs[j] * self._codeword_vars[i][j] for j in range(L)) == num_vars[i]
            )

        for i in range(n - 1):
            # Strict less-than ensures all codewords are distinct AND ordered
            self._model.Add(num_vars[i] < num_vars[i + 1])

    def _add_hints(self, previous_solution):
        """Warm-start CP-SAT with a solution from a previous (smaller) solve."""
        if previous_solution is None:
            return
        for i, codeword in enumerate(previous_solution):
            if i >= self.number_of_codewords:
                break
            for j, symbol in enumerate(codeword):
                try:
                    idx = self.alphabet.index(symbol)
                    self._model.AddHint(self._codeword_vars[i][j], idx)
                except (ValueError, IndexError):
                    pass

    # ------------------------------------------------------------------
    # Public solve interface
    # ------------------------------------------------------------------

    def solve(self, length_of_codeword, distance_threshold, number_of_codewords,
              distance_metric="hamming", timeout=None, hints=None):
        """Build a fresh CP-SAT model and solve it.

        Args:
            length_of_codeword:  Length L of each codeword (codeWordLength).
            distance_threshold:  Minimum required pairwise sum-distance (minDist).
            number_of_codewords: Number of distinct codewords to find (numOfCodeWords).
            distance_metric:     'hamming' or 'lee'.
            timeout:             Optional wall-clock limit in seconds.
            hints:               Optional list of codeword strings from a previous
                                 solve to warm-start the solver.
        """
        self.length_of_codeword = length_of_codeword
        self.distance_threshold = distance_threshold
        self.number_of_codewords = number_of_codewords
        self.distance_metric = distance_metric
        self.timeout_occurred = False

        self._model = cp_model.CpModel()
        self._solver_obj = cp_model.CpSolver()

        self._create_variables()
        self._create_distance_constraints()
        self._create_symmetry_breaking()

        if hints is not None:
            self._add_hints(hints)

        if timeout is not None:
            self._solver_obj.parameters.max_time_in_seconds = float(timeout)

        status = self._solver_obj.Solve(self._model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("Solution found!")
            self.solution = [
                ''.join(
                    self.alphabet[self._solver_obj.Value(self._codeword_vars[i][j])]
                    for j in range(length_of_codeword)
                )
                for i in range(number_of_codewords)
            ]
        elif status == cp_model.UNKNOWN:
            print("Timeout occurred.")
            self.timeout_occurred = True
            self.solution = None
        else:
            print("No solution exists.")
            self.solution = None


# ---------------------------------------------------------------------------
# High-level functional interfaces
# ---------------------------------------------------------------------------

def solve_flecc(length_of_codeword, distance_threshold, number_of_codewords,
                distance_metric="hamming", alphabet=None, test=False, validate=False,
                timeout=None):
    """Solve a single FLECC instance and optionally save results to Excel.

    Args:
        length_of_codeword:  Codeword length (codeWordLength in Essence).
        distance_threshold:  Minimum pairwise distance (minDist in Essence).
        number_of_codewords: Number of codewords to find (numOfCodeWords).
        distance_metric:     'hamming' or 'lee'.
        alphabet:            List of symbol strings, e.g. ['0','1'] or ['0','1','2','3'].
                             Defaults to ['0','1'] (binary) when None.
        test:                If True, skip Excel output.
        validate:            If True, verify all pairwise distances after solving.
        timeout:             Optional time limit in seconds per solve.

    Returns:
        List of codeword strings, or None if unsatisfiable / timeout.
    """
    solver = FleccWithOrTools()
    if alphabet is not None:
        solver.alphabet = alphabet

    start = timeit.default_timer()
    solver.solve(length_of_codeword, distance_threshold, number_of_codewords,
                 distance_metric, timeout)
    runtime = timeit.default_timer() - start

    print("Codewords:", solver.solution)
    print(f"Runtime: {runtime:.2f}s")

    if validate and solver.solution:
        validate_codewords(solver.solution, distance_threshold, distance_metric, solver.alphabet)

    status = "TIMEOUT" if solver.timeout_occurred else ("SAT" if solver.solution else "UNSAT")
    instance_name = (
        f"FLECC_{length_of_codeword}_{distance_threshold}_{number_of_codewords}"
        f"_{distance_metric}_ortools"
    )
    result = {
        'Instance':  instance_name,
        'Variables': solver.variables_count,
        'Clauses':   0,
        'Runtime':   runtime,
        'Codewords': str(solver.solution) if solver.solution else "None",
        'Status':    status,
    }

    if not test:
        excel_file = 'FLECC_OrTools.xlsx'
        new_df = pd.DataFrame([result])
        if os.path.exists(excel_file):
            updated_df = pd.concat([pd.read_excel(excel_file), new_df], ignore_index=True)
        else:
            updated_df = new_df
        updated_df.to_excel(excel_file, index=False)
        print(f"Results saved to {excel_file}")

    return solver.solution


def solve_flecc_multi_sat(length_of_codeword, distance_threshold, number_of_codewords,
                          distance_metric="hamming", alphabet=None, test=False,
                          validate=False, max_iterations=100, timeout=600,
                          max_timeout_retries=1):
    """Search for the maximum number of codewords by incrementing until UNSAT.

    Starts at *number_of_codewords* (lower bound) and increases by 1 each
    iteration, reusing the previous SAT solution as a warm-start hint so
    CP-SAT can find a solution for n+1 codewords faster.

    Args:
        length_of_codeword:   Codeword length.
        distance_threshold:   Minimum pairwise distance.
        number_of_codewords:  Lower bound for the search.
        distance_metric:      'hamming' or 'lee'.
        alphabet:             Symbol list; None → binary ['0','1'].
        test:                 If True, skip Excel output.
        validate:             If True, validate the best solution found.
        max_iterations:       Hard cap on the number of iterations.
        timeout:              Per-solve time limit in seconds (default 600).
        max_timeout_retries:  How many times to retry the same problem on timeout
                              before stopping (default 1).

    Returns:
        Tuple (max_codewords, best_solution_dict, all_results)
            max_codewords      — maximum number of codewords found (or None).
            best_solution_dict — dict with keys: num_codewords, codewords,
                                 variables_count, clauses_count, result.
            all_results        — list of per-iteration result dicts.
    """
    print(f"\n{'='*70}")
    print("Starting OR-Tools Multi-SAT Solver")
    print(f"{'='*70}")
    print(f"Lower bound: {number_of_codewords}  |  Length: {length_of_codeword}  |  "
          f"Distance: {distance_threshold}  |  Metric: {distance_metric}\n")

    current_n = number_of_codewords
    max_codewords_found = None
    best_solution = None
    all_results = []
    iteration = 0
    prev_solution = None   # warm-start hint for the next iteration

    while iteration < max_iterations:
        iteration += 1
        timeout_retries = 0
        is_sat = False
        status = "UNSAT"
        runtime = 0.0
        solver = None

        while timeout_retries <= max_timeout_retries:
            print(f"[Iter {iteration}] Trying {current_n} codewords...", end=" ", flush=True)

            solver = FleccWithOrTools()
            if alphabet is not None:
                solver.alphabet = alphabet

            start = timeit.default_timer()
            solver.solve(length_of_codeword, distance_threshold, current_n,
                         distance_metric, timeout, hints=prev_solution)
            runtime = timeit.default_timer() - start

            if solver.timeout_occurred:
                timeout_retries += 1
                if timeout_retries <= max_timeout_retries:
                    print(f"TIMEOUT (retry {timeout_retries}/{max_timeout_retries})")
                else:
                    print(f"TIMEOUT (max retries exhausted)")
                    status = "TIMEOUT"
                    break
            else:
                is_sat = solver.solution is not None
                status = "SAT" if is_sat else "UNSAT"
                print(f"{status} ({runtime:.2f}s)")
                break

        instance_name = (
            f"FLECC_{length_of_codeword}_{distance_threshold}_{current_n}"
            f"_{distance_metric}_ortools"
        )
        result = {
            'Iteration':     iteration,
            'Instance':      instance_name,
            'Num_Codewords': current_n,
            'Variables':     solver.variables_count if solver else 0,
            'Clauses':       0,
            'Runtime':       runtime,
            'Codewords':     str(solver.solution) if (solver and solver.solution) else "None",
            'Status':        status,
        }
        all_results.append(result)

        if is_sat:
            max_codewords_found = current_n
            prev_solution = solver.solution
            best_solution = {
                'num_codewords':   current_n,
                'codewords':       solver.solution,
                'variables_count': solver.variables_count,
                'clauses_count':   0,
                'result':          result,
            }
            if validate:
                validate_codewords(solver.solution, distance_threshold,
                                   distance_metric, solver.alphabet)
            current_n += 1
        else:
            # UNSAT or TIMEOUT → stop search
            sep = '='*70
            if status == "UNSAT":
                print(f"\n{sep}")
                print("Search complete: UNSAT reached")
            else:
                print(f"\n{sep}")
                print("Search stopped: timeout exhausted")
            print(sep)
            print(f"Maximum codewords found: {max_codewords_found}")
            if best_solution:
                print(f"Best codewords: {best_solution['codewords']}")
            print(f"Total iterations: {iteration}")
            print(f"{sep}\n")
            break
    else:
        sep = '='*70
        print(f"\n{sep}")
        print("Search stopped: max iterations reached")
        print(f"Maximum codewords found: {max_codewords_found}")
        if best_solution:
            print(f"Best codewords: {best_solution['codewords']}")
        print(f"Total iterations: {iteration}")
        print(f"{sep}\n")

    if not test and all_results:
        excel_file = 'FLECC_MultiSAT_OrTools.xlsx'
        new_df = pd.DataFrame(all_results)
        if os.path.exists(excel_file):
            updated_df = pd.concat([pd.read_excel(excel_file), new_df], ignore_index=True)
        else:
            updated_df = new_df
        updated_df.to_excel(excel_file, index=False)
        print(f"Results saved to {excel_file}")

    return max_codewords_found, best_solution, all_results


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default configuration — mirrors flecc_with_sat.py for easy comparison
    config = {
        'length_of_codeword':  10,
        'distance_threshold':  4,
        'number_of_codewords': 40,
        'distance_metric':     'lee',
        'alphabet':            None,   # None → binary ['0','1']
        'test':                False,
        'validate':            False,
        'multi_sat':           False,
        'max_iterations':      50,
        'timeout':             600,
        'max_timeout_retries': 1,
    }

    parser = argparse.ArgumentParser(
        description="Solve Fixed Length Error Correcting Codes with OR-Tools CP-SAT"
    )
    parser.add_argument("--length",   type=int,
                        help=f"Length of codeword (default: {config['length_of_codeword']})")
    parser.add_argument("--distance", type=int,
                        help=f"Minimum distance threshold (default: {config['distance_threshold']})")
    parser.add_argument("--codewords", type=int,
                        help=f"Number of codewords (default: {config['number_of_codewords']})")
    parser.add_argument("--metric",   type=str, choices=["hamming", "lee"],
                        help=f"Distance metric (default: {config['distance_metric']})")
    parser.add_argument("--alphabet", type=str,
                        help="Comma-separated alphabet symbols, e.g. '0,1,2,3'")
    parser.add_argument("--test",     action="store_true",
                        help="Test mode: skip Excel output")
    parser.add_argument("--validate", action="store_true",
                        help="Validate pairwise distances of the solution")
    parser.add_argument("--multi-sat", action="store_true",
                        help="Search for maximum number of codewords")
    parser.add_argument("--max-iterations", type=int,
                        help=f"Max iterations for multi-SAT (default: {config['max_iterations']})")
    parser.add_argument("--timeout",  type=int,
                        help=f"Per-solve time limit in seconds (default: {config['timeout']})")
    parser.add_argument("--max-timeout-retries", type=int,
                        help=f"Max retries on timeout (default: {config['max_timeout_retries']})")

    args = parser.parse_args()

    final = config.copy()
    if args.length              is not None: final['length_of_codeword']  = args.length
    if args.distance            is not None: final['distance_threshold']  = args.distance
    if args.codewords           is not None: final['number_of_codewords'] = args.codewords
    if args.metric              is not None: final['distance_metric']     = args.metric
    if args.alphabet            is not None: final['alphabet']            = args.alphabet.split(',')
    if args.test:                            final['test']                = True
    if args.validate:                        final['validate']            = True
    if args.multi_sat:                       final['multi_sat']           = True
    if args.max_iterations      is not None: final['max_iterations']      = args.max_iterations
    if args.timeout             is not None: final['timeout']             = args.timeout
    if args.max_timeout_retries is not None: final['max_timeout_retries'] = args.max_timeout_retries

    if final['multi_sat']:
        solve_flecc_multi_sat(
            length_of_codeword  = final['length_of_codeword'],
            distance_threshold  = final['distance_threshold'],
            number_of_codewords = final['number_of_codewords'],
            distance_metric     = final['distance_metric'],
            alphabet            = final['alphabet'],
            test                = final['test'],
            validate            = final['validate'],
            max_iterations      = final['max_iterations'],
            timeout             = final['timeout'],
            max_timeout_retries = final['max_timeout_retries'],
        )
    else:
        solve_flecc(
            length_of_codeword  = final['length_of_codeword'],
            distance_threshold  = final['distance_threshold'],
            number_of_codewords = final['number_of_codewords'],
            distance_metric     = final['distance_metric'],
            alphabet            = final['alphabet'],
            test                = final['test'],
            validate            = final['validate'],
            timeout             = final['timeout'],
        )
