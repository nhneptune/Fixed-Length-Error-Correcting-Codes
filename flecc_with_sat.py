from pycryptosat import Solver as CryptoMiniSat5
from pysat.formula import CNF
from pypblib import pblib
from pypblib.pblib import PBConfig, AuxVarManager, VectorClauseDatabase, WeightedLit, PBConstraint, Pb2cnf
import pandas as pd
import timeit
import os
import argparse
import signal
import threading
import math

def solve_with_timeout(solver, timeout_seconds, assumptions=None):
    """
    Solve SAT problem with timeout using threading.
    
    Args:
        solver: FleccWithSat solver instance
        timeout_seconds: Timeout in seconds
        
    Returns:
        Tuple: (sat_result, assignment) or ("timeout", None) if timeout
    """
    result = ["timeout", None]  # Default to timeout
    
    def solve_thread():
        try:
            if assumptions is None:
                sat, assignment = solver.solver.solve()
            else:
                try:
                    sat, assignment = solver.solver.solve(assumptions=assumptions)
                except TypeError:
                    sat, assignment = solver.solver.solve(assumptions)
            result[0] = sat
            result[1] = assignment
        except Exception as e:
            result[0] = False
            result[1] = None
    
    thread = threading.Thread(target=solve_thread)
    thread.daemon = True  # Make thread daemon so it doesn't prevent program exit
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Timeout occurred - thread is still running but marked as daemon
        # Return timeout indicator
        return "timeout", None
    
    return result[0], result[1]


def get_cp_mip_c_multiplier_placeholder():
    """Return placeholder value for c in UB = min(q^n, q^(n-d+1), c*M).

    TODO: replace this with a CP/MIP-based estimator once those models are ready.
    """
    return None


def _compute_lee_weight_multiplicities(alphabet_size):
    """Count how many q-ary symbols have each possible Lee weight."""
    q = alphabet_size
    max_weight = q // 2
    multiplicities = {0: 1}

    for weight in range(1, max_weight + 1):
        if q % 2 == 0 and weight == q // 2:
            multiplicities[weight] = 1
        else:
            multiplicities[weight] = 2

    return multiplicities


def compute_lee_sphere_volume(alphabet_size, length_of_codeword, radius):
    """Compute V(n, t): number of q-ary vectors with Lee weight <= t.

    Lee weight of a coordinate is min(|x|, q - |x|), so per-coordinate
    weights range from 0 to floor(q/2). The sphere volume is counted with
    dynamic programming over the total Lee weight.
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be > 0")
    if length_of_codeword < 0:
        raise ValueError("length_of_codeword must be >= 0")
    if radius < 0:
        return 0

    multiplicities = _compute_lee_weight_multiplicities(alphabet_size)

    dp = [0] * (radius + 1)
    dp[0] = 1

    for _ in range(length_of_codeword):
        next_dp = [0] * (radius + 1)
        for current_weight in range(radius + 1):
            count = dp[current_weight]
            if count == 0:
                continue

            for symbol_weight, multiplicity in multiplicities.items():
                new_weight = current_weight + symbol_weight
                if new_weight <= radius:
                    next_dp[new_weight] += count * multiplicity

        dp = next_dp

    return sum(dp)


def compute_lee_sphere_packing_bound(alphabet_size, length_of_codeword, minimum_distance):
    """Compute the Lee sphere packing bound floor(q^n / V(n, t))."""
    q = alphabet_size
    n = length_of_codeword
    d = minimum_distance
    packing_radius = max(0, (d - 1) // 2)
    sphere_volume = compute_lee_sphere_volume(q, n, packing_radius)

    if sphere_volume <= 0:
        return q ** n

    return (q ** n) // sphere_volume


def _compute_hamming_sphere_packing_bound(alphabet_size, length_of_codeword, minimum_distance):
    """Compute the Hamming sphere packing bound.

    Returns floor(q^n / V_q(n, t)) where t = floor((d-1)/2).
    """
    q = alphabet_size
    n = length_of_codeword
    d = minimum_distance
    packing_radius = max(0, (d - 1) // 2)

    ball_volume = 0
    for i in range(packing_radius + 1):
        ball_volume += math.comb(n, i) * ((q - 1) ** i)

    if ball_volume <= 0:
        return q ** n

    return (q ** n) // ball_volume


def compute_upper_bound_max_possible_codewords(
    alphabet_size,
    length_of_codeword,
    distance_threshold,
    requested_codewords,
    distance_metric="hamming",
    c_multiplier=None,
):
    """Compute an upper bound for max_possible_codewords by distance metric.

    Args:
        alphabet_size: q
        length_of_codeword: n
        distance_threshold: d
        requested_codewords: M
        distance_metric: 'hamming' or 'lee'
        c_multiplier: c from CP/MIP upper-bound model (placeholder for now)
    """
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be > 0")
    if length_of_codeword < 0:
        raise ValueError("length_of_codeword must be >= 0")
    if requested_codewords < 0:
        raise ValueError("requested_codewords must be >= 0")

    q = alphabet_size
    n = length_of_codeword
    d = distance_threshold
    m = requested_codewords
    metric = str(distance_metric).strip().lower()

    if metric == "hamming":
        term_q_pow_n = q ** n

        exponent = n - d + 1
        # If exponent < 0 then q^(n-d+1) is fractional (<1); use 1 as integer cap floor.
        term_q_pow_n_minus_d_plus_1 = q ** exponent if exponent >= 0 else 1

        if c_multiplier is None:
            term_c_times_m = float("inf")
        else:
            term_c_times_m = max(0, math.ceil(c_multiplier * m))

        return int(min(term_q_pow_n, term_q_pow_n_minus_d_plus_1, term_c_times_m))

    if metric == "lee":
        half_alphabet = q // 2
        if half_alphabet == 0:
            return int(q ** n)

        translated_hamming_distance = math.ceil(d / half_alphabet)

        singleton_exponent = n - translated_hamming_distance + 1
        heuristic_singleton_1 = q ** singleton_exponent if singleton_exponent >= 0 else 1

        heuristic_singleton_2 = _compute_hamming_sphere_packing_bound(
            alphabet_size=q,
            length_of_codeword=n,
            minimum_distance=translated_hamming_distance,
        )

        lee_sphere_packing_bound = compute_lee_sphere_packing_bound(
            alphabet_size=q,
            length_of_codeword=n,
            minimum_distance=d,
        )

        return int(min(heuristic_singleton_1, heuristic_singleton_2, lee_sphere_packing_bound))

    raise ValueError(f"unknown distance metric '{distance_metric}'")


def _sheet_name_from_distance_metric(distance_metric):
    metric = str(distance_metric).strip().lower()
    if metric in {"hamming", "lee"}:
        return metric
    return "other"


def append_results_to_excel_by_metric_sheet(excel_file, results_df, distance_metric):
    """Append results to a sheet chosen by distance metric.

    - hamming -> sheet "hamming"
    - lee -> sheet "lee"
    """
    if results_df is None or results_df.empty:
        return _sheet_name_from_distance_metric(distance_metric)

    sheet_name = _sheet_name_from_distance_metric(distance_metric)
    existing_df = pd.DataFrame()

    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file, sheet_name=sheet_name)
        except ValueError:
            # Sheet does not exist yet.
            existing_df = pd.DataFrame()

    if existing_df.empty:
        updated_df = results_df.copy()
    else:
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)

    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(excel_file, mode="w", engine="openpyxl") as writer:
            updated_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return sheet_name

class FleccWithSat:
    """A class to solve the Fixed Length Error Correcting Codes problem using SAT solvers."""
    def __init__(self, alphabet_size=2):
        self.solver = CryptoMiniSat5()
        self.cnf = CNF()
        self.solution = None
        self.next_aux_var = 1
        self.alphabet_size = None
        self.alphabet = tuple()
        self.set_alphabet_size(alphabet_size)
        self.length_of_codeword = None
        self.distance_threshold = None
        self.number_of_codewords = None
        self.codeword_vars = None
        self.codewords = None
        self.variables_count = 0
        self.clauses_count = 0
        self.timeout_occurred = False

    def set_alphabet_size(self, alphabet_size):
        """Set alphabet to symbols '0'..str(q-1)."""
        if alphabet_size is None or alphabet_size < 2:
            raise ValueError("alphabet_size (q) must be >= 2")
        self.alphabet_size = int(alphabet_size)
        self.alphabet = tuple(str(i) for i in range(self.alphabet_size))
    
    def set_next_aux_var(self, next_var):
        """Set the next auxiliary variable ID."""
        self.next_aux_var = next_var

    def allocate_variables(self, count):
        start = self.next_aux_var
        self.next_aux_var += count
        return start
    
    def append_formula(self):
        for clause in self.cnf:
            self.solver.add_clause(clause)

    def add_xor(self, lits, rhs=False):
        """Add an XOR constraint (native to CryptoMiniSat).

        lits: list of integer literals (positive for var, negative for negation)
        rhs: boolean, parity (False for even / 0, True for odd / 1)
        """
        self.solver.add_xor_clause(lits, rhs)
    
    def create_variables(self):
        """Create a variable for each alphabet symbol in each position of each codeword."""
        self.codeword_vars = {}
        for i in range(self.number_of_codewords):
            for j in range(self.length_of_codeword):
                for symbol in self.alphabet:
                    var_id = self.allocate_variables(1)
                    self.codeword_vars[(i, j, symbol)] = var_id

    def create_exactly_one_symbol_per_position_constraints(self):
        """Create constraints to ensure that exactly one symbol is chosen for each position in each codeword."""
        for i in range(self.number_of_codewords):
            for j in range(self.length_of_codeword):
                # At least one symbol must be chosen
                at_least_one_clause = [self.codeword_vars[(i, j, symbol)] for symbol in self.alphabet]
                self.cnf.append(at_least_one_clause)
                
                # At most one symbol must be chosen
                for symbol1 in self.alphabet:
                    for symbol2 in self.alphabet:
                        if symbol1 < symbol2:  # Avoid duplicates
                            at_most_one_clause = [-self.codeword_vars[(i, j, symbol1)], -self.codeword_vars[(i, j, symbol2)]]
                            self.cnf.append(at_most_one_clause)

    def create_hamming_distance_constraints(self):
        """Create constraints to ensure that the Hamming distance between any two codewords is at least the distance threshold."""
        for i in range(self.number_of_codewords):
            for j in range(i + 1, self.number_of_codewords):
                # For each position create a position-difference var which is the OR
                # of per-symbol XORs between the two codewords at that position.
                diff_literals = []
                for k in range(self.length_of_codeword):
                    xor_literals = []
                    for symbol in self.alphabet:
                        var_i = self.codeword_vars[(i, k, symbol)]
                        var_j = self.codeword_vars[(j, k, symbol)]
                        xor_lit = self.allocate_variables(1)
                        # Use native XOR constraint: xor_lit == var_i XOR var_j
                        self.add_xor([var_i, var_j, xor_lit], False)
                        xor_literals.append(xor_lit)

                    # pos_diff is true iff any xor_lit is true (i.e. symbols differ at this position)
                    pos_diff = self.allocate_variables(1)
                    for xl in xor_literals:
                        # if a symbol XOR is true => position differs
                        self.cnf.append([-xl, pos_diff])
                    # if position differs => at least one symbol XOR is true
                    clause = [-pos_diff] + xor_literals
                    self.cnf.append(clause)
                    diff_literals.append((WeightedLit(pos_diff, 1)))

                # At least distance_threshold of the position-difference literals must be true
                config = PBConfig()
                aux_var_manager = AuxVarManager(self.next_aux_var)
                clause_database = VectorClauseDatabase(config)
                constraint = PBConstraint(diff_literals, pblib.GEQ , self.distance_threshold)
                
                # Encode the pseudo-Boolean constraint to CNF
                pb2cnf = Pb2cnf(config)
                pb2cnf.encode(constraint, clause_database, aux_var_manager)
                
                # Add the generated clauses to the CNF formula
                for clause in clause_database.get_clauses():
                    self.cnf.append(clause)

                # Update the next auxiliary variable ID
                self.next_aux_var = aux_var_manager.get_biggest_returned_auxvar() + 1

    def create_lee_distance_constraints(self):
        """Create Lee-distance constraints using order encoding.

        For each pair of codewords (i1, i2), position j and threshold v,
        create an auxiliary variable y_(i1,i2,j,v) with semantics:

            y_(i1,i2,j,v) <-> lee(codeword[i1][j], codeword[i2][j]) >= v

        Channeling is encoded in both directions:
        - Forward: supporting x-vars imply y.
        - Backward: y implies existence of a supporting symbol pair.

        Monotonicity is also enforced:
            y_(...,v) -> y_(...,v-1)

        Since local Lee distance d_j equals sum_{v>=1} [d_j >= v], the total Lee
        distance between two codewords is the sum of these y variables over all
        positions and thresholds.
        """
        symbols = sorted(self.alphabet, key=int)
        alphabet_size = len(symbols)
        max_lee_per_position = alphabet_size // 2

        if max_lee_per_position == 0:
            if self.distance_threshold > 0:
                self.cnf.append([])
            return

        def lee(s, t):
            a = int(s)
            b = int(t)
            diff = abs(a - b)
            return min(diff, alphabet_size - diff)

        for i1 in range(self.number_of_codewords):
            for i2 in range(i1 + 1, self.number_of_codewords):
                ordered_literals = []

                for pos in range(self.length_of_codeword):
                    y_by_threshold = {}

                    for v in range(1, max_lee_per_position + 1):
                        # y_(i1,i2,pos,v): Lee distance at this position is >= v
                        y_var = self.allocate_variables(1)
                        y_by_threshold[v] = y_var
                        supporting_pairs = []

                        for k1 in symbols:
                            for k2 in symbols:
                                if lee(k1, k2) >= v:
                                    x_i1 = self.codeword_vars[(i1, pos, k1)]
                                    x_i2 = self.codeword_vars[(i2, pos, k2)]
                                    supporting_pairs.append((x_i1, x_i2))

                                    # Forward channeling:
                                    # x_i1 AND x_i2 AND lee(k1,k2)>=v -> y_var
                                    self.cnf.append([-x_i1, -x_i2, y_var])

                        if supporting_pairs:
                            # Backward channeling:
                            # y_var -> OR over all supporting (x_i1 AND x_i2)
                            backward_clause = [-y_var]
                            for x_i1, x_i2 in supporting_pairs:
                                pair_var = self.allocate_variables(1)
                                self.cnf.append([-pair_var, x_i1])
                                self.cnf.append([-pair_var, x_i2])
                                self.cnf.append([-x_i1, -x_i2, pair_var])
                                backward_clause.append(pair_var)
                            self.cnf.append(backward_clause)
                        else:
                            # No support means this threshold cannot hold.
                            self.cnf.append([-y_var])

                        ordered_literals.append(WeightedLit(y_var, 1))

                    # Monotonicity: y_(...,v) -> y_(...,v-1)
                    for v in range(2, max_lee_per_position + 1):
                        self.cnf.append([-y_by_threshold[v], y_by_threshold[v - 1]])

                config = PBConfig()
                aux_var_manager = AuxVarManager(self.next_aux_var)
                clause_database = VectorClauseDatabase(config)
                constraint = PBConstraint(ordered_literals, pblib.GEQ, self.distance_threshold)

                pb2cnf = Pb2cnf(config)
                pb2cnf.encode(constraint, clause_database, aux_var_manager)
                for clause in clause_database.get_clauses():
                    self.cnf.append(clause)

                self.next_aux_var = aux_var_manager.get_biggest_returned_auxvar() + 1

    def solve(self, length_of_codeword, distance_threshold, number_of_codewords, distance_metric="hamming", timeout=None):     
        # Set parameters
        self.set_next_aux_var(1)
        self.length_of_codeword = length_of_codeword
        self.distance_threshold = distance_threshold
        self.number_of_codewords = number_of_codewords

        # Create variables
        self.create_variables()

        # Create constraints
        # 1. Each position in each codeword must have exactly one symbol
        self.create_exactly_one_symbol_per_position_constraints()
        # 2. distance constraints between codewords (hamming or lee)
        if distance_metric == "hamming":
            self.create_hamming_distance_constraints()
        elif distance_metric == "lee":
            self.create_lee_distance_constraints()
        else:
            raise ValueError(f"unknown distance metric '{distance_metric}'")

        # Update counts
        self.variables_count = self.next_aux_var - 1
        self.clauses_count = len(self.cnf.clauses)

        # Solve the CNF formula
        self.append_formula()  

        # solve() returns (is_sat, assignment_list). assignment_list is indexed by var-1
        if timeout is not None:
            print(f"Solving with timeout: {timeout}s")
            sat, assignment = solve_with_timeout(self, timeout)
            if sat == "timeout":
                print(f"Timeout reached after {timeout}s")
                self.timeout_occurred = True
                sat = False
                assignment = None
        else:
            sat, assignment = self.solver.solve()

        if sat:
            print("Solution found!")
            self.solution = []
            # assignment is a list of 0/1/None values; var numbers start at 1
            for i in range(self.number_of_codewords):
                codeword = []
                for j in range(self.length_of_codeword):
                    for symbol in self.alphabet:
                        var_id = self.codeword_vars[(i, j, symbol)]
                        val = assignment[var_id]
                        if val is True:
                            codeword.append(symbol)
                            break
                self.solution.append(''.join(codeword))
        else:
            print("No solution exists.")
            self.solution = None


class FleccWithSatIncremental(FleccWithSat):
    """Incremental SAT solver with count-activation variables p_M.

    Semantics:
    - p_M is True iff we are solving for at least M codewords.
    - Chain constraints enforce: p_M -> p_(M-1).
    - For codeword slot i (0-based), activation is p_(i+1).
      If slot i is inactive, all symbol variables in that slot are forced False.
    """

    def __init__(self, alphabet_size=2):
        super().__init__(alphabet_size=alphabet_size)
        self.current_num_codewords = 0
        self.base_constraints_added = False
        self.max_possible_codewords = 200
        self.count_activation_vars = {}

    def initialize_base_constraints(self, length_of_codeword, distance_threshold, distance_metric, max_possible_codewords=None):
        """Build base formula with p_M chain and conditional codeword activation."""
        self.set_next_aux_var(1)
        self.length_of_codeword = length_of_codeword
        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric
        self.current_num_codewords = 0

        if max_possible_codewords is not None:
            if max_possible_codewords <= 0:
                raise ValueError("max_possible_codewords must be > 0")
            self.max_possible_codewords = max_possible_codewords

        # Allocate p_M variables first: p_M means we require M codewords.
        self.count_activation_vars = {}
        for m in range(1, self.max_possible_codewords + 1):
            self.count_activation_vars[m] = self.allocate_variables(1)

        # Monotonic chain: p_M -> p_(M-1)
        for m in range(2, self.max_possible_codewords + 1):
            self.cnf.append([-self.count_activation_vars[m], self.count_activation_vars[m - 1]])

        # Pre-allocate all codeword symbol vars.
        self.codeword_vars = {}
        for i in range(self.max_possible_codewords):
            for j in range(self.length_of_codeword):
                for symbol in self.alphabet:
                    var_id = self.allocate_variables(1)
                    self.codeword_vars[(i, j, symbol)] = var_id

        # Conditional exactly-one per position based on p_(i+1).
        # inactive slot -> all symbol vars False
        # active slot -> exactly one symbol chosen
        for i in range(self.max_possible_codewords):
            activation_var = self.count_activation_vars[i + 1]
            for j in range(self.length_of_codeword):
                symbol_vars = [self.codeword_vars[(i, j, symbol)] for symbol in self.alphabet]

                for x_var in symbol_vars:
                    self.cnf.append([activation_var, -x_var])

                self.cnf.append([-activation_var] + symbol_vars)

                for idx1 in range(len(symbol_vars)):
                    for idx2 in range(idx1 + 1, len(symbol_vars)):
                        self.cnf.append([-activation_var, -symbol_vars[idx1], -symbol_vars[idx2]])

        # Flush base constraints to solver once.
        self.append_formula()
        self.cnf = CNF()
        self.base_constraints_added = True

    def _build_target_assumptions(self, num_codewords):
        """Build assumptions for an exact target count M.

        - p_M = True
        - p_k = False for every k > M
        """
        if num_codewords < 0 or num_codewords > self.max_possible_codewords:
            raise ValueError(
                f"num_codewords must be in [0, {self.max_possible_codewords}], got {num_codewords}"
            )

        assumptions = []
        if num_codewords > 0:
            assumptions.append(self.count_activation_vars[num_codewords])

        for m in range(num_codewords + 1, self.max_possible_codewords + 1):
            assumptions.append(-self.count_activation_vars[m])

        return assumptions

    def add_distance_constraints_for_codeword(self, codeword_index):
        """Add distance constraints between codeword[codeword_index] and all earlier codewords."""
        if codeword_index >= self.max_possible_codewords:
            raise ValueError(
                f"codeword_index={codeword_index} exceeds max_possible_codewords={self.max_possible_codewords}"
            )

        if self.distance_metric == "hamming":
            self._add_hamming_constraints(codeword_index)
        elif self.distance_metric == "lee":
            self._add_lee_constraints(codeword_index)
        else:
            raise ValueError(f"unknown distance metric '{self.distance_metric}'")

    def _add_hamming_constraints(self, new_idx):
        """Add Hamming distance constraints for codeword[new_idx] vs every earlier codeword."""
        for i in range(new_idx):
            diff_literals = []
            for k in range(self.length_of_codeword):
                xor_literals = []
                for symbol in self.alphabet:
                    var_i = self.codeword_vars[(i, k, symbol)]
                    var_j = self.codeword_vars[(new_idx, k, symbol)]
                    xor_lit = self.allocate_variables(1)
                    self.solver.add_xor_clause([var_i, var_j, xor_lit], False)
                    xor_literals.append(xor_lit)

                pos_diff = self.allocate_variables(1)
                for xl in xor_literals:
                    self.solver.add_clause([-xl, pos_diff])
                self.solver.add_clause([-pos_diff] + xor_literals)
                diff_literals.append(WeightedLit(pos_diff, 1))

            config = PBConfig()
            aux_var_manager = AuxVarManager(self.next_aux_var)
            clause_database = VectorClauseDatabase(config)
            constraint = PBConstraint(diff_literals, pblib.GEQ, self.distance_threshold)
            pb2cnf = Pb2cnf(config)
            pb2cnf.encode(constraint, clause_database, aux_var_manager)
            for clause in clause_database.get_clauses():
                self.solver.add_clause(clause)
            self.next_aux_var = aux_var_manager.get_biggest_returned_auxvar() + 1

    def _add_lee_constraints(self, new_idx):
        """Add Lee distance constraints for codeword[new_idx] vs every earlier codeword."""
        symbols = sorted(self.alphabet, key=int)
        alphabet_size = len(symbols)
        max_lee_per_position = alphabet_size // 2

        if max_lee_per_position == 0:
            if self.distance_threshold > 0:
                self.solver.add_clause([])
            return

        def lee(s, t):
            a, b = int(s), int(t)
            diff = abs(a - b)
            return min(diff, alphabet_size - diff)

        for i in range(new_idx):
            ordered_literals = []

            for pos in range(self.length_of_codeword):
                y_by_threshold = {}

                for v in range(1, max_lee_per_position + 1):
                    y_var = self.allocate_variables(1)
                    y_by_threshold[v] = y_var
                    supporting_pairs = []

                    for k1 in symbols:
                        for k2 in symbols:
                            if lee(k1, k2) >= v:
                                x_i = self.codeword_vars[(i, pos, k1)]
                                x_j = self.codeword_vars[(new_idx, pos, k2)]
                                supporting_pairs.append((x_i, x_j))

                                self.solver.add_clause([-x_i, -x_j, y_var])

                    if supporting_pairs:
                        backward_clause = [-y_var]
                        for x_i, x_j in supporting_pairs:
                            pair_var = self.allocate_variables(1)
                            self.solver.add_clause([-pair_var, x_i])
                            self.solver.add_clause([-pair_var, x_j])
                            self.solver.add_clause([-x_i, -x_j, pair_var])
                            backward_clause.append(pair_var)
                        self.solver.add_clause(backward_clause)
                    else:
                        self.solver.add_clause([-y_var])

                    ordered_literals.append(WeightedLit(y_var, 1))

                for v in range(2, max_lee_per_position + 1):
                    self.solver.add_clause([-y_by_threshold[v], y_by_threshold[v - 1]])

            config = PBConfig()
            aux_var_manager = AuxVarManager(self.next_aux_var)
            clause_database = VectorClauseDatabase(config)
            constraint = PBConstraint(ordered_literals, pblib.GEQ, self.distance_threshold)
            pb2cnf = Pb2cnf(config)
            pb2cnf.encode(constraint, clause_database, aux_var_manager)
            for clause in clause_database.get_clauses():
                self.solver.add_clause(clause)
            self.next_aux_var = aux_var_manager.get_biggest_returned_auxvar() + 1

    def solve_incremental(self, num_codewords, timeout=None):
        """Solve for an exact target count using p_M assumptions."""
        if not self.base_constraints_added:
            raise RuntimeError("Call initialize_base_constraints() first.")

        if num_codewords < self.current_num_codewords:
            raise ValueError(
                "Incremental solver expects non-decreasing num_codewords across iterations."
            )

        # Add distance constraints only when a codeword index is first activated.
        for k in range(self.current_num_codewords, num_codewords):
            self.add_distance_constraints_for_codeword(k)
        self.current_num_codewords = num_codewords

        assumptions = self._build_target_assumptions(num_codewords)

        self.number_of_codewords = num_codewords
        self.variables_count = self.next_aux_var - 1
        self.clauses_count = 0  # Not easily tracked for incremental mode
        self.timeout_occurred = False

        if timeout is not None:
            sat, assignment = solve_with_timeout(self, timeout, assumptions=assumptions)
            if sat == "timeout":
                self.timeout_occurred = True
                sat = False
                assignment = None
        else:
            try:
                sat, assignment = self.solver.solve(assumptions=assumptions)
            except TypeError:
                sat, assignment = self.solver.solve(assumptions)

        if sat:
            self.solution = []
            for i in range(num_codewords):
                codeword = []
                for j in range(self.length_of_codeword):
                    for symbol in self.alphabet:
                        var_id = self.codeword_vars[(i, j, symbol)]
                        val = assignment[var_id] if var_id < len(assignment) else None
                        if val is True:
                            codeword.append(symbol)
                            break
                self.solution.append(''.join(codeword))
        else:
            self.solution = None

        return sat, assignment


def validate_codewords(codewords, distance_threshold, distance_metric, alphabet_size):
    """Check pairwise distances of codewords and print results.

    Returns True if all pairs meet the threshold, False otherwise.
    """
    if not codewords:
        print("No codewords to validate")
        return False
    n = len(codewords)

    def lee_dist(a, b):
        m = len(a)  # assume equal length
        total = 0
        for x, y in zip(a, b):
            ai = int(x)
            bi = int(y)
            diff = abs(ai - bi)
            total += min(diff, alphabet_size - diff)
        return total

    print(f"Validating {n} codewords with {distance_metric} distance >= {distance_threshold}")
    ok = True
    for i in range(n):
        for j in range(i + 1, n):
            a = codewords[i]
            b = codewords[j]
            if distance_metric == "hamming":
                dist = sum(1 for x, y in zip(a, b) if x != y)
            else:
                dist = lee_dist(a, b)
            print(f"pair ({i},{j}): {a} vs {b} -> dist = {dist}")
            if dist < distance_threshold:
                ok = False
    if ok:
        print("All pairs meet the threshold.")
    else:
        print("Some pairs failed the threshold!")
    return ok


def solve_flecc(length_of_codeword, distance_threshold, number_of_codewords, distance_metric="hamming", alphabet_size=2, test=False, validate=False, timeout=None):
    start = timeit.default_timer()
    
    flecc_solver = FleccWithSat(alphabet_size=alphabet_size)
    flecc_solver.solve(length_of_codeword, distance_threshold, number_of_codewords, distance_metric, timeout)
    
    stop = timeit.default_timer()
    runtime = stop - start
    
    print("Codewords:", flecc_solver.solution)
    print(f"Runtime: {runtime:.2f}s")

    # validate distances if requested
    if validate and flecc_solver.solution:
        validate_codewords(flecc_solver.solution, distance_threshold, distance_metric, alphabet_size)
    
    # Prepare result data
    instance_name = f"FLECC_{length_of_codeword}_{distance_threshold}_{number_of_codewords}_{distance_metric}"
    codewords_str = str(flecc_solver.solution) if flecc_solver.solution else "None"
    status = "SAT" if flecc_solver.solution else "UNSAT"
    
    result = {
        'Instance': instance_name,
        'Variables': flecc_solver.variables_count,
        'Clauses': flecc_solver.clauses_count,
        'Runtime': runtime,
        'Codewords': codewords_str,
        'Status': status
    }
    
    if not test:
        # Save to Excel
        excel_file = 'FLECC.xlsx'
        results_df = pd.DataFrame([result])
        sheet_name = append_results_to_excel_by_metric_sheet(excel_file, results_df, distance_metric)
        print(f"Results saved to {excel_file} (sheet: {sheet_name})")
    
    return flecc_solver.solution


def solve_flecc_multi_sat(length_of_codeword, distance_threshold, number_of_codewords, distance_metric="hamming", alphabet_size=2, test=False, validate=False, max_iterations=100, timeout=600, max_timeout_retries=1):
    """
    Solve FLECC problem using multi-SAT approach.
    
    Starts with number_of_codewords as lower bound and iteratively increases 
    the number of codewords by 1 until SAT becomes unsatisfiable.
    
    Args:
        length_of_codeword: Length of each codeword
        distance_threshold: Minimum distance between codewords
        number_of_codewords: Lower bound on number of codewords
        distance_metric: 'hamming' or 'lee'
        test: If True, don't save to Excel
        validate: If True, validate the solution
        max_iterations: Maximum number of iterations to prevent infinite loops
        timeout: Time limit in seconds for each SAT solve (default: 600s)
        max_timeout_retries: Maximum retries when timeout occurs (default: 1)
    
    Returns:
        Tuple: (max_codewords, solution_dict, all_results)
            - max_codewords: Maximum number of codewords found
            - solution_dict: Solution details for the maximum
            - all_results: List of results for all iterations
    """
    print(f"\n{'='*70}")
    print("Starting Multi-SAT Solver")
    print(f"{'='*70}")
    print(f"Lower bound (initial number of codewords): {number_of_codewords}")
    print(f"Codeword length: {length_of_codeword}, Distance threshold: {distance_threshold}")
    print(f"Distance metric: {distance_metric}\n")
    
    current_num_codewords = number_of_codewords
    max_codewords_found = None
    best_solution = None
    all_results = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        current_timeout_retries = 0
        timeout_occurred = False
        
        while current_timeout_retries <= max_timeout_retries:
            print(f"[Iteration {iteration}] Trying {current_num_codewords} codewords...", end=" ", flush=True)
            
            start = timeit.default_timer()
            
            flecc_solver = FleccWithSat(alphabet_size=alphabet_size)
            flecc_solver.solve(length_of_codeword, distance_threshold, current_num_codewords, distance_metric, timeout)
            
            stop = timeit.default_timer()
            runtime = stop - start
            
            is_sat = flecc_solver.solution is not None
            status = "SAT" if is_sat else "UNSAT"
            
            if flecc_solver.timeout_occurred:
                timeout_occurred = True
                current_timeout_retries += 1
                if current_timeout_retries <= max_timeout_retries:
                    print(f"TIMEOUT ({runtime:.2f}s) - Retrying ({current_timeout_retries}/{max_timeout_retries})")
                    continue
                else:
                    print(f"TIMEOUT ({runtime:.2f}s) - Max retries reached, treating as UNSAT")
                    status = "TIMEOUT"
                    is_sat = False
            else:
                print(f"{status} ({runtime:.2f}s)")
                break
        
        # If we exhausted retries due to timeout, mark as timeout
        if timeout_occurred and current_timeout_retries > max_timeout_retries:
            status = "TIMEOUT"
        
        # Prepare result data for this iteration
        instance_name = f"FLECC_{length_of_codeword}_{distance_threshold}_{current_num_codewords}_{distance_metric}"
        codewords_str = str(flecc_solver.solution) if flecc_solver.solution else "None"
        
        result = {
            'Iteration': iteration,
            'Instance': instance_name,
            'Num_Codewords': current_num_codewords,
            'Variables': flecc_solver.variables_count,
            'Clauses': flecc_solver.clauses_count,
            'Runtime': runtime,
            'Codewords': codewords_str,
            'Status': status
        }
        all_results.append(result)
        
        if is_sat:
            # Solution found - update best solution and continue
            max_codewords_found = current_num_codewords
            best_solution = {
                'num_codewords': current_num_codewords,
                'codewords': flecc_solver.solution,
                'variables_count': flecc_solver.variables_count,
                'clauses_count': flecc_solver.clauses_count,
                'result': result
            }
            
            # Validate if requested
            if validate:
                validate_codewords(flecc_solver.solution, distance_threshold, distance_metric, alphabet_size)
            
            # Increment and try next
            current_num_codewords += 1
        else:
            # UNSAT or TIMEOUT - we found the maximum
            print(f"\n{'='*70}")
            print("Multi-SAT Search Complete!")
            print(f"{'='*70}")
            print(f"Maximum number of codewords found: {max_codewords_found}")
            if best_solution:
                print(f"Best solution: {best_solution['codewords']}")
            print(f"Total iterations: {iteration}")
            print(f"{'='*70}\n")
            break
    else:
        # Loop completed without finding UNSAT (reached max_iterations)
        print(f"\n{'='*70}")
        print("Multi-SAT Search Stopped (max iterations reached)")
        print(f"{'='*70}")
        print(f"Maximum number of codewords found: {max_codewords_found}")
        if best_solution:
            print(f"Best solution: {best_solution['codewords']}")
        print(f"Total iterations: {iteration}")
        print(f"{'='*70}\n")
    
    # Save results to Excel if not in test mode
    if not test and all_results:
        excel_file = 'FLECC_MultiSAT.xlsx'
        results_df = pd.DataFrame(all_results)
        sheet_name = append_results_to_excel_by_metric_sheet(excel_file, results_df, distance_metric)
        print(f"Results saved to {excel_file} (sheet: {sheet_name})")
    
    return max_codewords_found, best_solution, all_results


def solve_flecc_multi_sat_incremental(length_of_codeword, distance_threshold, number_of_codewords, distance_metric="hamming", alphabet_size=2, test=False, validate=False, max_iterations=100, timeout=600, max_timeout_retries=1):
    """
    Solve FLECC problem using incremental multi-SAT with learned clause reuse.

    Uses a single FleccWithSatIncremental instance for all iterations so that
    learned clauses accumulated by CryptoMiniSat are never discarded between
    successive solves.  Distance constraints for each new codeword are added
    directly to the solver (not rebuilt from scratch).

    Args:
        length_of_codeword: Length of each codeword
        distance_threshold: Minimum distance between codewords
        number_of_codewords: Lower bound on number of codewords
        distance_metric: 'hamming' or 'lee'
        test: If True, don't save to Excel
        validate: If True, validate the solution
        max_iterations: Maximum number of iterations to prevent infinite loops
        timeout: Time limit in seconds for each SAT solve (default: 600s)
        max_timeout_retries: Maximum retries when timeout occurs (default: 1)

    Returns:
        Tuple: (max_codewords, solution_dict, all_results)
    """
    print(f"\n{'='*70}")
    print("Starting Incremental Multi-SAT Solver (with Learned Clause Reuse)")
    print(f"{'='*70}")
    print(f"Lower bound (initial number of codewords): {number_of_codewords}")
    print(f"Codeword length: {length_of_codeword}, Distance threshold: {distance_threshold}")
    print(f"Distance metric: {distance_metric}")
    print("Learned clauses will be preserved between iterations\n")

    # One solver instance for the entire search.
    flecc_solver = FleccWithSatIncremental(alphabet_size=alphabet_size)

    c_multiplier = get_cp_mip_c_multiplier_placeholder()
    ub_max_possible_codewords = compute_upper_bound_max_possible_codewords(
        alphabet_size=len(flecc_solver.alphabet),
        length_of_codeword=length_of_codeword,
        distance_threshold=distance_threshold,
        requested_codewords=number_of_codewords,
        distance_metric=distance_metric,
        c_multiplier=c_multiplier,
    )

    # Keep allocation within search trajectory while respecting the UB formula.
    trajectory_cap = number_of_codewords + max_iterations
    max_possible_codewords = max(number_of_codewords, min(ub_max_possible_codewords, trajectory_cap))
    print(
        f"Upper bound estimate: UB={ub_max_possible_codewords}, "
        f"trajectory_cap={trajectory_cap}, "
        f"using max_possible_codewords={max_possible_codewords}"
    )

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

    while iteration < max_iterations:
        iteration += 1
        current_timeout_retries = 0
        timeout_occurred = False

        while current_timeout_retries <= max_timeout_retries:
            print(f"[Iteration {iteration}] Trying {current_num_codewords} codewords...", end=" ", flush=True)

            start = timeit.default_timer()
            sat, assignment = flecc_solver.solve_incremental(current_num_codewords, timeout)
            stop = timeit.default_timer()
            runtime = stop - start

            is_sat = flecc_solver.solution is not None
            status = "SAT" if is_sat else "UNSAT"

            if flecc_solver.timeout_occurred:
                timeout_occurred = True
                current_timeout_retries += 1
                if current_timeout_retries <= max_timeout_retries:
                    print(f"TIMEOUT ({runtime:.2f}s) - Retrying ({current_timeout_retries}/{max_timeout_retries})")
                    continue
                else:
                    print(f"TIMEOUT ({runtime:.2f}s) - Max retries reached, treating as UNSAT")
                    status = "TIMEOUT"
                    is_sat = False
            else:
                print(f"{status} ({runtime:.2f}s)")
                break

        if timeout_occurred and current_timeout_retries > max_timeout_retries:
            status = "TIMEOUT"

        instance_name = f"FLECC_{length_of_codeword}_{distance_threshold}_{current_num_codewords}_{distance_metric}_incremental"
        codewords_str = str(flecc_solver.solution) if flecc_solver.solution else "None"

        result = {
            'Iteration': iteration,
            'Instance': instance_name,
            'Num_Codewords': current_num_codewords,
            'Variables': flecc_solver.variables_count,
            'Clauses': flecc_solver.clauses_count,
            'Runtime': runtime,
            'Codewords': codewords_str,
            'Status': status
        }
        all_results.append(result)

        if is_sat:
            max_codewords_found = current_num_codewords
            best_solution = {
                'num_codewords': current_num_codewords,
                'codewords': flecc_solver.solution,
                'variables_count': flecc_solver.variables_count,
                'clauses_count': flecc_solver.clauses_count,
                'result': result
            }

            if validate:
                validate_codewords(flecc_solver.solution, distance_threshold, distance_metric, alphabet_size)

            current_num_codewords += 1
        else:
            print(f"\n{'='*70}")
            print("Incremental Multi-SAT Search Complete!")
            print(f"{'='*70}")
            print(f"Maximum number of codewords found: {max_codewords_found}")
            if best_solution:
                print(f"Best solution: {best_solution['codewords']}")
            print(f"Total iterations: {iteration}")
            print("Learned clauses were preserved throughout the search")
            print(f"{'='*70}\n")
            break
    else:
        print(f"\n{'='*70}")
        print("Incremental Multi-SAT Search Stopped (max iterations reached)")
        print(f"{'='*70}")
        print(f"Maximum number of codewords found: {max_codewords_found}")
        if best_solution:
            print(f"Best solution: {best_solution['codewords']}")
        print(f"Total iterations: {iteration}")
        print(f"{'='*70}\n")

    if not test and all_results:
        excel_file = 'FLECC_MultiSAT_Incremental.xlsx'
        results_df = pd.DataFrame(all_results)
        sheet_name = append_results_to_excel_by_metric_sheet(excel_file, results_df, distance_metric)
        print(f"Results saved to {excel_file} (sheet: {sheet_name})")

    return max_codewords_found, best_solution, all_results


if __name__ == "__main__":
    # Configuration - dễ dàng thay đổi các giá trị đầu vào ở đây
    config = {
        'length_of_codeword': 7,    # Độ dài codeword
        'distance_threshold': 3,    # Ngưỡng khoảng cách tối thiểu
        'number_of_codewords': 2,   # Số lượng codewords
        'alphabet_size': 2,         # q: kích thước bảng chữ cái (0..q-1)
        'distance_metric': 'hamming',  # 'hamming' hoặc 'lee'
        'test': False,              # True để chạy test (không lưu Excel)
        'validate': False,          # True để kiểm tra khoảng cách
        'multi_sat': False,         # True để sử dụng multi-SAT solver
        'incremental': False,       # True để sử dụng incremental multi-SAT (giữ learned clauses)
        'max_iterations': 50,       # Số lần lặp tối đa cho multi-SAT
        'timeout': 600,             # Giới hạn thời gian cho mỗi lần giải SAT (giây)
        'max_timeout_retries': 1    # Số lần thử lại tối đa khi timeout
    }
    
    # Override config với command line arguments nếu có
    parser = argparse.ArgumentParser(description="Solve Fixed Length Error Correcting Codes problem")
    parser.add_argument("--length", type=int, help=f"Length of codeword (default: {config['length_of_codeword']})")
    parser.add_argument("--distance", type=int, help=f"Minimum distance threshold (default: {config['distance_threshold']})")
    parser.add_argument("--codewords", type=int, help=f"Number of codewords (default: {config['number_of_codewords']})")
    parser.add_argument("--q", type=int, help=f"Alphabet size q (symbols 0..q-1), default: {config['alphabet_size']}")
    parser.add_argument("--metric", type=str, choices=["hamming", "lee"], help=f"Distance metric (default: {config['distance_metric']})")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no Excel saving)")
    parser.add_argument("--validate", action="store_true", help="Validate codeword distances")
    parser.add_argument("--multi-sat", action="store_true", help="Use multi-SAT solver to find maximum codewords")
    parser.add_argument("--incremental", action="store_true", help="Use incremental multi-SAT solver (preserves learned clauses between iterations)")
    parser.add_argument("--max-iterations", type=int, help=f"Maximum iterations for multi-SAT (default: {config['max_iterations']})")
    parser.add_argument("--timeout", type=int, help=f"Time limit in seconds for each SAT solve (default: {config['timeout']})")
    parser.add_argument("--max-timeout-retries", type=int, help=f"Maximum retries when timeout occurs (default: {config['max_timeout_retries']})")
    
    args = parser.parse_args()
    
    # Merge config với args (args sẽ override config nếu được cung cấp)
    final_config = config.copy()
    if args.length is not None:
        final_config['length_of_codeword'] = args.length
    if args.distance is not None:
        final_config['distance_threshold'] = args.distance
    if args.codewords is not None:
        final_config['number_of_codewords'] = args.codewords
    if hasattr(args, 'q') and args.q is not None:
        final_config['alphabet_size'] = args.q
    if args.metric is not None:
        final_config['distance_metric'] = args.metric
    if args.test:
        final_config['test'] = args.test
    if args.validate:
        final_config['validate'] = args.validate
    if args.multi_sat:
        final_config['multi_sat'] = args.multi_sat
    if args.incremental:
        final_config['incremental'] = args.incremental
    if args.max_iterations is not None:
        final_config['max_iterations'] = args.max_iterations
    if args.timeout is not None:
        final_config['timeout'] = args.timeout
    if hasattr(args, 'max_timeout_retries') and args.max_timeout_retries is not None:
        final_config['max_timeout_retries'] = args.max_timeout_retries

    if final_config['alphabet_size'] < 2:
        raise ValueError("--q must be >= 2")
    
    # Execute appropriate solver
    if final_config['incremental']:
        solve_flecc_multi_sat_incremental(
            length_of_codeword=final_config['length_of_codeword'],
            distance_threshold=final_config['distance_threshold'],
            number_of_codewords=final_config['number_of_codewords'],
            alphabet_size=final_config['alphabet_size'],
            distance_metric=final_config['distance_metric'],
            test=final_config['test'],
            validate=final_config['validate'],
            max_iterations=final_config['max_iterations'],
            timeout=final_config['timeout'],
            max_timeout_retries=final_config['max_timeout_retries']
        )
    elif final_config['multi_sat']:
        solve_flecc_multi_sat(
            length_of_codeword=final_config['length_of_codeword'],
            distance_threshold=final_config['distance_threshold'],
            number_of_codewords=final_config['number_of_codewords'],
            alphabet_size=final_config['alphabet_size'],
            distance_metric=final_config['distance_metric'],
            test=final_config['test'],
            validate=final_config['validate'],
            max_iterations=final_config['max_iterations'],
            timeout=final_config['timeout'],
            max_timeout_retries=final_config['max_timeout_retries']
        )
    else:
        solve_flecc(
            length_of_codeword=final_config['length_of_codeword'],
            distance_threshold=final_config['distance_threshold'],
            number_of_codewords=final_config['number_of_codewords'],
            alphabet_size=final_config['alphabet_size'],
            distance_metric=final_config['distance_metric'],
            test=final_config['test'],
            validate=final_config['validate'],
            timeout=final_config['timeout']
        )