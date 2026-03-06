from pycryptosat import Solver as CryptoMiniSat5
from pysat.formula import CNF
from pypblib import pblib
from pypblib.pblib import PBConfig, AuxVarManager, VectorClauseDatabase, WeightedLit, PBConstraint, Pb2cnf
import pandas as pd
import timeit
import os
import argparse

class FleccWithSat:
    """A class to solve the Fixed Length Error Correcting Codes problem using SAT solvers."""
    def __init__(self):
        self.solver = CryptoMiniSat5()
        self.cnf = CNF()
        self.solution = None
        self.next_aux_var = 1
        self.alphabet =  {'0', '1', '2', '3'}
        self.length_of_codeword = None
        self.distance_threshold = None
        self.number_of_codewords = None
        self.codeword_vars = None
        self.codewords = None
        self.variables_count = 0
        self.clauses_count = 0
    
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
        """Create constraints to ensure that the Lee distance between any two codewords is at least the distance threshold.

        The Lee distance for two symbols is the minimum of the forward and backward
        distance on the alphabet viewed as a cycle.  Because the alphabet here is
        represented by the strings `'1'..'4'` we convert them to integers and
        compute

            lee(a,b) = min(|a - b|, m - |a - b|)

        where ``m`` is the size of the alphabet.

        We create a fresh auxiliary variable for each pair of symbols with a
        positive Lee distance; that variable is equivalent to the conjunction of
        the corresponding symbol variables from the two codewords.  The
        weighted sum of these auxiliaries over all positions is then constrained
        to be at least ``self.distance_threshold`` using a pseudo-boolean
        constraint (just as the Hamming version does, but with weights > 1 when
        letters are two steps apart).
        """
        # helper that computes the Lee distance between two symbol strings
        def lee(s, t):
            a = int(s)
            b = int(t)
            diff = abs(a - b)
            m = len(self.alphabet)
            return min(diff, m - diff)

        for i in range(self.number_of_codewords):
            for j in range(i + 1, self.number_of_codewords):
                weighted_literals = []
                for k in range(self.length_of_codeword):
                    for symbol1 in self.alphabet:
                        for symbol2 in self.alphabet:
                            w = lee(symbol1, symbol2)
                            # zero‑distance pairs don't contribute to the sum;
                            # we can safely skip creating a variable for them.
                            if w == 0:
                                continue

                            var_i = self.codeword_vars[(i, k, symbol1)]
                            var_j = self.codeword_vars[(j, k, symbol2)]
                            pair_var = self.allocate_variables(1)

                            # pair_var <-> (var_i AND var_j)
                            self.cnf.append([-pair_var, var_i])
                            self.cnf.append([-pair_var, var_j])
                            self.cnf.append([-var_i, -var_j, pair_var])

                            weighted_literals.append(WeightedLit(pair_var, w))

                # construct a pseudo-boolean constraint requiring the total Lee
                # distance to be at least the threshold
                config = PBConfig()
                aux_var_manager = AuxVarManager(self.next_aux_var)
                clause_database = VectorClauseDatabase(config)
                constraint = PBConstraint(weighted_literals, pblib.GEQ, self.distance_threshold)

                pb2cnf = Pb2cnf(config)
                pb2cnf.encode(constraint, clause_database, aux_var_manager)
                for clause in clause_database.get_clauses():
                    self.cnf.append(clause)

                self.next_aux_var = aux_var_manager.get_biggest_returned_auxvar() + 1

    def solve(self, length_of_codeword, distance_threshold, number_of_codewords, distance_metric="hamming"):     
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

def validate_codewords(codewords, distance_threshold, distance_metric):
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
            total += min(diff, len(set('0123')) - diff)
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


def solve_flecc(length_of_codeword, distance_threshold, number_of_codewords, distance_metric="hamming", test=False, validate=False):
    start = timeit.default_timer()
    
    flecc_solver = FleccWithSat()
    flecc_solver.solve(length_of_codeword, distance_threshold, number_of_codewords, distance_metric)
    
    stop = timeit.default_timer()
    runtime = stop - start
    
    print("Codewords:", flecc_solver.solution)
    print(f"Runtime: {runtime:.2f}s")

    # validate distances if requested
    if validate and flecc_solver.solution:
        validate_codewords(flecc_solver.solution, distance_threshold, distance_metric)
    
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
        if os.path.exists(excel_file):
            existing_df = pd.read_excel(excel_file)
            new_df = pd.DataFrame([result])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = pd.DataFrame([result])
        
        updated_df.to_excel(excel_file, index=False)
        print(f"Results saved to {excel_file}")
    
    return flecc_solver.solution


def solve_flecc_multi_sat(length_of_codeword, distance_threshold, number_of_codewords, distance_metric="hamming", test=False, validate=False, max_iterations=100):
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
        print(f"[Iteration {iteration}] Trying {current_num_codewords} codewords...", end=" ", flush=True)
        
        start = timeit.default_timer()
        
        flecc_solver = FleccWithSat()
        flecc_solver.solve(length_of_codeword, distance_threshold, current_num_codewords, distance_metric)
        
        stop = timeit.default_timer()
        runtime = stop - start
        
        is_sat = flecc_solver.solution is not None
        status = "SAT" if is_sat else "UNSAT"
        
        print(f"{status} ({runtime:.2f}s)")
        
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
                validate_codewords(flecc_solver.solution, distance_threshold, distance_metric)
            
            # Increment and try next
            current_num_codewords += 1
        else:
            # UNSAT - we found the maximum
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
        results_df.to_excel(excel_file, index=False)
        print(f"Results saved to {excel_file}")
    
    return max_codewords_found, best_solution, all_results

if __name__ == "__main__":
    # Configuration - dễ dàng thay đổi các giá trị đầu vào ở đây
    config = {
        'length_of_codeword': 5,    # Độ dài codeword
        'distance_threshold': 3,    # Ngưỡng khoảng cách tối thiểu
        'number_of_codewords': 4,   # Số lượng codewords
        'distance_metric': 'lee',  # 'hamming' hoặc 'lee'
        'test': False,              # True để chạy test (không lưu Excel)
        'validate': False,          # True để kiểm tra khoảng cách
        'multi_sat': False,         # True để sử dụng multi-SAT solver
        'max_iterations': 100       # Số lần lặp tối đa cho multi-SAT
    }
    
    # Override config với command line arguments nếu có
    parser = argparse.ArgumentParser(description="Solve Fixed Length Error Correcting Codes problem")
    parser.add_argument("--length", type=int, help=f"Length of codeword (default: {config['length_of_codeword']})")
    parser.add_argument("--distance", type=int, help=f"Minimum distance threshold (default: {config['distance_threshold']})")
    parser.add_argument("--codewords", type=int, help=f"Number of codewords (default: {config['number_of_codewords']})")
    parser.add_argument("--metric", type=str, choices=["hamming", "lee"], help=f"Distance metric (default: {config['distance_metric']})")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no Excel saving)")
    parser.add_argument("--validate", action="store_true", help="Validate codeword distances")
    parser.add_argument("--multi-sat", action="store_true", help="Use multi-SAT solver to find maximum codewords")
    parser.add_argument("--max-iterations", type=int, help=f"Maximum iterations for multi-SAT (default: {config['max_iterations']})")
    
    args = parser.parse_args()
    
    # Merge config với args (args sẽ override config nếu được cung cấp)
    final_config = config.copy()
    if args.length is not None:
        final_config['length_of_codeword'] = args.length
    if args.distance is not None:
        final_config['distance_threshold'] = args.distance
    if args.codewords is not None:
        final_config['number_of_codewords'] = args.codewords
    if args.metric is not None:
        final_config['distance_metric'] = args.metric
    if args.test:
        final_config['test'] = args.test
    if args.validate:
        final_config['validate'] = args.validate
    if args.multi_sat:
        final_config['multi_sat'] = args.multi_sat
    if args.max_iterations is not None:
        final_config['max_iterations'] = args.max_iterations
    
    # Execute appropriate solver
    if final_config['multi_sat']:
        solve_flecc_multi_sat(
            length_of_codeword=final_config['length_of_codeword'],
            distance_threshold=final_config['distance_threshold'],
            number_of_codewords=final_config['number_of_codewords'],
            distance_metric=final_config['distance_metric'],
            test=final_config['test'],
            validate=final_config['validate'],
            max_iterations=final_config['max_iterations']
        )
    else:
        solve_flecc(
            length_of_codeword=final_config['length_of_codeword'],
            distance_threshold=final_config['distance_threshold'],
            number_of_codewords=final_config['number_of_codewords'],
            distance_metric=final_config['distance_metric'],
            test=final_config['test'],
            validate=final_config['validate']
        )