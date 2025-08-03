# -*- coding: utf-8 -*-
import numpy as np
import time

# Code to solve an integer program via branch and bound with ESG constraints
def branch_and_bound_with_esg(c, A, b, esg_scores, target_esg, int_variables=[], print_info=True):
    if not int_variables:  # All variables are set to be integers
        int_variables = [i for i in range(len(c))]

    num_nodes = 0

    n = c.shape[0]

    # Initialize current best solution and value
    current_best_value = np.inf
    current_best_solution = None

    # Solve the LP relaxation and check that it is feasible
    optimal_solution, LP_lower_bound = solve_LP(c, A, b, print_info=False)
    if not isinstance(optimal_solution, np.ndarray):
        print("Error: the LP relaxation is infeasible!")
        return [], np.inf, num_nodes

    # Stack to manage branch-and-bound nodes
    stack = [(0, optimal_solution, LP_lower_bound, c, A, b)]

    if print_info:
        print("Value of the initial linear relaxation:", LP_lower_bound)

    # Iterate until the stack is empty
    while stack:
        # Get the node with the best objective value
        node_id, lp_solution, lp_obj_value, c, A, b = stack.pop(np.argmin([node[2] for node in stack]))
        if print_info:
            print("\nCurrent node:", node_id, "LP optimal solution:", lp_solution, ", value =", lp_obj_value)

        if lp_obj_value >= current_best_value:
            continue  # Prune the branch

        # Check if the LP solution is integer
        if all(np.isclose(lp_solution[int_variables], np.round(lp_solution[int_variables]), atol=1e-5)):
            # If better than the current best, update
            if lp_obj_value < current_best_value:
                if print_info:
                    print("New best integer solution found")
                current_best_value = lp_obj_value
                current_best_solution = lp_solution
                if current_best_value == LP_lower_bound:
                    return current_best_solution[:n], current_best_value, num_nodes
        else:
            # Choose a branching variable (heuristic: choose the most fractional variable)
            branching_variable = int_variables[np.argmax([np.abs(lp_solution[i] - np.round(lp_solution[i])) for i in int_variables])]
            branching_value = np.floor(lp_solution[branching_variable])

            if print_info:
                print("Branch on variable", branching_variable)
            if branching_value < 0:
                branching_value = 0

            # Branch on the variable (add two subproblems)
            # LP 1:
            c_new, A_new, b_new = create_subproblem(c, A, b, branching_variable, branching_value)
            optimal_solution, optimal_value = solve_LP(c_new, A_new, b_new)

            if isinstance(optimal_solution, np.ndarray):
                num_nodes += 1
                stack.append((num_nodes, optimal_solution, optimal_value, c_new, A_new, b_new))

            # LP 2:
            c_new, A_new, b_new = create_subproblem(c, A, b, branching_variable, branching_value + 1, at_most=False)
            optimal_solution, optimal_value = solve_LP(c_new, A_new, b_new)

            if isinstance(optimal_solution, np.ndarray):
                num_nodes += 1
                stack.append((num_nodes, optimal_solution, optimal_value, c_new, A_new, b_new))

        # Update LP lower bound
        LP_lower_bound = min([node[2] for node in stack])

    return current_best_solution[:n], current_best_value, num_nodes

def find_starting_vtx(A, b, tol = 1e-4, print_info = False):
    m, n = A.shape
    
    if np.linalg.matrix_rank(A) != m:
        print('Error: matrix A is not full row rank')
        return []
    if any(b<-tol):
        print('Error: right-hand side vector b must be non-negative:  ', b)
        return []
    
    #Define an auxiliary LP
    A_I =  np.concatenate((A, np.eye(m)), axis=1)
    c_aux = np.concatenate((np.zeros(n), np.ones(m)))
    
    # The following is always a feasible basis for the auxiliary LP 
    feasible_basis = [i for i in range(n, n+m)]
    
    # Run the simplex method on the auxiliary LP
    opt_sol, opt_val, feasible_basis, _ = simplex_method(c_aux, A_I, b, feasible_basis, print_info = print_info)

    # The auxiliary LP has positive optimum if and only if the original LP is infeasible
    if opt_val > tol:
        if print_info:
            print('The LP is infeasible!')
        return []
    else: 
        if max(feasible_basis) < n: # the feasible basis contains only indices of the original variables
            return feasible_basis
        else:
            # The feasible basis contains indices of some slack variables: then we ignore those indices and add indices of the original variables until we obtain a basis (see function below)
            incomplete_basis = [i for i in feasible_basis if i<n]
            return complete_to_basis(A, incomplete_basis)

def complete_to_basis(A, basis):
    m, n = A.shape
    non_basis = [i for i in range(n) if i not in basis]

    while len(basis) < m:
        for j in non_basis:
            B = A[:, basis + [j]]
            if np.linalg.matrix_rank(B) == len(basis)+1: # if column j is linearly independent with the other columns of B, add it to B
                basis.append(j)
                non_basis.remove(j)
                break

    return sorted(basis)
    
def simplex_method(c, A, b, feasible_basis, tol = 1e-4, print_info = False):
    m, n = A.shape
    num_iterations = 0
    

    while True:
        num_iterations += 1
        
        if print_info:
            print('\nIteration '+str(num_iterations))

      # Get c_B, A_B and compute the current BFS and the corresponding reduced cost. 
        c_B = c[feasible_basis]
        A_B = A[:, feasible_basis]
        
        # Instead of computing inverses, we call linalg.solve from numpy to solve a linear system
        basic_feasible_solution = np.linalg.solve(A_B, b)
        reduced_costs = c - np.dot(c_B, np.linalg.solve(A_B, A))
        
        if print_info:
            print('Current BFS: '+str(basic_feasible_solution))
            print('Reduced costs: '+str(reduced_costs))
        

        # Check for optimality
        if np.all(reduced_costs >= -tol): # Optimal solution found
            optimal_solution = np.zeros(n)
            optimal_solution[feasible_basis] = basic_feasible_solution
            optimal_value = np.dot(c, optimal_solution)
            return optimal_solution, optimal_value, feasible_basis, num_iterations

        # Choose entering variable: smallest index whose reduced cost is negative
        entering_var = min([i for i in range(n) if reduced_costs[i]<-tol]) 

        # Compute the ratios for the leaving variable
        u = np.linalg.solve(A_B, A[:,entering_var])
        if np.all(u <= 0): # if all components of u are non-negative, the problem is unbounded and we stop
            print('The problem is unbounded')
            return [], -1*np.inf, feasible_basis, num_iterations 
        
        # Select as leaving variable the index minimizing x_i/u_i, among the indices where u is positive
        nonzero_idx =  [i for i in range(len(u)) if u[i]>tol]
        leaving_var = nonzero_idx[np.argmin([basic_feasible_solution[i]/u[i] for i in nonzero_idx])]
       
        # Update basic feasible solution
        feasible_basis[leaving_var] = entering_var
        # sort the basis in increasing order, to avoid confusion with indices
        feasible_basis = sorted(feasible_basis)

def solve_LP(c, A, b, print_info=False):
    feasible_basis = find_starting_vtx(A, b, print_info=print_info)

    if feasible_basis:
        optimal_solution, optimal_value, _, _ = simplex_method(c, A, b, feasible_basis, print_info=print_info)
        return optimal_solution, optimal_value
    else:
        return [], np.inf


def create_subproblem(c, A, b, idx, val, at_most=True):
    m, n = A.shape

    # Create a new column for the constraint matrix
    new_col = np.zeros((m, 1))
    A_new = np.hstack([A, new_col])

    # Create a new row for the constraint matrix
    new_row = np.zeros((1, n + 1))
    new_row[0, idx] = 1
    new_row[0, n] = 1 if at_most else -1
    A_new = np.vstack([A_new, new_row])

    # Update the right-hand side of the constraints
    b_new = np.hstack([b, val])

    # Update the objective function
    c_new = np.hstack([c, 0])

    return c_new, A_new, b_new

# Other auxiliary functions (simplex_method, find_starting_vtx, complete_to_basis) remain unchanged.


def write_IP_index_fund_with_esg(rho, n, q, exp_return, target_return, esg_scores, target_esg):
    
    #total number of variables: x | y | slack
    N = 2*(n**2 + n) +1

    # Objective function coefficients
    c = np.array([-rho[i, j] for i in range(n) for j in range(n)]+[0]*(N-n**2))

    # Constraint matrix A and rhs b
    A = []
    b = []
    
    # Sum(x_ij) = 1 for each i
    for i in range(n):
        row = [0] * (N)
        row[i * n: (i + 1) * n] = [1] * n  
        A.append(row)
        b.append(1)

    # x_ij - y_j <= 0 for each j
    for j in range(n):
        for i in range(n):
            row = [0] * (N)
            row[j + i * n] = 1
            row[n**2 + j] = -1
            row[n**2 +n + j + i * n] = 1
            A.append(row)
            b.append(0)
            
     # Sum(y_i) = q 
    row = [0] * (N)
    row[n**2: n**2+n] = [1] * n 
    A.append(row)
    b.append(q)
    
    # y_i <= 1
    for i in range(n): 
        row = [0] * (N)
        row[n**2 + i] = 1
        row[2*(n**2) +n+i] = 1
        A.append(row)
        b.append(1)
        
    # extra constraint: average return of selected assets >= target
    row = [0] * (N)
    row[n**2: n**2+n] = exp_return
    row[-1] = -1
    A.append(row)
    b.append(target_return)
    
    return c, np.array(A), np.array(b)


# Load data from Excel files
prices = np.loadtxt("prices.txt")
avg_returns = np.loadtxt("mom.txt")

# Given ESG scores
esg_scores = np.array([31.6, 20.1, 19.9, 20.0, 21.9, 15.3, 18.6, 24, 43.1, 31.8, 31.6, 22.0,
                        16.5, 23.8, 18.6, 27.8, 16.1, 26.7, 15.7, 22.2, 12.5, 11.2, 15.3, 12.5,
                        19.9, 12.2, 13.6, 18.1, 14.5, 30.5, 34.3, 24.3])

n = 33  # Number of companies
q_value = 11  # Number of components to be chosen

# Compute correlation matrix
rho_matrix = np.corrcoef(prices.T)

# Target return
target_return = q_value * np.mean(avg_returns[:n])

# Target ESG score
target_esg = np.mean(esg_scores[:n])
c, A, b = write_IP_index_fund_with_esg(rho_matrix, n, q_value, avg_returns[:n], target_return, esg_scores[:n], target_esg)
int_variables = [i for i in range(n**2 + n)]


start_time = time.time()
integer_solution, obj_value, num_nodes = branch_and_bound_with_esg(c, A, b, esg_scores[:n], target_esg, int_variables, print_info=False)
print("Integer solution:", integer_solution[n**2 : n**2 + n])
print("Objective value:", obj_value)
print("B&B nodes:", num_nodes)
print("Running time:", time.time() - start_time)



