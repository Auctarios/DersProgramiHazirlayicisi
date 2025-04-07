import time
import json
from generate_data import generate_data
from genetic import GeneticAlgorithm
from simulated_annealing import SimulatedAnnealing
from constraint_satisfaction import ConstraintSatisfactionApproach

def run_experiments():
    # Define different dataset complexity levels (0: easy, 1: very constrained)
    complexity_levels = [0.2, 0.5, 0.7]
    # Define a problem size (number of courses)
    size = 100

    # Define parameter sets for the Genetic Algorithm (GA)
    ga_params_list = [
        {"population_size": 100, "generations": 1000, "mutation_rate": 0.3},
        {"population_size": 200, "generations": 2000, "mutation_rate": 0.5},
    ]
    
    # Define parameter sets for Simulated Annealing (SA)
    sa_params_list = [
        {"initial_temperature": 1000, "cooling_rate": 0.995, "max_iter": 5000},
        {"initial_temperature": 1000, "cooling_rate": 0.99, "max_iter": 10000},
    ]
    
    results = []  # To collect performance metrics
    
    for complexity in complexity_levels:
        # Generate data with the current complexity
        data = generate_data(size=size, complexity=complexity)
        print("=== Running experiments for dataset complexity:", complexity, "===")
        
        # --- Genetic Algorithm experiments ---
        for params in ga_params_list:
            ga = GeneticAlgorithm(
                data=data,
                population_size=params["population_size"],
                generations=params["generations"],
                mutation_rate=params["mutation_rate"]
            )
            start_time = time.time()
            ga.run()
            elapsed_time = time.time() - start_time
            best_cost = -ga.fitness(ga.best_individual)  # lower cost is better (0 is optimal)
            results.append({
                "algorithm": "GeneticAlgorithm",
                "complexity": complexity,
                "params": params,
                "best_cost": best_cost,
                "time": elapsed_time
            })
            print("GA Params:", params, "-> Best Cost:", best_cost, "Time: {:.4f}s".format(elapsed_time))
        
        # --- Simulated Annealing experiments ---
        for params in sa_params_list:
            sa = SimulatedAnnealing(
                data=data,
                initial_temperature=params["initial_temperature"],
                cooling_rate=params["cooling_rate"],
                max_iter=params["max_iter"]
            )
            start_time = time.time()
            sa.run()
            elapsed_time = time.time() - start_time
            best_cost = -sa.fitness(sa.best_individual)
            results.append({
                "algorithm": "SimulatedAnnealing",
                "complexity": complexity,
                "params": params,
                "best_cost": best_cost,
                "time": elapsed_time
            })
            print("SA Params:", params, "-> Best Cost:", best_cost, "Time: {:.4f}s".format(elapsed_time))
        
        # --- Constraint Satisfaction experiment ---
        # (Note: CSP is not parameterized and ideally finds a solution that satisfies all constraints.)
        csp = ConstraintSatisfactionApproach(data=data)
        start_time = time.time()
        solution = csp.run()
        elapsed_time = time.time() - start_time
        if solution is not None:
            best_cost = 0  # All constraints are satisfied in a valid CSP solution.
        else:
            best_cost = None
        results.append({
            "algorithm": "ConstraintSatisfaction",
            "complexity": complexity,
            "params": {},
            "best_cost": best_cost,
            "time": elapsed_time,
            "nodes_explored": csp.nodes_explored
        })
        print("CSP -> Best Cost:", best_cost, "Time: {:.4f}s".format(elapsed_time),
              "Nodes Explored:", csp.nodes_explored)
        print("-" * 80)
    
    # Save the results to a JSON file for further analysis or inclusion in your report.
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Experiment results saved to experiment_results.json")
    
if __name__ == "__main__":
    run_experiments()
