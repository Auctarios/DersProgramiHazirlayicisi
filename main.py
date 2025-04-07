from generate_data import generate_data
from genetic import GeneticAlgorithm
from simulated_annealing import SimulatedAnnealing
from constraint_satisfaction import ConstraintSatisfactionApproach


data = generate_data(size=20, complexity=0.65)


genetic = GeneticAlgorithm(data=data, population_size=200, generations=2000, mutation_rate=0.5)
sim_ann = SimulatedAnnealing(data=data, initial_temperature=1000, cooling_rate=0.99, max_iter=10000)
csp = ConstraintSatisfactionApproach(data=data)


genetic.run()
genetic.print_best_schedule()

sim_ann.run()
sim_ann.print_best_schedule()

csp.run()
csp.print_best_schedule()