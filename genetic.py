import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, data):
        self.course_details = data["course_details"]
        self.courses = data["courses"]
        self.room_capacities = data["room_capacities"]
        self.rooms = data["rooms"]
        self.times = data["times"]
        self.course_conflicts = data["course_conflicts"]

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.PENALTIES = [1000,  # Constraint 1
                1000,  # Constraint 2
                500,  # Constraint 3
                1000,  # Constraint 4
                1000,  # Constraint 5
                ]

        self.population = [self.create_individual() for _ in range(population_size)]

        self.best_individual = None
        self.best_fitness = float('-inf')
        self.current_best_history = []

    def run(self):
        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in self.population]

            current_best = max(fitnesses)
            self.current_best_history.append(current_best)
            # print(f"{current_best=}")

            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.best_individual = self.population[fitnesses.index(current_best)]
            if self.best_fitness == 0:
                print(f"Found a constraint-free schedule in generation {gen}.")
                break

            selected = self.select_population(self.population, fitnesses, self.population_size)

            next_generation = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i+1) % self.population_size]
                child1, child2 = self.crossover(parent1, parent2)
                next_generation.extend([child1, child2])

            self.population = [self.mutate(child) for child in next_generation]

        return self.best_individual
    
    def fitness(self, individual):
        """
        Evaluates the schedule by counting room-time conflicts.
        A conflict occurs if more than one course is 
        scheduled in the same room at the same time.
        The fitness is defined as the negative number of conflicts (0 is best).
        """
        total_cost = 0

        total_cost += self.constraint1(individual)
        total_cost += self.constraint2(individual, self.course_details, self.room_capacities)
        total_cost += self.constraint3(individual, self.course_details)
        total_cost += self.constraint4(individual, self.course_details)
        total_cost += self.constraint5(individual, self.course_conflicts)

        return -total_cost

    def mutate(self, individual):
        """
        With given probability, reassign a random course to a new (time, room).
        """
        for course in individual.keys():
            if random.random() < self.mutation_rate:
                individual[course] = (random.choice(self.times), random.choice(self.rooms))
        return individual
    
    def crossover(self, parent1, parent2):
        """
        Create two children by exchanging assignments for each course with 50% chance.
        """
        child1, child2 = {}, {}
        for course in self.courses:
            if random.random() < 0.5:
                child1[course] = parent1[course]
                child2[course] = parent2[course]
            else:
                child1[course] = parent2[course]
                child2[course] = parent1[course]
        return child1, child2  

    def select_population(self, population, fitness, num_selected):
        """
        Select individuals from the population based on their fitness using 
        tournament selection.
        """
        selected = []
        tournament_size = 3
        for _ in range(num_selected):
            competitors = random.sample(list(zip(population, fitness)), tournament_size)

            winner = max(competitors, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def create_individual(self):
        """
        Creates a random schedule.
        Each course is randomly assigned a (time slot, room).
        """
        schedule = {}
        for course in self.courses:
            schedule[course] = (random.choice(self.times), random.choice(self.rooms))
        return schedule
    
    def constraint1(self, individual):
        """
        Constraint 1: Each course should be assigned to exactly one time slot and room.
        """
        assignments = {}
        for course, (t, r) in individual.items():
            assignments.setdefault((t, r), []).append(course)

        conflict_count = 0
        for course_list in assignments.values():
            if len(course_list) > 1:
                conflict_count += len(course_list) - 1
        return conflict_count * self.PENALTIES[0]

    def constraint2(self, individual, course_details, room_capacities):
        """
        Constraint 2: The number of students in a room at
        any time should not exceed the room capacity.
        """
        penalty = 0
        for course, (t, r) in individual.items():

            enrollment = course_details.get(course, {}).get('students', 0)
            room_cap = room_capacities.get(r, 0)
            if room_cap < enrollment:
                penalty += (enrollment - room_cap) * self.PENALTIES[1]
        return penalty

    def constraint3(self, individual, course_details):
        """
        For each department, ensure that the courses offered do not conflict in time.
        This assumes that courses in the same department are required and thus cannot
        be scheduled simultaneously.
        """
        dept_courses = {}
        for course, details in course_details.items():
            dept = details.get('department')
            if dept:
                dept_courses.setdefault(dept, []).append(course)
        conflict_count = 0
        for dept, courses_in_dept in dept_courses.items():
            times_assigned = [individual[course][0] for course in courses_in_dept if course in individual]

            for t in set(times_assigned):
                count = times_assigned.count(t)
                if count > 1:
                    conflict_count += count - 1
        return conflict_count * self.PENALTIES[2]
    

    def constraint4(self, individual, course_details):
        """
        Constraint 4: Each instructor should teach at most one course at a time.
        """
        teacher_courses = {}
        for course, details in course_details.items():
            teacher = details.get('instructor')
            if teacher:
                teacher_courses.setdefault(teacher, []).append(course)

        conflict_count = 0
        for teacher, courses_taught in teacher_courses.items():
            times_assigned = [individual[course][0] for course in courses_taught if course in individual]

            for t in set(times_assigned):
                count = times_assigned.count(t)
                if count > 1:
                    conflict_count += count - 1
        return conflict_count * self.PENALTIES[3]

    def constraint5(self, individual, course_conflicts):
        """
        Constraint 5: Courses that are required concurrently (specified in course_conflicts)
        should not be scheduled at the same time.
        
        For each conflict group, if two or more courses are scheduled simultaneously, 
        a penalty is added for each extra course.
        """
        conflict_count = 0
        for group in course_conflicts:
            times_assigned = [individual[course][0] for course in group if course in individual]
            for t in set(times_assigned):
                count = times_assigned.count(t)
                if count > 1:
                    conflict_count += (count - 1)
        return conflict_count * self.PENALTIES[4]
    
    def print_best_schedule(self):
        print("Best Schedule Found:")
        print(f"{'Course':<10} {'Instructor':<25} {'Time':<6} {'Room':<5} {'Room Cap':<10} {'Course Cap'}")
        print("=" * 65)
        for course, (time, room) in sorted(self.best_individual.items()):
            room_cap = self.room_capacities[room]
            course_cap = self.course_details[course]["students"]
            instructor = self.course_details[course]["instructor"]
            print(f"{course:<10} {instructor:<25} {time:<6} {room:<5} {room_cap:<10} {course_cap}")

        print("Cost:", -self.fitness(self.best_individual))

    def plot_fitness_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.current_best_history)), [i/1000.0 for i in self.current_best_history], marker='o', linestyle='-')
        plt.xlabel("Generation")
        plt.ylabel("Current Best Fitness")
        plt.title("Current Best Fitness Values by Generation")
        plt.grid(True)
        plt.show()
