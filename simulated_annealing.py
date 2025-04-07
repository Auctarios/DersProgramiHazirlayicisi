import random
import math
import matplotlib.pyplot as plt

class SimulatedAnnealing:
    def __init__(self, data, initial_temperature=1000, cooling_rate=0.995, max_iter=10000):
        self.course_details = data["course_details"]
        self.courses = data["courses"]
        self.room_capacities = data["room_capacities"]
        self.rooms = data["rooms"]
        self.times = data["times"]
        self.course_conflicts = data.get("course_conflicts", [])
        
        # SA parameters
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        
        # Penalties for constraints:
        # Constraint 1: multiple courses in same (time, room)
        # Constraint 2: room capacity violations
        # Constraint 3: department courses conflict
        # Constraint 4: instructor teaching more than one course at same time
        # Constraint 5: explicitly defined course conflicts (courses required together)
        self.PENALTIES = [1000, 1000, 500, 1000, 1000]
        
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        
        # Initialize with a random schedule
        self.current = self.create_individual()

    def create_individual(self):
        """Creates a random schedule: each course is assigned a random (time, room) pair."""
        schedule = {}
        for course in self.courses:
            schedule[course] = (random.choice(self.times), random.choice(self.rooms))
        return schedule

    def fitness(self, individual):
        """
        Evaluates the schedule by summing constraint penalties.
        Returns the negative total cost (0 is optimal).
        """
        total_cost = 0
        total_cost += self.constraint1(individual)
        total_cost += self.constraint2(individual)
        total_cost += self.constraint3(individual)
        total_cost += self.constraint4(individual)
        total_cost += self.constraint5(individual)
        return -total_cost

    def constraint1(self, individual):
        """Constraint 1: Each course should have a unique (time, room) assignment."""
        assignments = {}
        for course, (t, r) in individual.items():
            assignments.setdefault((t, r), []).append(course)
        conflict_count = 0
        for course_list in assignments.values():
            if len(course_list) > 1:
                conflict_count += len(course_list) - 1
        return conflict_count * self.PENALTIES[0]

    def constraint2(self, individual):
        """Constraint 2: Room capacity must not be exceeded."""
        penalty = 0
        for course, (t, r) in individual.items():
            enrollment = self.course_details.get(course, {}).get("students", 0)
            room_cap = self.room_capacities.get(r, 0)
            if room_cap < enrollment:
                penalty += (enrollment - room_cap) * self.PENALTIES[1]
        return penalty

    def constraint3(self, individual):
        """
        Constraint 3: Courses in the same department should not conflict.
        (Assumes that courses in the same department are required concurrently.)
        """
        dept_courses = {}
        for course, details in self.course_details.items():
            dept = details.get("department")
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

    def constraint4(self, individual):
        """
        Constraint 4: Each instructor should teach at most one course at the same time.
        """
        teacher_courses = {}
        for course, details in self.course_details.items():
            teacher = details.get("instructor")
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

    def constraint5(self, individual):
        """
        Constraint 5: Courses that are required together (conflict groups) 
        should not be scheduled at the same time.
        """
        conflict_count = 0
        for group in self.course_conflicts:
            times_assigned = [individual[course][0] for course in group if course in individual]
            for t in set(times_assigned):
                count = times_assigned.count(t)
                if count > 1:
                    conflict_count += count - 1
        return conflict_count * self.PENALTIES[4]

    def neighbor(self, individual):
        """
        Generates a neighbor solution by randomly reassigning one course.
        """
        neighbor = individual.copy()
        course = random.choice(self.courses)
        new_time = random.choice(self.times)
        new_room = random.choice(self.rooms)
        neighbor[course] = (new_time, new_room)
        return neighbor

    def run(self):
        """
        Executes the simulated annealing algorithm.
        """
        current = self.current
        current_fitness = self.fitness(current)
        self.best_individual = current
        self.best_fitness = current_fitness
        temperature = self.initial_temperature

        for i in range(self.max_iter):
            # Cooling schedule
            temperature *= self.cooling_rate
            if temperature <= 1e-10:
                break

            candidate = self.neighbor(current)
            candidate_fitness = self.fitness(candidate)
            delta = candidate_fitness - current_fitness

            # Accept if the candidate is better, or with a probability exp(delta/temperature) if worse.
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current = candidate
                current_fitness = candidate_fitness

            if current_fitness > self.best_fitness:
                self.best_individual = current
                self.best_fitness = current_fitness

            self.fitness_history.append(current_fitness)

        return self.best_individual

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
        plt.plot(range(len(self.fitness_history)), [f/1000.0 for f in self.fitness_history], marker='o', linestyle='-')
        plt.xlabel("Iteration")
        plt.ylabel("Current Fitness")
        plt.title("Fitness History Over Iterations")
        plt.grid(True)
        plt.show()
