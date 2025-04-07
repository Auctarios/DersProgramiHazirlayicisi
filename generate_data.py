import json
import random

def generate_data(path="seed_data.json", size=50, complexity=0.5):
    """
    Generates data for the University Course Timetabling Problem based on a seed file.
    
    Parameters:
        path (str): Path to the seed JSON file containing base data.
        size (int): Desired number of courses in the generated instance.
        complexity (float): A float between 0 and 1, where 0 means minimal constraints and 
                            1 means maximal constraints.
                            
    Returns:
        Tuple: (course_details, courses, room_capacities, rooms, times, course_conflicts)
    """
    # Load seed data from file.
    with open(path, 'r') as f:
        seed_data = json.load(f)
    
    seed_course_details = seed_data["course_details"]
    # seed_courses = seed_data["courses"]
    seed_courses = list(seed_course_details.keys())
    seed_room_capacities = seed_data["room_capacities"]
    seed_rooms = seed_data["rooms"]
    seed_times = seed_data["times"]
    seed_course_conflicts = seed_data["course_conflicts"]
    
    # Determine the courses for the new instance.
    # If the requested size is less than or equal to the seed, sample directly.
    if size <= len(seed_courses):
        courses = random.sample(seed_courses, size)
    else:
        courses = seed_courses[:]  # start with all seed courses
        additional_needed = size - len(seed_courses)
        # Use seed departments to generate additional courses.
        departments = list({details["department"] for details in seed_course_details.values()})
        counter = 1
        while additional_needed > 0:
            dept = random.choice(departments)
            course_name = f"{dept}{1000 + counter}"  # generate a new course name
            if course_name not in seed_course_details and course_name not in courses:
                courses.append(course_name)
                # Generate random course details.
                base_students = random.randint(20, 40)
                students = base_students + int(complexity * random.randint(0, 20))
                instructor = f"Dr. {random.choice(['Alice', 'Bob', 'Carol', 'David', 'Eva'])} " \
                             f"{random.choice(['Smith', 'Johnson', 'Williams', 'Brown'])}"
                seed_course_details[course_name] = {
                    "department": dept,
                    "students": students,
                    "instructor": instructor
                }
                additional_needed -= 1
            counter += 1

    # Filter course_details for only the courses we are including.
    course_details = {course: details for course, details in seed_course_details.items() if course in courses}
    
    # Adjust rooms: With higher complexity, fewer rooms are available.
    num_rooms = max(1, int((1 - complexity) * len(seed_rooms)))
    rooms = random.sample(seed_rooms, num_rooms) if num_rooms < len(seed_rooms) else seed_rooms[:]
    room_capacities = {room: seed_room_capacities[room] for room in rooms}
    
    # Adjust timeslots: Fewer timeslots are available when complexity is high.
    num_times = max(1, int((1 - complexity) * len(seed_times)))
    times = random.sample(seed_times, num_times) if num_times < len(seed_times) else seed_times[:]
    
    # Filter conflict groups to include only courses in our current list.
    course_conflicts = []
    for group in seed_course_conflicts:
        filtered_group = [course for course in group if course in courses]
        if len(filtered_group) >= 2:
            course_conflicts.append(filtered_group)
    
    # Optionally, add additional random conflict groups if complexity is high.
    if complexity > 0.5:
        extra_groups = max(1, size // 20)
        for _ in range(extra_groups):
            group_size = random.randint(2, min(5, len(courses)))
            group = random.sample(courses, group_size)
            # Avoid duplicates.
            if group not in course_conflicts:
                course_conflicts.append(group)
    
    return {"course_details": course_details, 
            "courses": courses, 
            "room_capacities": room_capacities, 
            "rooms": rooms, 
            "times": times, 
            "course_conflicts": course_conflicts}
