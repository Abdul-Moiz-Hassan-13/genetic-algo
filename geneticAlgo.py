import random
import datetime
import csv

# Check if a year is a leap year
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Classify a date into different categories
def classify_date(date_str):
    day, month, year = map(int, date_str.split("/"))

    if date_str in ["01/01/0000", "31/12/9999"]:
        return "Boundary Date"
    
    if not (0 <= year <= 9999):
        return "Year Out of Range"
    
    if not (1 <= month <= 12):
        return "Invalid Month"
    
    if not (1 <= day <= 31):
        return "Invalid Day"
    
    if year == 0000:
        if month == 2 and day == 29:
            return "Leap Year Date"
        elif month in [4, 6, 9, 11] and day == 30:
            return "Valid 30-Day Month"
        elif month in [1, 3, 5, 7, 8, 10, 12] and day == 31:
            return "Valid 31-Day Month"
        else:
            return "Regular Valid Date"
        
    try:
        datetime.datetime(year, month, day)
        if month == 2 and day == 29:
            return "Leap Year Date"
        if month in [4, 6, 9, 11] and day == 30:
            return "Valid 30-Day Month"
        if month in [1, 3, 5, 7, 8, 10, 12] and day == 31:
            return "Valid 31-Day Month"
        return "Regular Valid Date"
    except ValueError:
        if month in [4, 6, 9, 11] and day > 30:
            return "Day Exceeds Max of Month"
        if month == 2:
            return "Invalid Leap Year Date" if day == 29 and not is_leap_year(year) else "Day Exceeds Max of Month"
    
    return "Exception"

# Generate a random date in DD/MM/YYYY format
def generate_random_date():
    return f"{random.randint(1, 31):02}/{random.randint(1, 12):02}/{random.randint(0, 9999):04}"

# Initialize test cases with a mix of valid and invalid dates
def initialize_population(size=100):
    predefined_cases = [
        # Boundary cases
        "01/01/0000", "31/12/9999", "29/02/2020", "28/02/2021", "31/12/0000",
        # Valid cases
        "15/06/2023", "30/04/2023", "31/01/2023", "29/02/2020", "15/07/2023",
        "30/06/2023", "31/03/2023", "15/08/2023", "30/09/2023", "31/05/2023",
        # Invalid cases
        "31/04/2023", "29/02/2021", "32/01/2023", "00/01/2023", "13/01/2023",
        "31/06/2023", "30/02/2023", "31/11/2023", "29/02/2022", "31/09/2023"
    ]

    population = predefined_cases[:]
    while len(population) < size:
        population.append(generate_random_date())
    
    return population

# Calculate fitness scores to encourage diverse categories
def compute_fitness(population):
    category_counts = {}
    fitness_scores = {date: 0 for date in population}

    for date in population:
        category = classify_date(date)
        fitness_scores[date] -= category_counts.get(category, 0) * 10
        category_counts[category] = category_counts.get(category, 0) + 1

    return fitness_scores

# Select best test cases based on rank-based probability
def rank_based_selection(population, fitness_scores, num_selected=50):
    sorted_population = sorted(population, key=lambda date: fitness_scores[date], reverse=True)
    probabilities = [1 / (i + 1) for i in range(len(sorted_population))]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    return random.choices(sorted_population, weights=probabilities, k=num_selected)

# Perform crossover by swapping parts of the date
def crossover(population, crossover_rate=0.7):
    new_population = []
    
    for _ in range(len(population) // 2):
        parent1, parent2 = random.sample(population, 2)

        if random.random() < crossover_rate:
            d1, m1, y1 = parent1.split("/")
            d2, m2, y2 = parent2.split("/")
            swap_type = random.choice(["day", "month", "year"])

            if swap_type == "day":
                child1, child2 = f"{d2}/{m1}/{y1}", f"{d1}/{m2}/{y2}"
            elif swap_type == "month":
                child1, child2 = f"{d1}/{m2}/{y1}", f"{d2}/{m1}/{y2}"
            else:
                child1, child2 = f"{d1}/{m1}/{y2}", f"{d2}/{m2}/{y1}"

            new_population.extend([child1, child2])
        else:
            new_population.extend([parent1, parent2])

    return new_population

# Function to perform mutation
def mutate(population, mutation_rate=0.15):
    mutated_population = []

    for date in population:
        if random.random() < mutation_rate:
            d, m, y = map(int, date.split("/"))

            mutation_type = random.choice(["day", "month", "year"])

            if mutation_type == "day":
                d += random.choice([-3, -2, -1, 1, 2, 3])
            elif mutation_type == "month":
                m += random.choice([-1, 1])
            else:
                y += random.choice([-100, 100])

            mutated_population.append(f"{d:02}/{m:02}/{y:04}")
        else:
            mutated_population.append(date)

    return mutated_population

# Main function to run the genetic algorithm
def run_genetic_algorithm():
    max_generations = 100
    target_coverage = 9
    current_generation = 0

    population = initialize_population(1000)
    initial_categories = set(classify_date(date) for date in population)
    print(f"\nInitial Categories: {initial_categories}")
    print(f"Initial Category Coverage: {len(initial_categories)}/10")

    best_individuals = []

    while current_generation < max_generations:
        current_generation += 1

        fitness_scores = compute_fitness(population)
        selected_population = rank_based_selection(population, fitness_scores, num_selected=50)
        new_population = crossover(selected_population)
        mutated_population = mutate(new_population)
        population = mutated_population

        fitness_scores = compute_fitness(population)
        sorted_population = sorted(population, key=lambda date: fitness_scores[date], reverse=True)
        best_individuals = sorted_population[:10]

        final_categories = set(classify_date(date) for date in population)
        coverage = len(final_categories)

        print(f"\nGeneration {current_generation}:")
        print(f"Categories: {final_categories}")
        print(f"Category Coverage: {coverage}/10")

        if coverage >= target_coverage:
            print("\nTarget coverage (90%) achieved! Stopping early.")
            break

        if current_generation == max_generations:
            print("\nReached maximum generations (100). Stopping.")

    print("\nBest-Evolved Test Cases:")
    for date in best_individuals:
        category = classify_date(date)
        print(f"Date: {date}, Category: {category}")

    with open("best_evolved_test_cases.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Category"])
        for date in best_individuals:
            category = classify_date(date)
            writer.writerow([date, category])

run_genetic_algorithm()