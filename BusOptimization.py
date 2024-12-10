import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import networkx as nx
import csv


class BusRouteOptimization:
    def __init__(self):
        # Reading the stops from the file
        self.distance_matrix = {}
        path = 'C:\\Users\\alyss\\PycharmProjects\\Assignment%\\.venv\\'
        file = open(path + "DistanceData.csv", "r")
        reader = csv.reader(file)

        self.stops = next(reader)  # Assuming the first row has stop names
        for row in reader:
            for i in range(1, len(row)):
                self.distance_matrix[(row[0], self.stops[i])] = float(row[i])  # Use float for distances

        file.close()

        # Generate travel times
        self.travel_time = {}

        # Enforce symmetry and realism
        for stop1 in self.stops:
            for stop2 in self.stops:
                if stop1 != stop2:
                    if (stop1, stop2) not in self.travel_time:
                        # Generate a time based on distance (e.g., 10x shorter than distance or random if not provided)
                        distance = self.distance_matrix.get((stop1, stop2), random.uniform(1, 10))
                        time = max(1, int(distance * random.uniform(0.8, 1.2)))  # Add variability, ensure min time is 1
                        self.travel_time[(stop1, stop2)] = time
                        self.travel_time[(stop2, stop1)] = time  # Symmetry

        self.num_buses = 5
        self.starting_pop = {
            "Bus 1": ["Fisher Court", "Balfour", "Kappa Sigma/Chi Omega", "Lingelbach", "Kappa Delta/SRSC", "Neal Marshall", "3rd and Eagleson", "Biology Building", "Law School", "Kappa Delta/SRSC", "Lingelbach", "17th & Baker/Phi Sigma Kappa","Kappa Sigma/Chi Omega", "Balfour", "Fisher Court"],
            "Bus 2": ["RAHC", "Redbud", "Campus View", "Union Street Center", "7th & Union", "Willkie", "Forest","3rd and Eagleson", "Biology Building", "Law School", "IMU Shelter", "10th & Woodlawn", "Psychology", "Wells Library", "10th & Sunrise", "Campus View", "Redbud", "RAHC"],
            "Bus 3": ["Assembly Hall", "Briscoe", "Foster/McNutt", "Kelley School", "Wells Library", "3rd and Eagleson", "Biology Building", "Law School", "Kirkwood & Indiana", "IMU Shelter", "10th & Woodlawn", "Psychology", "Kelley School", "Foster/McNutt", "Briscoe", "Assembly Hall"],
            "Bus 4": ["Stadium", "Luddy Hall","7th & Woodlawn", "IMU Shelter", "Auditorium", "10th & Woodlawn", "Luddy Hall", "Stadium"],
            "Bus 5": ["Stadium", "Luddy Hall", "Psychology", "Kelley School", "Stadium"],
        }

        # DEAP setup
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", dict, fitness=creator.FitnessMulti)
        self.toolbox = base.Toolbox()

        # Registering all the functions I'll need later
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.create_individual
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selNSGA2)

    # Creating the individual creation method that makes only a valid individual
    # All stops are assigned to at least one bus.
    # Buses can have different numbers of stops.
    # Each bus has unique stops (no duplicates within a single bus's route).
    def create_individual(self):
        buses = {f"Bus {i + 1}": [] for i in range(self.num_buses)}
        shuffled_stops = self.stops[:]
        random.shuffle(shuffled_stops)
        # Assign stops to buses
        for i, stop in enumerate(shuffled_stops):
            bus = f"Bus {(i % self.num_buses) + 1}"
            buses[bus].append(stop)
        return self.starting_pop

    # Creating the evaluation function
    # Should look at a given route and calculate the average distance, wait time, and travel time
    def evaluate(self, individual):
        total_distance, total_wait_time, total_travel_time = 0, 0, 0

        # Repair invalid routes directly
        individual = self.repair_routes(individual)

        for bus, route in individual.items():
            if len(route) < 2:
                continue

            for i in range(len(route) - 1):
                stop1, stop2 = route[i], route[i + 1]
                if not self.isRouteValid(stop1, stop2):
                    return 1e10, 1e10, 1e10

                total_distance += self.distance_matrix.get((stop1, stop2), 1e10)
                total_travel_time += self.travel_time.get((stop1, stop2), 1e10)

        average_distance = total_distance / len(self.stops)
        average_travel_time = total_travel_time / len(self.stops)
        average_wait_time = average_travel_time  # Adjust logic as needed

        return average_distance, average_wait_time, average_travel_time

    # Check if a route between two stops is valid
    def isRouteValid(self, stop1, stop2):
        if stop1 or stop2 == "":
            return True
        if (stop1, stop2) not in self.distance_matrix and (stop2, stop1) not in self.distance_matrix:
            return False
        return True

    # Crossover function
    # Randomly selects two buses from each individual and swaps a random segment of their routes
    def crossover(self, ind1, ind2):
       # Select two random buses from each parent
        bus1, bus2 = random.sample(list(ind1.keys()), 1)[0], random.sample(list(ind2.keys()), 1)[0]

        # Get the routes for the selected buses
        route1, route2 = ind1[bus1], ind2[bus2]


        size = min(len(route1), len(route2))
        if size > 1:
            idx = random.randint(1, size - 1)
            ind1[bus1], ind2[bus2] = route1[:idx] + route2[idx:], route2[:idx] + route1[idx:]

        # Ensure all stops are covered
        return self.repair_routes(ind1), self.repair_routes(ind2)

    def mutate(self, individual):
        # Choose a random stop from the routes
        stop = random.choice(self.stops)
        current_bus = next(
            (bus for bus, route in individual.items()
             if
             stop in route and all(self.isRouteValid(stop, other_stop) for other_stop in route if other_stop != stop)),
            None
        )

        if current_bus is None:
            return individual,

        # Ensure the stop exists in the route before removing
        if stop in individual[current_bus]:
            individual[current_bus].remove(stop)
        else:
            raise ValueError(f"Stop {stop} not found in the route of bus {current_bus}.")

        # Ensure there is more than one bus available for reassignment
        if len(individual) > 1:
            # Select a new bus that is not the current one
            other_buses = [bus for bus in individual if bus != current_bus]
            new_bus = random.choice(other_buses)

            # Remove stop from the current bus and assign it to the new bus

            individual[new_bus].append(stop)
        else:
            # If only one bus exists, simply shuffle the order of stops
            random.shuffle(individual[current_bus])

            # Remove duplicates from each route
        for bus in individual.keys():
            unique_route = []
            seen_stops = set()
            for stop in individual[bus]:
                if stop not in seen_stops:
                    unique_route.append(stop)
                    seen_stops.add(stop)
            individual[bus] = unique_route

        # Ensure all stops are covered
        return self.repair_routes(individual),

    def repair_routes(self, individual):
        all_stops = set(self.stops)
        assigned_stops = set(stop for route in individual.values() for stop in route)
        missing_stops = all_stops - assigned_stops

        # Assign missing stops to random buses
        for stop in missing_stops:
            random_bus = random.choice(list(individual.keys()))
            individual[random_bus].append(stop)

        return individual

    def run(self, generations=100, population_size=50):
        population = self.toolbox.population(n=population_size)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg",
                       lambda values: tuple(sum(v[i] for v in values) / len(values) for i in range(len(values[0]))))

        stats.register("min", min)
        algorithms.eaMuPlusLambda(
            population, self.toolbox,
            mu=population_size, lambda_=2 * population_size,
            cxpb=0.6, mutpb=0.4, ngen=generations,
            stats=stats, halloffame=hof, verbose=True
        )
        return population, hof

def plot_bus_routes(individual, distance_matrix):
    G = nx.DiGraph()

    # Add nodes and edges for each bus
    for bus, route in individual.items():
        for i in range(len(route) - 1):
            stop1, stop2 = route[i], route[i + 1]
            distance = distance_matrix.get((stop1, stop2), distance_matrix.get((stop2, stop1), 0))
            G.add_edge(stop1, stop2, weight=distance, bus=bus)

    # Plot the graph
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    color_map = ['red', 'blue', 'green', 'orange', 'purple']  # Extend as needed

    plt.figure(figsize=(12, 8))
    for i, bus in enumerate(individual.keys()):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['bus'] == bus]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color_map[i % len(color_map)], width=2, alpha=0.7)

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightgray")
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Bus Routes")
    plt.show()


def plot_pareto_front(hof):
    # Extract fitness values from the Pareto front
    fitness_values = [ind.fitness.values for ind in hof]
    distances, wait_times, travel_times = zip(*fitness_values)

    # Create a 3D scatter plot for the Pareto front
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(distances, wait_times, travel_times, c="blue", label="Pareto Front", s=50)
    ax.set_xlabel("Average Distance")
    ax.set_ylabel("Average Wait Time")
    ax.set_zlabel("Average Travel Time")
    ax.set_title("Pareto Front: Bus Route Optimization")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    bro = BusRouteOptimization()
    population, hof = bro.run()

    print("Pareto Front:")
    for solution in hof:
        print(solution, bro.evaluate(solution))

    # Visualize the Pareto front
    plot_pareto_front(hof)

    # Visualize bus routes for the best solution in the Pareto front
    if hof:
        best_solution = hof[0]
        plot_bus_routes(best_solution, bro.distance_matrix)
