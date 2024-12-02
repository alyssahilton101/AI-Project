import random
import numpy
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from numpy.ma.extras import average


class BusRouteOptimization:
    # Setting up problem data. Using sample data for a single route for now
    # This will be replaced by the actual data and many buses later
    # List of all stops
    stops = ["A", "B", "C", "D", "E"]
    # Dictionary of distances between stops. Currently, they are tuples of the form (stop1, stop2): distance
    distance_matrix = {
        ("A", "D"): 10,
        ("D", "E"): 2,
        ("B", "D"): 1,
        ("B", "E"): 5,
        ("C", "E"): 15,
        ("A", "B"): 4,
        ("B", "C"): 6,
    }
    # Dictionary of wait times at each stop
    wait_time = {
        "A": 1,
        "B": 5,
        "C": 10,
        "D": 15,
        "E": 20,
    }
    # Dictionary of travel times between stops
    travel_time = {
        ("A", "D"): 5,
        ("D", "E"): 2,
        ("B", "D"): 1,
        ("B", "E"): 5,
        ("C", "E"): 9,
        ("A", "B"): 4,
        ("B", "C"): 6,
    }
    num_buses = 1

    # Creating the fitness function
        # The fitness function gives a tuple of the form (total_time, total_distance, total_wait_time)
        # for a given individual, giving its fitness in all three aspects
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    # Creating the individual class
         #Each individual is a dict of all the buses and their routes
         #For example, an individual could be of the form {1: ["A", "B", "C", "D", "E"]}
    creator.create("Individual", dict, fitness=creator.FitnessMulti)

    # Creating the individual creation method that makes only a valid individual
        #All stops are assigned to at least one bus.
        #Buses can have different numbers of stops.
        #Each bus has unique stops (no duplicates within a single bus's route).
        #TODO: There is a valid path between each stop in a bus's route.
    def create_individual(self, num_buses, stops):
       # Initialize an empty route for each bus
        buses = {f"Bus {i + 1}": [] for i in range(num_buses)}
        # Shuffle stops to randomize assignment
        shuffled_stops = stops[:]
        random.shuffle(shuffled_stops)
        # Randomly assign stops to the buses, ensuring all stops are covered
        for stop in shuffled_stops:
            bus = random.choice(list(buses.keys()))
            buses[bus].append(stop)
        return buses

    # Registering the individual and population creation methods to the toolbox for the genetic algorithm
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # TODO: Creating the evaluation function
        # Should look at a given route and calculate the average distance, wait time, and travel time
    def evaluate(self, individual, distance_matrix, wait_time, travel_time):
        average_distance = 0
        average_wait_time = 0
        average_travel_time = 0
        return average_distance, average_wait_time, average_travel_time
    toolbox.register("evaluate", evaluate)

    #Making a crossover function
    #TODO: Test/look over this
    def crossover(self, ind1, ind2):
        bus1, bus2 = random.sample(ind1.keys(), 2)
        route1, route2 = ind1[bus1], ind2[bus2]
        # Perform a simple swap of a random segment
        size = min(len(route1), len(route2))
        idx = random.randint(1, size - 1)
        ind1[bus1], ind2[bus2] = route1[:idx] + route2[idx:], route2[:idx] + route1[idx:]
        return ind1, ind2
    toolbox.register("mate", crossover)

    def mutate(self, individual):
       #TODO: not yet implemented
        pass
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)


def main():
    # TODO: not yet implemented
    pass


if __name__ == "__main__":
    # TODO: not yet implemented
    pass



