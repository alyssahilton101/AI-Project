This code is necessary in order to find an optimized route through analyzing the input data
and creating generations upon generations of sets of bus routes to contrast against the
current bus route.

This algorithm takes each existing stop, the distance between it and every other stop, and
the time between stops on existing routes, and analyzes them to create a route. In the 
source code, comments are utilized above each method to explain how they work. While it 
runs, the algorithm takes the most optimized routes for each generation, and creates
modifications of them, creating the next generation. Without this algorithm, doing this
manually would take years to create a route optimized to the extent our algorithm has.


To run the code:
1. Be sure to set up deap libary and other dependcies like random, csv, and matplotlib
2. Download the csv file and change the path in the code to reflect the locaiton of the csv file
