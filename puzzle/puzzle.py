import itertools
import logging
import math
import random
from typing import List, Tuple


class Puzzle():

    def __init__(self,
                 grid: str = None,
                 grid_size: int = None,
                 seed: int = None):
        if grid:
            self.grid = self._parse_grid(grid)
            parsed_grid_size = math.sqrt(len(self.grid))
            if math.isqrt(len(self.grid)) != parsed_grid_size:
                raise ValueError("Grid must be square")
            if grid_size and parsed_grid_size != grid_size:
                raise ValueError("Grid size does not match given grid")
        else:
            self.grid = self._generate_grid(grid_size, seed)

        self.grid_size = grid_size if grid_size else int(parsed_grid_size)
        self.path_height = self.grid_size - 1
        self.path_size = self.path_height * 2

        self.solver_type = None
        self.path = None
        self.score = None

    def solve_grid_brute_force(self):
        """solve by calculating all paths and selecting the maximum"""
        self.solver_type = "BRUTE FORCE"

        if self.path_size > 22:
            logging.warning(
                "Brute force is too slow for this size, returning None")
            return None

        path_stubs = itertools.combinations(list(range(self.path_size)),
                                            self.path_height)

        paths = []
        for stub in path_stubs:
            path = [0] + list(
                itertools.accumulate([
                    self.grid_size if x in stub else 1
                    for x in range(self.path_size)
                ]))
            paths.append(path)

        scores = [sum(self.grid[x] for x in path) for path in paths]
        best_path = max(range(len(scores)), key=scores.__getitem__)
        self.score = scores[best_path]
        self.path = self.format_path(paths[best_path])

    def solve_grid_recursive(self):
        """solve by recursively calculating the best path"""
        self.solver_type = "RECURSIVE"

        # set up memoisation and grid
        memo = [[(0, []) for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]
        grid_2d = self._flat_to_2d()

        # recursive function to calculate the best path
        def helper(i, j) -> Tuple[int, List[Tuple[int, int]]]:
            if memo[i][j][0] > 0:
                return memo[i][j]
            if i == 0 and j == 0:
                memo[i][j] = (grid_2d[i][j], [(i, j)])
                return memo[i][j]
            max_sum = -1
            max_path = []
            if i > 0:
                sum_right, path_right = helper(i - 1, j)
                if sum_right > max_sum:
                    max_sum = sum_right
                    max_path = path_right
            if j > 0:
                sum_up, path_up = helper(i, j - 1)
                if sum_up > max_sum:
                    max_sum = sum_up
                    max_path = path_up
            max_path = max_path.copy()

            # coordinates are reversed because the grid is flipped in the recursive solution
            max_path.append((j, i))
            memo[i][j] = (max_sum + grid_2d[i][j], max_path)
            return memo[i][j]

        self.score, self.path = helper(self.grid_size - 1, self.grid_size - 1)

    def solve_grid_dijkstra(self):
        """solve by using dijkstra's algorithm"""
        self.solver_type = "DIJKSTRAS"

        # create a graph of the grid
        nodes = [x for x in range(self.grid_size**2)]
        edges = ([(x, x + 1)
                  for x in nodes if x % self.grid_size != self.grid_size - 1] +
                 [(x, x + self.grid_size)
                  for x in nodes if x < self.grid_size**2 - self.grid_size])
        max_dist = max(self.grid)

        # invert the grid so that the path is the shortest distance
        dist = [max_dist - x for x in self.grid]

        # dijkstra's algorithm initialisation
        target = nodes[-1]
        unvisited = set(nodes)
        path = []
        path_cost = {i: float('inf') for i, _ in enumerate(unvisited)}
        path_cost[0] = 0

        # find the shortest path using dijkstra's algorithm from source to target
        while unvisited:
            current = min(unvisited, key=path_cost.__getitem__)
            unvisited.remove(current)
            if current == target:
                break
            for edge in [
                    x for x in edges if current == x[0] and x[1] in unvisited
            ]:
                new_cost = path_cost[current] + dist[edge[1]]
                if new_cost < path_cost[edge[1]]:
                    path_cost[edge[1]] = new_cost
                    path.append((current, edge[1]))

        # extract the shortest path from the path list
        shortest_path = []
        for edge in reversed(path):
            if len(shortest_path) == 0 and edge[1] == target:
                shortest_path.append(edge)
            if len(shortest_path) > 0 and edge[1] == shortest_path[-1][0]:
                shortest_path.append(edge)

        self.path = [0] + [x[1] for x in reversed(shortest_path)]
        self.score = sum(self.grid[x] for x in self.path)
        self.path = self.format_path(self.path)

    def _flat_to_2d(self) -> list:
        return [
            self.grid[x * self.grid_size:(x * self.grid_size) + self.grid_size]
            for x in range(self.grid_size)
        ]

    def _generate_grid(self, grid_size, seed) -> List[List[int]]:
        random.seed(seed)
        return [
            math.ceil(random.random() * grid_size) for _ in range(grid_size**2)
        ]

    def _parse_grid(self, grid: str) -> List[List[int]]:
        return [int(x) for x in grid.split(",")]

    def print_grid(self):
        logging.info(
            f"Searching {math.comb(self.path_size, self.path_height)} paths")
        grid_2d = self._flat_to_2d()
        width = math.floor(math.log10(self.grid_size)) + 1
        for row in reversed(grid_2d):
            for col in row:
                print(f"{col:>{width}}", end=" ")
            print()
        print()

    def format_path(self, path) -> List[Tuple[int, int]]:
        return [(x % self.grid_size, x // self.grid_size) for x in path]

    def __repr__(self):
        print(self.solver_type)
        if self.score:
            print(self.score)
        if self.path:
            print(self.path)
        print()
