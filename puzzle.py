import itertools
import logging
import math
import random


class Puzzle():

    def __init__(self, grid_size: int, seed: int):
        self.grid_size = grid_size
        self.path_height = grid_size - 1
        self.path_size = self.path_height * 2
        self.seed = seed

        logging.info(f"Searching {math.comb(self.path_size, self.path_height)} paths")

        random.seed(seed)
        self.grid = [math.ceil(random.random() * grid_size) for _ in range(grid_size ** 2)]

    def solve_grid_brute_force(self):
        """solve by calculating all paths and selecting the maximum"""
        path_stubs = itertools.combinations(list(range(self.path_size)), self.path_height)

        paths = []
        for stub in path_stubs:
            path = [0] + list(itertools.accumulate([self.grid_size if x in stub else 1 for x in range(self.path_size)]))
            paths.append(path)

        scores = [sum(self.grid[x] for x in path) for path in paths]
        best_path = max(range(len(scores)), key=scores.__getitem__)
        self.score = scores[best_path]
        self.path = paths[best_path]

    def __repr__(self):
        grid_2d = [
            self.grid[x * self.grid_size:(x * self.grid_size) + 5]
            for x in range(self.grid_size)
        ]
        [print(x) for x in reversed(grid_2d)]
        print(self.score)
        print([(x % self.grid_size, x // self.grid_size) for x in self.path])
