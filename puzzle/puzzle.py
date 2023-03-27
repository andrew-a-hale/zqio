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

        logging.info(
            f"Searching {math.comb(self.path_size, self.path_height)} paths")

        random.seed(seed)
        self.grid = [
            math.ceil(random.random() * grid_size) for _ in range(grid_size**2)
        ]

    def solve_grid_brute_force(self):
        """solve by calculating all paths and selecting the maximum"""
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
        self.path = paths[best_path]

    def solve_grid_recursive(self):
        """solve by recursively calculating the best path"""
        memo = [[0 for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]
        grid_2d = self._flat_to_2d()

        def helper(i, j) -> int:
            max_sum = -1
            if i == 0 and j == 0: return grid_2d[i][j]
            if memo[i][j] > 0: return memo[i][j]
            if j > 0: max_sum = max(max_sum, grid_2d[i][j] + helper(i, j - 1))
            if i > 0: max_sum = max(max_sum, grid_2d[i][j] + helper(i - 1, j))
            memo[i][j] = max_sum
            return max_sum

        self.score = helper(self.grid_size - 1, self.grid_size - 1)
        self.path = None

    def _flat_to_2d(self) -> list:
        return [
            self.grid[x * self.grid_size:(x * self.grid_size) + self.grid_size]
            for x in range(self.grid_size)
        ]

    def __repr__(self):
        grid_2d = self._flat_to_2d()
        [print(x) for x in reversed(grid_2d)]
        print(self.score)
        if self.path:
            print([(x % self.grid_size, x // self.grid_size)
                   for x in self.path])
