from puzzle import puzzle
import logging
import timeit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def main():
    p = puzzle.Puzzle(grid='1,4,2,4,4,3,3,2,3,1,5,1,3,5,2,3,2,4,3,4,2,3,5,4,1')
    p.print_grid()
    p.solve_grid_dijkstra()
    p.__repr__()
    p.solve_grid_recursive()
    p.__repr__()
    p.solve_grid_brute_force()
    p.__repr__()


def benchmark():
    call = "puzzle.Puzzle(grid_size=5, seed=1)"
    setup = f"from puzzle import puzzle; p = {call}"
    return {
        x: timeit.timeit(f"p.{x}()", setup=setup, number=100)
        for x in [
            "solve_grid_dijkstra",
            "solve_grid_recursive",
            "solve_grid_brute_force"
        ]
    }


if __name__ == "__main__":
    main()
    print(benchmark())