from puzzle import puzzle
import logging
import timeit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def main():
    p = puzzle.Puzzle(50, 1)
    p.print_grid()
    p.solve_grid_dijkstra()
    p.__repr__()
    p.solve_grid_recursive()
    p.__repr__()
    p.solve_grid_brute_force()
    p.__repr__()

def benchmark():
    call = "puzzle.Puzzle(50, 1)"
    return {
        "dijkstra": timeit.timeit("p.solve_grid_dijkstra()", setup=f"from puzzle import puzzle; p = {call}", number=1),
        "recursive": timeit.timeit("p.solve_grid_recursive()", setup=f"from puzzle import puzzle; p = {call}", number=1),
        "brute_force": timeit.timeit("p.solve_grid_brute_force()", setup=f"from puzzle import puzzle; p = {call}", number=1)
    }

if __name__ == "__main__":
    main()
    print(benchmark())