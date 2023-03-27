from puzzle import puzzle
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def main():
    p = puzzle.Puzzle(10, 4)
    p.solve_grid_recursive()
    p.__repr__()
    p.solve_grid_brute_force()
    p.__repr__()


if __name__ == "__main__":
    main()