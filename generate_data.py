import sys
import os
import random
import numpy as np


DEFAULT_SAMPLES_COUNT = 10000
SAMPLE_MIN = -1_000
SAMPLE_MAX = 1_000

WORKING_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))
DATA_DIR = f"{WORKING_DIR}/data"

SEPARATOR = "\t"


def target_func(x: float) -> float:
    return np.sin(2 * np.pi * x) + np.sin(5 * np.pi * x)


def generate_sample_args(samples_count) -> set:
    samples = set()
    while len(samples) < samples_count:
        samples.add(random.uniform(SAMPLE_MIN, SAMPLE_MAX))

    return samples


def main() -> None:
    samples_count: int
    if len(sys.argv) < 2:
        samples_count = DEFAULT_SAMPLES_COUNT
    else:
        try:
            samples_count = int(sys.argv[1])
        except ValueError:
            samples_count = DEFAULT_SAMPLES_COUNT

    sample_args = generate_sample_args(samples_count)

    with open(f"{DATA_DIR}/data.csv", "w+") as f:
        f.write(f"x{SEPARATOR}y\n")
        for arg in sample_args:
            f.write(f"{arg}{SEPARATOR}{target_func(arg)}\n")


if __name__ == "__main__":
    main()