import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.network import TwoLayeredPerceptron
from generate_data import DATA_DIR, SEPARATOR, CHARTS_DIR
import matplotlib.pyplot as plt


def main() -> None:
    random.seed(42)
    x: list
    y: list
    with open(f"{DATA_DIR}/data.csv") as f:
        data = pd.read_csv(f, sep = SEPARATOR)
        x = data["x"].to_list()
        y = data["y"].to_list()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    N_NEURON_COUNTS = 30
    errors = {}
    for neuron_count in range(1, N_NEURON_COUNTS + 1):
        regressor = TwoLayeredPerceptron(neuron_count)
        regressor.feed(x_train, y_train)
        y_predicted = [regressor.feed_forward(sample) for sample in x_test]
        errors[neuron_count] = mean_absolute_error(y_test, y_predicted)

        plt.figure()
        plt.plot(x_test, y_test, 'b.', label='y_true')
        plt.plot(x_test, y_predicted, 'r.', label='y_predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f"{CHARTS_DIR}/plot_{neuron_count:02}.png")
        plt.close()

        print(f"{neuron_count}/{N_NEURON_COUNTS}", end="\r")

    print("ERRORS (hidden_neuron_count: MAE)")
    for neuron_count, error in errors.items():
        print(f"{neuron_count}: {error}")

    print("\nBEST")
    best = min(errors.keys(), key=errors.get)
    print(f"{best}: {errors.get(best)}")


if __name__ == "__main__":
    main()