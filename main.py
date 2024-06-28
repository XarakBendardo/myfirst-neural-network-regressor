import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.network import TwoLayeredPerceptron
from generate_data import DATA_DIR, SEPARATOR, SAMPLE_MAX
import matplotlib.pyplot as plt


def main() -> None:
    regressor = TwoLayeredPerceptron(20)
    with open(f"{DATA_DIR}/data.csv") as f:
        data = pd.read_csv(f, sep = SEPARATOR)
    x = data["x"].to_list()
    y = data["y"].to_list()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    regressor.feed(x_train, y_train)
    y_predicted = [regressor.feed_forward(sample) for sample in x_test]
    print(mean_absolute_error(y_test, y_predicted))

    plt.figure()
    plt.plot(x_test, y_test, 'b.', label='Rzeczywiste dane')
    plt.plot(x_test, y_predicted, 'r.', label='Przewidywane dane')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Rzeczywiste vs Przewidywane dane')
    plt.savefig(f"{DATA_DIR}/plot.png")


if __name__ == "__main__":
    main()