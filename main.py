import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.network import TwoLayeredPerceptron
from generate_data import DATA_DIR, SEPARATOR


def main() -> None:
    regressor = TwoLayeredPerceptron(5)
    with open(f"{DATA_DIR}/data.csv") as f:
        data = pd.read_csv(f, sep = SEPARATOR)
    x = data["x"].to_list()
    y = data["y"].to_list()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    regressor.feed(x_train, y_train)
    y_predicted = [regressor.feed_forward(sample) for sample in x_test]
    print(mean_absolute_error(y_test, y_predicted))
    # for i in range(len(x_test)):
    #     print(f"{x_test[i]}: {y_test[i]}; {regressor.feed_forward(x_test[i])}")


if __name__ == "__main__":
    main()