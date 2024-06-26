from src.network import TwoLayeredPerceptron


def main() -> None:
    regressor = TwoLayeredPerceptron(2)
    print(regressor.feed_forward(1))


if __name__ == "__main__":
    main()