import pandas as pd
import numpy as np
import json
from utils import refine_dataset


def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))


def save_in_json(weights, bias):
    data = {"weights": weights, "bias": bias}
    with open("weights_and_bias.json", "w") as f:
        json.dump(data, f, indent=4)


def adjust_weights_and_bias(data: pd.DataFrame, numeric_cols):
    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    weights = {houses[i]: {col: 0.0 for col in numeric_cols} for i in range(4)}
    bias = {houses[i]: 0.0 for i in range(4)}
    epoch = 2
    learnign_rate = 1
    x = data[numeric_cols].to_numpy()
    number_of_rows = len(x)
    batch_size = number_of_rows // 10
    eps = 1e-8

    for house in houses:
        w = np.zeros(len(numeric_cols))
        b = 0.0
        y = (data["Hogwarts House"] == house).astype(int).to_numpy()
        gradient_history_b = 0.0
        gradient_history_w = np.zeros_like(w)

        for _ in range(epoch):
            indices = np.random.permutation(number_of_rows)  # shuffle the indexes
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for i in range(0, number_of_rows, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                z = np.dot(x_batch, w) + b
                y_pred = sigmoid(z)
                errors = y_pred - y_shuffled[i:i + batch_size]

                res_of_derivative_for_bias = errors.mean()
                res_of_derivative_for_weights = np.dot(errors, x_batch) / batch_size

                gradient_history_b = gradient_history_b + res_of_derivative_for_bias ** 2
                gradient_history_w = gradient_history_w + res_of_derivative_for_weights ** 2

                w -= (learnign_rate * res_of_derivative_for_weights) / (np.sqrt(gradient_history_w) + eps)
                b -= (learnign_rate * res_of_derivative_for_bias) / (np.sqrt(gradient_history_b) + eps)

        weights[house] = {col: w[index] for index, col in enumerate(numeric_cols)}
        bias[house] = b
    save_in_json(weights, bias)


def main():
    try:
        data = pd.read_csv("./datasets/dataset_train.csv")
        numeric_cols = refine_dataset(data)
        adjust_weights_and_bias(data, numeric_cols)
    except KeyboardInterrupt:
        print("Program interrupted")
        exit(1)
    except FileNotFoundError:
        print("File not found")
        exit(1)


if __name__ == "__main__":
    main()
