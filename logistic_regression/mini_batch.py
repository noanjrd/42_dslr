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
    weights = {houses[i] : { col: 0.0 for col in numeric_cols} for i in range(4)}
    bias = {houses[i] : 0.0 for i in range(4)}
    epoch = 1000
    learnign_rate = 0.01
    x = data[numeric_cols].to_numpy()
    m = len(x)
    batch_size = m // 10
    for house in houses:
        w = np.zeros(len(numeric_cols))
        b = 0.0
        y = (data["Hogwarts House"] == house).astype(int).to_numpy()
        for _ in range(epoch):
            indices = np.random.permutation(m)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            for i in range(0, m, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = np.dot(x[i:i + batch_size], w) + b
                y_pred = sigmoid(y_batch)
                errors = y_pred - y_shuffled[i:i + batch_size]
                res_of_derivative_for_weights = np.dot(errors, x_batch) / batch_size
                w -= learnign_rate * res_of_derivative_for_weights
                b -= learnign_rate * errors.mean()
        weights[house] = {col:w[index] for index, col in enumerate(numeric_cols)}
        bias[house] = b
    save_in_json(weights, bias)
    return

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
    return

if __name__ == "__main__":
    main()
    