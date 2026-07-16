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

def get_z(data, row, weights):
    res = 0
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]) or col == "Index":
            continue
        res += weights[col] * row[col]
    return res


def adjust_weights_and_bias(data: pd.DataFrame, numeric_cols):
    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    weights = {}
    bias = {}
    epoch = 10
    learning_rate = 0.01
    indices = np.arange(len(data))  #create a list of indices
    x = data[numeric_cols].to_numpy()
    m = len(x)
    for house in houses:
        w = np.zeros(len(numeric_cols))
        b = 0.0
        y = (data["Hogwarts House"] == house).astype(int).to_numpy()
        for _ in range(epoch):
            np.random.shuffle(indices)
            for index in indices:
                x_i = x[index]
                y_i = y[index]
                z = np.dot(w, x_i) + b
                y_pred = sigmoid(z)
                errors = y_pred - y_i
                res_of_derivative_for_weights = errors * x_i
                w -= learning_rate * res_of_derivative_for_weights
                b -= learning_rate * errors
        weights[house] = {col: w[i] for i, col in enumerate(numeric_cols)}
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
    