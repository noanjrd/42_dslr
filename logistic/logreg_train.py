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
    epoch = 2500
    learning_rate = 0.01
    x = data[numeric_cols].to_numpy()
    m = len(x)
    for house in houses:
        w = np.zeros(len(numeric_cols))
        b = 0.0
        for _ in range(epoch):
            z = np.dot(x,w) + b
            y_pred = sigmoid(z)
            y = (data["Hogwarts House"] == house).astype(int)
            errors = y_pred - y
            res_of_derivative_for_weights = np.dot(errors, x) / m
            w -= (learning_rate * res_of_derivative_for_weights)
            b -= (learning_rate * errors.mean())
        weights[house] = {col:w[i] for i, col in enumerate(numeric_cols)}
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
    