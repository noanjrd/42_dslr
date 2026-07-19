"""self_check.py - pre-submission sanity check.

dataset_test.csv has no ground-truth 'Hogwarts House' column (the school
holds that back for grading), so there's no direct way to measure the
classifier's accuracy on it before submission. This script instead holds
out a slice of dataset_train.csv (which does have labels), trains on the
rest with the same AdaGrad gradient descent as logreg_train.py, predicts
on the held-out slice, and reports accuracy the same way the school will:
Scikit-Learn's accuracy_score against the 98% minimum from the subject.

Usage:
    python logistic_regression/self_check.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from utils import refine_dataset
from logreg_train import sigmoid

HOUSES = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
VAL_FRACTION = 0.2
EPOCHS = 20
LEARNING_RATE = 1
EPS = 1e-8
REQUIRED_ACCURACY = 0.98
SEED = 42


def train(train_data: pd.DataFrame, numeric_cols):
    x = train_data[numeric_cols].to_numpy()
    number_of_rows = len(x)
    weights = {}
    bias = {}

    for house in HOUSES:
        w = np.zeros(len(numeric_cols))
        b = 0.0
        y = (train_data["Hogwarts House"] == house).astype(int).to_numpy()
        gradient_history_w = np.zeros_like(w)
        gradient_history_b = 0.0

        for _ in range(EPOCHS):
            z = np.dot(x, w) + b
            y_pred = sigmoid(z)
            errors = y_pred - y

            res_of_derivative_for_weights = np.dot(errors, x) / number_of_rows
            res_of_derivative_for_bias = errors.mean()

            gradient_history_w = gradient_history_w + res_of_derivative_for_weights ** 2
            gradient_history_b = gradient_history_b + res_of_derivative_for_bias ** 2

            w -= (LEARNING_RATE * res_of_derivative_for_weights) / (np.sqrt(gradient_history_w) + EPS)
            b -= (LEARNING_RATE * res_of_derivative_for_bias) / (np.sqrt(gradient_history_b) + EPS)

        weights[house] = w
        bias[house] = b

    return weights, bias


def predict(data: pd.DataFrame, numeric_cols, weights, bias):
    x = data[numeric_cols].to_numpy()
    scores = pd.DataFrame({
        house: sigmoid(np.dot(x, weights[house]) + bias[house])
        for house in HOUSES
    })
    return scores.idxmax(axis=1)


def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        numeric_cols = refine_dataset(data)

        rng = np.random.RandomState(SEED)
        indices = rng.permutation(len(data))
        n_val = int(len(data) * VAL_FRACTION)
        val_data = data.iloc[indices[:n_val]].reset_index(drop=True)
        train_data = data.iloc[indices[n_val:]].reset_index(drop=True)

        weights, bias = train(train_data, numeric_cols)
        predictions = predict(val_data, numeric_cols, weights, bias)

        accuracy = accuracy_score(val_data["Hogwarts House"], predictions)
        print(f"Train size: {len(train_data)}  |  Validation size: {len(val_data)}")
        print(f"Self-check validation accuracy: {accuracy * 100:.2f}%")
        print("Required minimum: 98.00% ->", "PASS" if accuracy >= REQUIRED_ACCURACY else "FAIL")
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
