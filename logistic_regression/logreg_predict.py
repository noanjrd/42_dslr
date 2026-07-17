import pandas as pd
import json
import math
from utils import refine_dataset


def sigmoid(z):
    return 1 / (1 + (math.e ** -(z)))


def get_weights_and_bias():
    try:
        with open("weights_and_bias.json", "r") as f:
            data = json.load(f)
        weights = data["weights"]
        bias = data["bias"]
        return weights, bias
    except Exception:
        print("error")
        exit(1)
    return 0, 0


def get_z(data, row, weights):
    res = 0
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]) or col == "Index" or col == "Hogwarts House":
            continue
        res += weights[col] * row[col]
    return res


def main():
    """Load the trained model, compute predictions, and save them to CSV."""
    try:
        weights, bias = get_weights_and_bias()
        data = pd.read_csv("datasets/dataset_train.csv")
        refine_dataset(data)
        houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
        houses_sig = {}
        for house in houses:
            z = get_z(data, data, weights[house]) + bias[house]
            houses_sig[house] = sigmoid(z)
        prob_df = pd.DataFrame(houses_sig)
        new = prob_df.idxmax(axis=1)
        new.index.name = "Index"
        new.name = "Hogwarts House"
        new.to_csv("houses.csv")
    except Exception:
        print("Error")
        exit(1)


if __name__ == "__main__":
    main()
