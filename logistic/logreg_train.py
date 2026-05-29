import math
import pandas as pd
import json

def sigmoid(z):
    return 1 / (1 + (math.e ** -(z)))

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
    weights = {houses[i] : { col: 0.0 for col in numeric_cols} for i in range(4)}
    bias = {houses[i] : 0.0 for i in range(4)}
    nb_iterations = 1000
    learning_rate = 0.01
    for house in houses:
        for _ in range(nb_iterations):
            z = get_z(data, data,  weights[house]) + bias[house]
            y_pred = sigmoid(z)
            y = (data["Hogwarts House"] == house).astype(int)
            errors = y_pred - y
            # print(errors)
            for col in numeric_cols:
                res_of_derivative_for_weights = (errors * data[col]).mean()
                weights[house][col] -= (learning_rate * res_of_derivative_for_weights)
            bias[house] -= (learning_rate * errors.mean())
    save_in_json(weights, bias)
    return

def refine_dataset(data: pd.DataFrame):
    data["Best Hand"] = (data["Best Hand"] == "Right").astype(int)
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col != "Index"]
    for col in numeric_cols:
        median = data[col].median()
        data[col] = data[col].fillna(median)
        minn, maxx = data[col].min(), data[col].max()
        data[col] = (data[col] - minn) / (maxx - minn)

    return numeric_cols

def main():
    try:
        data = pd.read_csv("./datasets/dataset_train.csv")
        numeric_cols = refine_dataset(data)
        adjust_weights_and_bias(data, numeric_cols)
        # print(data)
    # print(data)
    except KeyboardInterrupt:
        print("Program interrupted")
        exit(1)
    except FileNotFoundError:
        print("File not found")
        exit(1)
    
    
    return

if __name__ == "__main__":
    main()
    