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
        print("here")
        weights = data["weights"]
        bias = data["bias"]
        return weights, bias
    except:
        print("error")
        exit(1)
    return 0,0
        
def get_z(data, row, weights):
    res = 0
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]) or col == "Index" or col == "Hogwarts House":
            continue
        res += weights[col] * row[col]
    return res


def main():
    print("start")
    weights, bias = get_weights_and_bias()
    # print(weights)
    data = pd.read_csv("datasets/dataset_test.csv")
    numeric_cols = refine_dataset(data)
    # for index, row in data.iterrows():
    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    houses_sig = {}
    for house in houses:
        z = get_z(data, data, weights[house]) + bias[house]
        houses_sig[house] = sigmoid(z)
    # print(houses_sig)
    prob_df = pd.DataFrame(houses_sig)
    # print(prob_df)
    data["Hogwarts House"] = prob_df.idxmax()
    print(data)
        
    return

if __name__ == "__main__": 
    main()