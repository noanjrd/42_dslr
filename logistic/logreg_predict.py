import pandas as pd
import json
import math


def sigmoid(z):
    return 1 / (1 + (math.e ** -(z)))


def get_weights_and_bias():
    try:
        with open("weights_and_bias", "r") as f:
            data = json.load(f)
        weights = data[weights]
        bias = data[bias]
        return weights, bias
    except:
        exit(1)
    return 0,0
        

def main():
    weights, bias = get_weights_and_bias()
    data = pd.read_csv("datasets/dataset_test.csv")
    # z = data[]
    # data["Hogwarts House"] = weights[] data[]
    return

if __name__ == "__main__": 
    main()