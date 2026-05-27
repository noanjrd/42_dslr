import math
import pandas as pd


def sigmoid(z):
    return 1 / (1 + (math.e ** -(z)))

def get_pred(data, row, weights):
    res = 0
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]) or col == "Index":
            continue
        res += weights[col] * row[col]
    return res

def derivative_with_respect_to_bias(data : pd.DataFrame, house, weights, bias ):
    errors = []
    for index, row in data.iterrows():
        y = 0
        if row["Hogwarts House"] == house:
            y = 1
        errors.append(sigmoid(get_pred(data, row, weights) + bias ) - y)
    return sum(errors) / data.shape[0]

def derivative_with_respect_to_weight(data : pd.DataFrame, house, subject, weights, bias ):
    errors = []
    for index, row in data.iterrows():
        y = 0
        if row["Hogwarts House"] == house:
            y = 1
        errors.append((sigmoid(get_pred(data, row, weights) + bias) - y) * row[subject])
    return sum(errors) / data.shape[0]



def adjust_weights_and_bias(data: pd.DataFrame, numeric_cols):
    weights = [{ col: 0 for col in numeric_cols} for _ in range(4)]
    bias = [0] * 4
    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    nb_iterations = 1
    learning_rate = 0.01
    # print(weights)
    for i_house in range(len(houses)):
        for _ in range(nb_iterations):
            weights_temp = weights[i_house].copy()
            for col in numeric_cols:
                res_of_derivative_for_weights = derivative_with_respect_to_weight(data, houses[i_house], col, weights[i_house], bias[i_house])
                weights_temp[col] -= (learning_rate * res_of_derivative_for_weights)
            res_of_derivative_for_bias = derivative_with_respect_to_bias(data, houses[i_house], weights[i_house], bias[i_house])
            bias[i_house] -= (learning_rate * res_of_derivative_for_bias)
            weights[i_house] = weights_temp
    
    print(weights)
    print(bias)
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
    except KeyboardInterrupt:
        print("Program interrupted")
        exit(1)
    except FileNotFoundError:
        print("File not found")
        exit(1)
    
    
    return

if __name__ == "__main__":
    main()
    