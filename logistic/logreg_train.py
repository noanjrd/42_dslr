import math
import pandas as pd


def sigmoid(value, bias, weight):
    return 1 / (1 + (math.e ** -(weight * value + bias)))

def derivative_with_respect_to_bias(data : pd.DataFrame, house, ):
    summ = 0
    index = 0
    for col in data.colums: 
        if not pd.api.types.is_numeric_dtype(data[col]) and col != "Index":
            continue
        summ = 0
        for index, row in data.iterrows():
            index +=1
            y = 0
            if row["Hogwarts House"] == house:
                y = 1
            sigmoid()
            
            
    return



def adjust_weights_and_bias(data: pd.DataFrame):
    weights = [ [0] * 13  for _ in range(4)]
    bias = [ [0] for _ in range(4)]
    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    nb_iterations = 1000
    learning_rate = 0.01
    for house in houses:
        for _ in range(nb_iterations):
            res_of_derivative_for_bias = 0
            res_of_derivative_for_weights = 0
            weights[houses.index(house)] = weights[houses.index(house)] - (learning_rate * res_of_derivative_for_weights)
            bias[houses.index(house)] = bias[houses.index(house)] - (learning_rate * res_of_derivative_for_bias)
            
    
        # print(row)
    # print(weights)
    return

def refine_dataset(data: pd.DataFrame):
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]) and col != "Index":
            continue
        # print(col)
        median = data[col].median()
        data[col] = data[col].fillna(median)
        minn = data[col].min()
        maxx = data[col].max()
        data[col] = (data[col] - minn) / (maxx - minn)

    data["Best Hand"] = data["Best Hand"] == "Right"
    return

def main():
    heights = 0
    data = pd.read_csv("./datasets/dataset_train.csv")
    refine_dataset(data)
    adjust_weights_and_bias(data)
    # print(data)
    
    
    return

if __name__ == "__main__":
    main()
    