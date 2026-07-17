from sklearn.metrics import precision_score
import pandas as pd


def evaluate_precision(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='micro', zero_division=1)
    print("Here is the accuracy score: ", precision)
    #  Calculates precision for each class, then takes a simple average.
    #  micro: counts all predictions together.
    #  macro: computes precision for each class, then averages them equally.


def main():
    # print(weights)
    training = pd.read_csv("datasets/dataset_train.csv")
    predictions = pd.read_csv("houses.csv")
    evaluate_precision(training["Hogwarts House"], predictions["Hogwarts House"])


if __name__ == "__main__": 
    main()
