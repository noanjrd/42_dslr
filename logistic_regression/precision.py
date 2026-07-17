from sklearn.metrics import precision_score
import pandas as pd


def evaluate_precision(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='micro', zero_division=1)
    print("Here is the accuracy score: ", precision)
    #  Calculates precision for each class, then takes a simple average.
    #  micro: counts all predictions together.
    #  macro: computes precision for each class, then averages them equally.


def main():
    """Load the training labels and predicted houses, then compute precision.

    The evaluation compares the true Hogwarts House values against the
    predictions saved in houses.csv and prints the resulting score.
    """
    try:
        training = pd.read_csv("datasets/dataset_train.csv")
        predictions = pd.read_csv("houses.csv")
        evaluate_precision(training["Hogwarts House"], predictions["Hogwarts House"])
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
