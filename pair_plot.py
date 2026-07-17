import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# From this visualization, which features are you going to use for your logistic regression?
# The numerical ones


def pair_plot(data: pd.DataFrame):
    data_columns_list = data.select_dtypes(include="number").drop(columns="Index").columns.tolist()
    plot_data = data[['Hogwarts House'] + data_columns_list]  # here we keep te colomn 'Hogwarts House' and the numerical columns in data
    # print(plot_data)
    sns.pairplot(plot_data, hue='Hogwarts House',  palette="husl", markers=".")
    plt.show()


def main():
    """Load the training dataset and display the pair plot visualization."""
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        pair_plot(data)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
