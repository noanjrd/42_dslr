import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

# From this visualization, which features are you going to use for your logistic regression?

def pair_plot(data : pd.DataFrame):
    data_columns_list = data.select_dtypes(include="number").columns.tolist()
    if "Index" in data_columns_list:
        data_columns_list.remove("Index")
    plot_data = data[['Hogwarts House'] + data_columns_list]
    #  here we keep te colomn 'Hogwarts House' and the oens in the data_list
    # print(plot_data)
    sns.pairplot(plot_data, hue='Hogwarts House',  palette="husl", markers=".") 
    plt.show()
    return

def main():
    # try:
    data = pd.read_csv("datasets/dataset_train.csv")
    pair_plot(data)
    # except Exception as  e:
    #     print(e)
    #     exit(1)

    return


if __name__ == "__main__":
    main()