import pandas as pd
import matplotlib.pyplot as plt

#  What are the two features that are similar?


def scatter_plot(data: pd.DataFrame):
    data = data.select_dtypes(include="number")
    corr_matrix = data.drop(columns="Index").corr(method="pearson")
    # print(corr_matrix)
    corr_unstack = abs(corr_matrix.unstack())  # reshapes it into a 1D Series
    # print(corr_unstack)
    corr_unstack = corr_unstack[corr_unstack < 1]  # removes self correlations

    corr_unstack = corr_unstack.sort_values(ascending=False)
    # print(corr_unstack) #  rounds 0.99999 to 1
    best = corr_unstack.index[0]
    data.plot.scatter(x=best[0], y=best[1])
    plt.show()


def main():
    """Load the training dataset and display the most correlated scatter plot."""
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        scatter_plot(data)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
