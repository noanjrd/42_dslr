import pandas as pd
import matplotlib.pyplot as plt

#  What are the two features that are similar?

def scatter_plot(data : pd.DataFrame):
    data = data.select_dtypes(include="number")
    corr_matrix  = data.corr(method="pearson")
    # print(corr_matrix)
    corr_unstack = corr_matrix.unstack() 
    # print(corr_unstack)
    corr_unstack = corr_unstack[corr_unstack < 1]#  removes self corelations

    corr_unstack = corr_unstack.sort_values(ascending=False)
    # print(corr_unstack)
    best = corr_unstack.index[0]
    # print(corr_list)
    data.plot.scatter(x=best[0], y=best[1])
    plt.show()
    return



def main():
    data = pd.read_csv("datasets/dataset_train.csv")
    scatter_plot(data)
    return 


if __name__ == "__main__":
    main()