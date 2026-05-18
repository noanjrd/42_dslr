import pandas as pd
import matplotlib.pyplot as plt

#  Which Hogwarts course has a homogeneous score distribution between all four houses?

def histogram(data: pd.DataFrame):
    means = data.groupby("Hogwarts House").mean(numeric_only=True)
    std = means.std()
    print(means)
    print(std)
    # mean = means.mean()
    # cv = std/mean
    # print(means)
    minn = std.idxmin()
    print("min : ", minn)
    # print(cv)

    Hufflepuff = data[data["Hogwarts House"] == "Hufflepuff"][minn]
    Ravenclaw = data[data["Hogwarts House"] == "Ravenclaw"][minn]
    Gryffindor = data[data["Hogwarts House"] == "Gryffindor"][minn]
    Slytherin = data[data["Hogwarts House"] == "Slytherin"][minn]
    plt.hist(Hufflepuff)
    plt.hist(Ravenclaw)
    plt.hist(Gryffindor)
    plt.hist(Slytherin)
    plt.xlabel(f"Grades of {minn}")
    plt.ylabel("Number of students")
    plt.title(f"Distribution of grades: {minn}")
    plt.show()
    # cv.plot()
    # plt.hist(cv)
    # plt.show()

                    
                
                


def main():
    # try:
    data = pd.read_csv("datasets/dataset_train.csv")
    histogram(data)
    # except Exception as  e:
    #     print(e)
    #     exit(1)

    return


if __name__ == "__main__":
    main()