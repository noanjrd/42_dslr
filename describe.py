import pandas as pd
import sys
import math


def count(data, column):
    i = 0
    for row in data[column]:
        if not math.isnan(row):
            i += 1
    return float(i)


def mean(data, column):
    i = 0
    summ = 0
    for row in data[column]:
        if math.isnan(row):
            continue
        i += 1
        summ += float(row)
    return float(summ / i)


def variance(data, column, mean):
    summ = 0
    count = 0
    for row in data[column]:
        if math.isnan(row):
            continue
        summ += (row - mean) ** 2
        count += 1
    return float(summ / (count - 1))  if count > 0 else 0.0


def min_and_max(data, column):
    min = data[column][0]
    max = data[column][0]
    for row in data[column]:
        if math.isnan(row):
            continue
        if row < min:
            min = row
        if row > max:
            max = row
    return (float(min), float(max))


def ft_len(element):
    j = 0
    for _ in element:
        j += 1
    return j


def quicksort(rows, left, right):
    if right <= left:
        return
    pivot = rows[(left + right) // 2]
    l, r = left, right
    while l <= r:
        while rows[l] < pivot:
            l += 1
        while rows[r] > pivot:
            r -= 1
        if l <= r:
            rows[l], rows[r] = rows[r], rows[l]
            l += 1
            r -= 1
    quicksort(rows, left, r)
    quicksort(rows, l, right)
    return rows


def calulate_25_50_75(data, column, count):
    c25 = None
    c50 = None
    c75 = None
    i = 0
    sorted_rows = quicksort(data[column].copy(), 0, ft_len(data[column]) - 1)
    for row in sorted_rows:
        if math.isnan(row):
            continue
        if not c25 and i == int(count * 0.25):
            c25 = row
        elif not c50 and i == int(count * 0.5):
            c50 = row
        elif i == int(count * 0.75):
            c75 = row
            break
        i += 1
    return c25, c50, c75


def create_stats(data: pd.DataFrame):
    stats = {
        "Count": [],
        "Mean": [],
        "Variance": [],
        "Std": [],  # Standard Deviation
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": []
    }

    for key, value in stats.items():
        i = 0
        if key == "50%":
            break
        for column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue
            elif key == "Count":
                stats["Count"].append(count(data, column))
            elif key == "Mean":
                stats["Mean"].append(mean(data, column))
            elif key == "Variance":
                stats["Variance"].append(variance(data, column, stats["Mean"][i]))
            elif key == "Std":
                v = stats["Variance"][i] ** 0.5
                stats["Std"].append(v)
            elif key == "Min":
                min, max = min_and_max(data, column)
                stats["Min"].append(min)
                stats["Max"].append(max)
            elif key == "25%":
                c25, c50, c75 = calulate_25_50_75(data,
                                                  column, stats["Count"][i])
                stats["25%"] = c25
                stats["50%"] = c50
                stats["75%"] = c75
            i += 1
    df = pd.DataFrame.from_dict(stats)
    df.index = [f"Feature {i}" for i in range(1, len(stats["Count"]) + 1)]
    print(df.T)


def main():
    try:
        argv = sys.argv
        assert len(argv) == 2, "Wrong number of arguments"
        data = pd.read_csv(argv[1])
        create_stats(data)
    except AssertionError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
    return


if __name__ == "__main__":
    main()
