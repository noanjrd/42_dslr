import pandas as pd
import sys
import math


def count(data, column):
    i = 0
    for row in data[column]:
        if not math.isnan(row):
            i+=1
    return float(i)

def mean(data, column):
    i = 0
    summ = 0
    for row in data[column]:
        if math.isnan(row):
            continue
        i+=1
        summ += float(row)
    return float(summ/i)


def std(data, column, mean):
    summ = 0
    i = 0
    for row in data[column]:
        if math.isnan(row):
            continue
        summ += (row - mean) ** 2
        i = 0
    return float((summ / 1) ** 0.5)  #sqrt

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

def quicksort(rows):
    return sorted(rows)
    

def calulate_25_50_75(data, column, count):
    c25 = None
    c50 = None
    c75 = None
    i = 0
    sorted_rows = quicksort(data[column])
    for row in sorted_rows:
        if math.isnan(row):
            continue
        if not c25 and i == int(count * 0.25):
            c25 = row
        elif not c50 and i == int(count * 0.5):
            c50 = row
        elif i == int (count * 0.75):
            c75 = row
            break
        i+=1
    return c25,c50,c75
    

def create_stats(data : pd.DataFrame):
    stats = {
        "Count":[],
        "Mean":[],
        "Std":[], # Standard Deviation
        "Min":[],
        "25%":[],
        "50%":[], 
        "75%":[],
        "Max":[]
    }
    
    for key,value in stats.items():
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
            elif key == "Std":
                stats["Std"].append(std(data, column, stats["Mean"][i]))
            elif key == "Min":
                min, max = min_and_max(data, column)
                stats["Min"].append(min)
                stats["Max"].append(max)
            elif key == "25%":
                c25,c50,c75 = calulate_25_50_75(data, column, stats["Count"][i])
                stats["25%"] = c25
                stats["50%"] = c50
                stats["75%"] = c75
                
                
            i+=1


    df = pd.DataFrame.from_dict(stats)
    df.index = [f"Feature {i}" for i in range(len(stats["Count"]))]
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