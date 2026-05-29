import pandas as pd 


def refine_dataset(data: pd.DataFrame):
    data["Best Hand"] = (data["Best Hand"] == "Right").astype(int)
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col != "Index"]
    for col in numeric_cols:
        median = data[col].median()
        data[col] = data[col].fillna(median)
        minn, maxx = data[col].min(), data[col].max()
        data[col] = (data[col] - minn) / (maxx - minn)

    return numeric_cols
