import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    csv = pd.read_csv(
        "./data/raw/data_2022.csv",
        encoding="unicode_escape",
        encoding_errors="backslashreplace",
    )
    csv.to_parquet("./data/raw/data_2022.parquet")

    train, test = train_test_split(csv, test_size=0.2)
    val, test = train_test_split(test, test_size=0.5)

    train.to_parquet("./data/split/train.parquet")
    test.to_parquet("./data/split/test.parquet")
    val.to_parquet("./data/split/val.parquet")
