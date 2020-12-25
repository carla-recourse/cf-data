import pandas as pd


def main():

    df = pd.read_csv("raw/adult.csv")

    df = handle_null_values(df)


def handle_null_values(df):

    df = df.replace("?", None)
    df = df.dropna()

    return df


if __name__ == "__main__":
    main()
