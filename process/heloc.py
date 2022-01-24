import pandas as pd
from saving import save_to_file


def main():

    df = pd.read_csv("raw/heloc.csv")

    # drop categorical columns
    df = df.drop(columns=["MaxDelq2PublicRecLast12M", "MaxDelqEver"])

    # write processed dataframe
    save_to_file(df, "heloc")


if __name__ == "__main__":
    main()
