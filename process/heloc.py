import pandas as pd


def main():

    df = pd.read_csv("raw/heloc.csv")

    # drop categorical columns
    df = df.drop(columns=["MaxDelq2PublicRecLast12M", "MaxDelqEver"])

    # write processed dataframe
    df.to_csv("heloc.csv", index=False)


if __name__ == "__main__":
    main()
