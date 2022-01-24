import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.25


def main():

    df = pd.read_csv("raw/heloc.csv")

    # drop categorical columns
    df = df.drop(columns=["MaxDelq2PublicRecLast12M", "MaxDelqEver"])

    df_train, df_test = train_test_split(df, test_size=TEST_SPLIT)

    # write processed dataframe
    df.to_csv("heloc.csv", index=False)
    df_train.to_csv("heloc_train.csv", index=False)
    df_test.to_csv("heloc_test.csv", index=False)


if __name__ == "__main__":
    main()
