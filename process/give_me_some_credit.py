import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.25


def main():
    path = "raw/"
    # Using only train file, because test file does not contain label information.
    trainFile = "givemecredit.csv"

    # Read Data from csv
    train_df = pd.read_csv(path + trainFile, index_col=False)

    # drop rows with missing values
    train_df = train_df.dropna(axis=0)

    continuous_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    target = ["SeriousDlqin2yrs"]

    # change labeling to be consistent with our notation
    label_map = {0: 1, 1: 0}
    train_df["SeriousDlqin2yrs"] = train_df["SeriousDlqin2yrs"].map(label_map)

    # get rid of outliers
    data_auxiliary = train_df[continuous_cols]
    idx = (np.abs(stats.zscore(data_auxiliary)) < 3.0).all(axis=1)
    train_df = train_df.iloc[idx]
    train_df = train_df[continuous_cols + target]
    train_df.to_csv("give_me_some_credit.csv", index=False)

    # NOTE train_df is original train split, read above comments. df_train, df_test is our split of train_df!
    df_train, df_test = train_test_split(train_df, test_size=TEST_SPLIT)
    df_train.to_csv("give_me_some_credit_train.csv", index=False)
    df_test.to_csv("give_me_some_credit_test.csv", index=False)


if __name__ == "__main__":
    main()
