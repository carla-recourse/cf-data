import numpy as np
import pandas as pd
from scipy import stats


def main():
    path = "raw/give_me_credit/"
    # Using only train file, because test file does not contain label information.
    trainFile = "cs-training.csv"

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


if __name__ == "__main__":
    main()
