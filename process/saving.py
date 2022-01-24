from sklearn.model_selection import train_test_split


def save_to_file(df, name, test_size=0.25):

    df_train, df_test = train_test_split(df, test_size=test_size)
    df.to_csv(f"{name}.csv", index=False)
    df_train.to_csv(f"{name}_train.csv", index=False)
    df_test.to_csv(f"{name}_test.csv", index=False)
    df_train.to_csv(f"{name}_train_index.csv", columns=[], header=False)
    df_test.to_csv(f"{name}_test_index.csv", columns=[], header=False)
