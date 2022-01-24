import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.25


def main():

    df = pd.read_csv("raw/adult.csv")

    df = df.drop(columns=["education"], axis="columns")
    df = df.replace("?", None).dropna()

    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

    df = binary_map_categories(df)

    df_train, df_test = train_test_split(df, test_size=TEST_SPLIT)
    df.to_csv("adult.csv", index=False)
    df_train.to_csv("adult_train.csv", index=False)
    df_test.to_csv("adult_test.csv", index=False)


def binary_map_categories(df):

    workclass_map = {
        "Federal-gov": "Non-Private",
        "Local-gov": "Non-Private",
        "Private": "Private",
        "Self-emp-inc": "Non-Private",
        "Self-emp-not-inc": "Non-Private",
        "State-gov": "Non-Private",
        "Without-pay": "Non-Private",
    }
    df["workclass"] = df["workclass"].map(workclass_map)

    occupation_map = {
        "Adm-clerical": "Managerial-Specialist",
        "Armed-Forces": "Other",
        "Craft-repair": "Other",
        "Exec-managerial": "Managerial-Specialist",
        "Farming-fishing": "Other",
        "Handlers-cleaners": "Other",
        "Machine-op-inspct": "Managerial-Specialist",
        "Other-service": "Other",
        "Priv-house-serv": "Other",
        "Prof-specialty": "Managerial-Specialist",
        "Protective-serv": "Other",
        "Sales": "Other",
        "Tech-support": "Other",
        "Transport-moving": "Other",
    }
    df["occupation"] = df["occupation"].map(occupation_map)

    marital_map = {
        "Divorced": "Non-Married",
        "Married-AF-spouse": "Married",
        "Married-civ-spouse": "Married",
        "Married-spouse-absent": "Non-Married",
        "Never-married": "Non-Married",
        "Separated": "Non-Married",
        "Widowed": "Non-Married",
    }
    df["marital-status"] = df["marital-status"].map(marital_map)

    race_map = {
        "White": "White",
        "Amer-Indian-Eskimo": "Non-White",
        "Asian-Pac-Islander": "Non-White",
        "Black": "Non-White",
        "Other": "Non-White",
    }
    df["race"] = df["race"].map(race_map)

    relationship_map = {
        "Unmarried": "Non-Husband",
        "Wife": "Non-Husband",
        "Husband": "Husband",
        "Not-in-family": "Non-Husband",
        "Own-child": "Non-Husband",
        "Other-relative": "Non-Husband",
    }
    df["relationship"] = df["relationship"].map(relationship_map)

    df.loc[df["native-country"] != "United-States", "native-country"] = "Non-US"
    df.loc[df["native-country"] == "United-States", "native-country"] = "US"

    df = df.dropna()

    return df


if __name__ == "__main__":
    main()
