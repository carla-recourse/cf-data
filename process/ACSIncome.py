"""
[Folktables](https://github.com/zykls/folktables) is a new dataset proposed to replace Adult.

```shell
pip install folktables
```
"""
import numpy as np

from folktables import ACSDataSource

from saving import save_to_file


def main():
    # Download raw dataset for California in 2018 (restricting dataset size for computational efficiency)
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)

    # Define classification label analogously to Adult Income classification task
    ca_data["label"] = np.where(ca_data["PINCP"] > 50000, 1, 0)

    # Remove other features that are not specified for the income classification task
    columns = ["label", "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
    df = ca_data[columns]

    df = binary_map_categories(df)

    save_to_file(df, "ACSIncome")


def binary_map_categories(df):
    df.loc[df["COW"].isin([1, 2]), "COW"] = "private"
    df.loc[~df["COW"].isin([1, 2]), "COW"] = "non-private"

    df.loc[df["SCHL"] < 18, "SCHL_copy"] = "high-school-level"
    df.loc[df["SCHL"] >= 18, "SCHL_copy"] = "beyond-high-school-level"

    df.loc[df["MAR"] == 1, "MAR"] = "married"
    df.loc[df["MAR"] != 1, "MAR"] = "non-married"

    df.loc[df["OCCP"] < 500, "OCCP_copy"] = "managerial-specialist"
    df.loc[df["OCCP"] >= 500, "OCCP_copy"] = "other"
    df["OCCP"] = df["OCCP_copy"]
    df.drop("OCCP_copy", axis=1)

    df.loc[df["POBP"] < 60, "POBP_copy"] = "US"
    df.loc[df["POBP"] >= 60, "POBP_copy"] = "non-US"
    df["POBP"] = df["POBP_copy"]
    df.drop("POBP_copy", axis=1)

    df.loc[df["RELP"] == 1, "RELP"] = "spouse"
    df.loc[df["RELP"] != 1, "RELP"] = "non-spouse"

    df.loc[df["RAC1P"] == 1, "RAC1P"] = "white"
    df.loc[df["RAC1P"] != 1, "RAC1P"] = "non-white"

    df = df.dropna()

    return df


if __name__ == "__main__":
    main()
