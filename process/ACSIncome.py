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
    df = ca_data[columns].dropna()  # Unfortunately CARLA cannot handle rows with missing values (yet)

    save_to_file(df, "ACSIncome")


if __name__ == "__main__":
    main()
