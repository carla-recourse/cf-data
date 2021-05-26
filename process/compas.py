import json

import numpy as np
import pandas as pd


class Params:
    """Parameters object taken from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py

    Parameters
    ----------
    json_path : string

    Returns
    ----------
    Parameters object
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def get_and_preprocess_compas_data(params):
    """Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis

    Parameters
    ----------
    params : Params

    Returns
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome

    compas_df = pd.read_csv("raw/compas-scores-2years.csv", index_col=0)
    compas_df = compas_df.loc[
        (compas_df["days_b_screening_arrest"] <= 30)
        & (compas_df["days_b_screening_arrest"] >= -30)
        & (compas_df["is_recid"] != -1)
        & (compas_df["c_charge_degree"] != "O")
        & (compas_df["score_text"] != "NA")
    ]

    compas_df["length_of_stay"] = (
        pd.to_datetime(compas_df["c_jail_out"]) - pd.to_datetime(compas_df["c_jail_in"])
    ).dt.days
    X = compas_df[
        [
            "age",
            "two_year_recid",
            "c_charge_degree",
            "race",
            "sex",
            "priors_count",
            "length_of_stay",
        ]
    ]

    # if person has high score give them the _negative_ model outcome
    X["score"] = np.array(
        [
            NEGATIVE_OUTCOME if score == "High" else POSITIVE_OUTCOME
            for score in compas_df["score_text"]
        ]
    )

    # assign African-American as the protected class
    race_map = {
        "African-American": "African-American",
        "Asian": "Other",
        "Caucasian": "Other",
        "Other": "Other",
        "Hispanic": "Other",
        "Native American": "Other",
    }
    X["race"] = X["race"].map(race_map)

    return X


def main():
    params = Params("raw/experiment_params_compas.json")
    np.random.seed(params.seed)
    df = get_and_preprocess_compas_data(params)

    df.to_csv("compas.csv", index=False)


if __name__ == "__main__":
    # execute data preprocessing
    main()
