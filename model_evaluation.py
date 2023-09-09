"""This file is used to evaluate the set of data to train a random forest regressor model
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import json


def listing_files(relative_path="."):
    """this function returns a file list from relative_path

    Args:
        relative_path (str): relative_path

    Returns:
        list: file list
    """
    file_list = os.listdir(relative_path)
    return file_list


def model_evaluation():
    file_list = listing_files("data")
    data = pd.read_csv("data/sample.csv")
    data["target_cat"] = pd.cut(
        data["target"],
        bins=[0.0, 4.0, 8.0, 12.0, 16.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    strat_train_set, strat_test_set = train_test_split(
        data, test_size=0.3, stratify=data["target_cat"], random_state=42
    )

    strat_test_set.drop("target_cat", axis=1, inplace=True)
    strat_train_set.drop("target_cat", axis=1, inplace=True)

    y_test = strat_test_set.pop("target")
    results = {}
    for local_file in file_list:
        if ".csv" == local_file[-4:]:
            print(local_file)
            train = pd.read_csv(f"data/{local_file}")
            y_train = train.pop("target")

            rforest = RandomForestRegressor(
                n_estimators=1000, max_depth=7, n_jobs=-1, random_state=42
            )
            rforest.fit(train, y_train)

            y_hat = rforest.predict(train)

            train_error = mean_squared_error(
                y_true=y_train, y_pred=y_hat, squared=False
            )
            if train.shape[0] < 3348:
                y_hat = rforest.predict(strat_test_set)

                test_error = mean_squared_error(
                    y_true=y_test, y_pred=y_hat, squared=False
                )
                print(
                    "train:",
                    train_error,
                    train.shape,
                    "\n",
                    "test:",
                    test_error,
                    strat_test_set.shape,
                    "\n\n",
                )

                results[local_file] = {
                    "train": {"error": train_error, "shape": train.shape},
                    "test": {"error": test_error, "shape": strat_test_set.shape},
                }

            else:
                print("train:", train_error, train.shape, "\n\n")
                results[local_file] = {
                    "train": {"error": train_error, "shape": train.shape},
                    "test": {},
                }

    # Serializing json
    json_object = json.dumps(results, indent=4)
    with open("data/results.json", "w") as outfile:
        outfile.write(json_object)
    return True


def reading_json_file(relative_path="."):
    f = open(relative_path)
    data = json.load(f)
    return data


def best_score():
    relative_path = "data/results.json"
    results = reading_json_file(relative_path)
    min_val = 100.0
    winning_file = ""
    for file in results:
        if results[file]["test"] == {}:
            continue
        if min_val > results[file]["test"]["error"]:
            min_val = results[file]["test"]["error"]
            winning_file = file
    return {winning_file: min_val}


if __name__ == "__main__":
    model_evaluation()
