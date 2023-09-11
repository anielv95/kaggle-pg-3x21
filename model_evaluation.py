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
        if ".csv"==local_file[-4:]:
            print(local_file)
            train = pd.read_csv(f"data/{local_file}")
            y_train = train.pop("target")

            rforest = RandomForestRegressor(
                n_estimators=1000, max_depth=7, n_jobs=-1, random_state=42
            )
            rforest.fit(train, y_train)

            y_hat = rforest.predict(train)

            train_error = mean_squared_error(y_true=y_train, y_pred=y_hat, squared=False)
            if train.shape[0]<3348:

                y_hat = rforest.predict(strat_test_set)

                test_error = mean_squared_error(y_true=y_test, y_pred=y_hat, squared=False)
                print("train:",train_error,train.shape,"\n","test:",test_error,strat_test_set.shape,"\n\n")
                
                results[local_file] = {"train":{"error":train_error,"shape":train.shape},
                                       "test":{"error":test_error,"shape":strat_test_set.shape}}

            else:
                print("train:",train_error,train.shape,"\n\n")
                results[local_file] = {"train":{"error":train_error,"shape":train.shape},
                                       "test":{}}
                
    # Serializing json
    json_object = json.dumps(results, indent=4)
    with open("data/results.json", "w") as outfile:
        outfile.write(json_object)
    return True

def reading_json_file(relative_path="."):
    f = open(relative_path)
    data = json.load(f)
    return data

def writing_json_file(dictionary, relative_path="."):
    json_object = json.dumps(dictionary, indent=4)
    with open(relative_path, "w") as outfile:
        outfile.write(json_object)
    return True

def best_score():
    relative_path = 'data/results.json'
    results = reading_json_file(relative_path)
    min_val = 100.0
    winning_file = ""
    for file in results:
        if results[file]["test"]=={}:
            continue
        if min_val>results[file]["test"]["error"]:
            min_val = results[file]["test"]["error"]
            winning_file = file
    return {winning_file:min_val}


def getting_relevant_features(file="candidate1.csv"):
    """Getting relevant features"""
    train = pd.read_csv(f"data/{file}")
    y_train = train.pop("target")

    rforest = RandomForestRegressor(
        n_estimators=1000, max_depth=7, n_jobs=-1, random_state=42
    )
    rforest.fit(train, y_train)
    feature_importance_values = rforest.feature_importances_
    sorted_value_index = np.argsort(feature_importance_values)
    features = np.array(train.columns)
    feature_importance_dict = {}
    feature_importance_dict = {features[i]: feature_importance_values[i] for i in sorted_value_index}
    print(feature_importance_dict)
    return feature_importance_dict

def deleting_column_list(list_columns=["id","NH4_7"]):
    """deleting a column list"""
    train,test = preparing_data()
    for col in train.columns:
        if col in list_columns:
            train[col] = 1.0
    candidate = train.copy()
    ###
    y_test = test.pop("target")

    y_train = train.pop("target")  

    rforest = RandomForestRegressor(
        n_estimators=1000, max_depth=7, n_jobs=-1, random_state=42
    )
    rforest.fit(train, y_train)

    y_hat = rforest.predict(test)
    test_error = mean_squared_error(y_true=y_test, y_pred=y_hat, squared=False)
    print("test error:",test_error)

    y_hat = rforest.predict(train)
    train_error = mean_squared_error(y_true=y_train, y_pred=y_hat, squared=False)
    print("train error:",train_error)

    #reading ids:
    ids_path = "data/ids.json"
    try:
        dict_id = reading_json_file(relative_path=ids_path)
    except Exception as e: 
        print(e)
        dict_id = {"id":2}
    id = dict_id["id"]
    dict_id["id"] = id+1
    writing_json_file(dict_id,ids_path)
    file_name = f"candidate{id}"
    candidate.to_csv(
        f"data/{file_name}.csv", index=False
    )

    nomenclature_path = "data/nomenclature.json"
    try:
        nomenclature_dict = reading_json_file(relative_path=nomenclature_path)
        nomenclature_dict[id] = {"cols_dropped":list_columns}
    except Exception as e: 
        print(e)
        nomenclature_dict = {id:{"cols_dropped":list_columns}}
    writing_json_file(nomenclature_dict,nomenclature_path)

    results_path = "data/results.json"
    results = reading_json_file(relative_path=results_path)

    results[file_name] = {"train":{"error":train_error,"shape":train.shape},
                            "test":{"error":test_error,"shape":test.shape}}
    
    writing_json_file(results,results_path)

    feature_importance_values = rforest.feature_importances_
    sorted_value_index = np.argsort(feature_importance_values)
    features = np.array(train.columns)
    feature_importance_dict = {}
    feature_importance_dict = {features[i]: feature_importance_values[i] for i in sorted_value_index}

    print(
        feature_importance_dict
    )

    feature_importance_path = "data/feature_importance.json"
    writing_json_file(feature_importance_dict,feature_importance_path)

    return train_error,test_error


def preparing_data():
    """function to prepare data and starting with experiment and testing"""

    train = pd.read_csv("/gh/kaggle-pg-3x21/kaggle-pg-3x21/data/sample.csv")
    train["target_cat"] = pd.cut(
        train["target"],
        bins=[0.0, 4.0, 8.0, 12.0, 16.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    strat_train_set, strat_test_set = train_test_split(
        train, test_size=0.3, stratify=train["target_cat"], random_state=42
    )

    strat_test_set.drop("target_cat", axis=1, inplace=True)
    strat_train_set.drop("target_cat", axis=1, inplace=True)

    return strat_train_set,strat_test_set


def main():
    column_list = ["id","NH4_7"]
    feature_importance_path = "data/feature_importance.json"
    feature_importance_dict = reading_json_file(feature_importance_path)
    min_val = 1.088901992209112
    for col in feature_importance_dict:
        if col not in column_list:
            column_list.append(col)
            errors = deleting_column_list(column_list)
            if errors[1]<min_val:
                min_val = errors[1]
                print("improvement:",col)
    return min_val, col


def testing_candidate_data(train_ext,test_ext):
    train = train_ext.copy()
    test = test_ext.copy()
    y_train = train.pop("target")  
    y_test = test.pop("target")

    rforest = RandomForestRegressor(
        n_estimators=1000, max_depth=7, n_jobs=-1, random_state=42
    )
    rforest.fit(train, y_train)

    y_hat = rforest.predict(test)
    test_error = mean_squared_error(y_true=y_test, y_pred=y_hat, squared=False)
    print("test error:",test_error)

    y_hat = rforest.predict(train)
    train_error = mean_squared_error(y_true=y_train, y_pred=y_hat, squared=False)
    print("train error:",train_error)


    return train_error,test_error, rforest.feature_importances_

        
if __name__=="__main__":
    min_val,col = main()
    print("/n/n",min_val,col)

    