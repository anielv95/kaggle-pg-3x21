"""This file is used to evaluate model_evaluation.py
"""
import model_evaluation as me


def test_listing_files():
    """function to test listing_files function"""
    relative_path = "data"
    list_ = me.listing_files(relative_path)
    assert "candidate1.csv" in list_


def test_reading_json_file():
    """function to test reading_json_file function"""
    relative_path = "data/results.json"
    dict_ = me.reading_json_file(relative_path)
    assert type({}) == type(dict_)
    assert type({}) == type(dict_["candidate1.csv"]["train"])


def test_best_score():
    """function to test test_score function"""
    results = me.best_score()
    assert type({}) == type(results)
    assert results["candidate1.csv"] >= 0.0
    assert results["candidate1.csv"] <= 1.5

def test_getting_relevant_features():
    """function to test getting_relevant_features"""
    features_dict = me.getting_relevant_features()
    assert len(features_dict)==36

def test_preparing_data():
    """function to test preparing_data"""
    train,test = me.preparing_data()
    assert train.shape[1]==37
    assert test.shape[1]==37

def test_writing_json_file():
    """function to test writing_json_file function"""
    relative_path = "data/testing_writing_json.json"
    target = {"key":"value"}
    boolean_value = me.writing_json_file(target,relative_path)
    dict_ = me.reading_json_file(relative_path)
    assert target == dict_

def test_deleting_column_list():
    """function to test deleting column list"""
    errors = me.deleting_column_list()
    assert errors[0]>0 and errors[1]>0