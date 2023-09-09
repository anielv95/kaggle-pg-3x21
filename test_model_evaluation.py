"""This file is used to evaluate model_evaluation.py
"""
import model_evaluation as me

def test_listing_files():
    """function to test listing_files function
    """
    relative_path = "data"
    list_ = me.listing_files(relative_path)
    assert "candidate1.csv" in list_
    