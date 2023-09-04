"""Module providingFunction printing python version."""
import exploring_data as ed


def test_eval():
    """function to test"""
    errors = ed.evaluation()
    assert 1.0925 >= errors[1]
