import exploring_data as ed

def test_eval():
    errors = ed.eval()
    assert 1.0925>=errors[1]