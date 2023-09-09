install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C exploring_data.py

format:
	black *.py

test:
	python -m pytest -vv -s --cov=exploring_data test_exploring_data.py

test2:
	python -m pytest -vv -s --cov=model_evaluation test_model_evaluation.py