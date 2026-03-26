setup:
	pip install -r requirements.txt

process:
	python src/data/load_data.py

features:
	python src/features/build_features.py

train:
	python src/models/train.py

evaluate:
	python src/models/evaluate.py

app:
	streamlit run src/app/dashboard.py

all: process features train evaluate
