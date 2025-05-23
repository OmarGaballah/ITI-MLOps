dvc stage add -n preprocess \
    -d data/raw/train.csv \
    -d src/preprocessing.py \
    -o data/interim/train.csv \
    -o data/interim/val.csv \
    -o data/processed/train.csv \
    -o data/processed/val.csv \
    -o models/random_forest/columns_transformer.pkl \
    -p paths \
    -p dataset \
    -p names \
    -p hyperparameters \
    -p tuning \
    -p pipeline_config \
    --force \
    uv run src/preprocessing.py

dvc stage add -n train \
    -d data/processed/train.csv \
    -d src/train.py \
    -d models/random_forest/columns_transformer.pkl \
    -o models/random_forest/random_forest.pkl \
    -p paths \
    -p dataset \
    -p names \
    -p hyperparameters \
    -p tuning \
    -p pipeline_config \
    --force \
    uv run src/train.py

dvc stage add -n evaluate \
    -d data/processed/val.csv \
    -d src/evaluate.py \
    -d models/random_forest/random_forest.pkl \
    -o reports/random_forest/evaluation_report.json \
    -p paths \
    -p dataset \
    -p names \
    -p hyperparameters \
    -p tuning \
    -p pipeline_config \
    --force \
    uv run src/evaluate.py

