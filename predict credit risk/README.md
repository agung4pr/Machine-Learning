<<<<<<< HEAD
# Machine-Learning
=======
# Predict Credit Risk

A portfolio project for training and serving a machine learning model that predicts loan default risk.

## What is in this repo

- `predict_credit_risk/`: training, inference, and FastAPI application code
- `tests/`: focused tests for the data-cleaning logic
- `notebooks/`: experimentation notebook
- `scripts/`: small utility scripts for quick dataset inspection

## Project structure

```text
predict-credit-risk/
  .github/
    workflows/
      ci.yml
      cd.yml
  predict_credit_risk/
    __init__.py
    api.py
    inference.py
    training.py
  notebooks/
    credit_risk_modeling_walkthrough.ipynb
  scripts/
    inspect_dataset.py
  tests/
    test_data_cleaning.py
  dataset/                # local only, ignored from git
  models/                 # local only, ignored from git
  Dockerfile
  .gitignore
  README.md
  requirements.txt
  requirements-dev.txt
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m predict_credit_risk.training
```

Start the API:

```bash
uvicorn predict_credit_risk.api:app --reload
```

Score a CSV file:

```bash
python -m predict_credit_risk.inference --input dataset/credit_risk_dataset.csv
```

Run tests:

```bash
pytest
```

Lint the code:

```bash
ruff check .
```

Build the production container:

```bash
docker build -t predict-credit-risk .
```

Run the API container with a mounted model artifact:

```bash
docker run --rm -p 8000:8000 ^
  -v %cd%\\models:/app/models ^
  -e CREDIT_RISK_MODEL_PATH=/app/models/credit_risk_pipeline.joblib ^
  predict-credit-risk
```

## Notebook

The main project notebook is `notebooks/credit_risk_modeling_walkthrough.ipynb`. It explains each cell and follows the same workflow as the package code: data loading, cleaning, threshold tuning, overfitting checks, and feature-importance review.

## CI/CD

The repository includes two GitHub Actions workflows:

- `CI`: runs `ruff`, `pytest`, and validates the Docker image on pull requests and pushes to `main`
- `CD`: publishes a production image to GitHub Container Registry (`ghcr.io`) after the CI workflow succeeds on `main`

The deployed container expects the trained model artifact to be available at `CREDIT_RISK_MODEL_PATH`. By default, the image looks for `/app/models/credit_risk_pipeline.joblib`, which is why the example `docker run` command mounts the local `models/` directory into the container.
>>>>>>> 27f938b (fixing model sensitivity and managing api services)
