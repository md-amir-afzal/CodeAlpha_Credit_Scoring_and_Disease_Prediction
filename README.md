# CodeAlpha Machine Learning Internship Projects

This repository contains two completed tasks from the CodeAlpha Machine Learning internship task list.

## Completed Tasks
1. **Credit Scoring Model** — classification of creditworthiness using financial-history features.
2. **Disease Prediction from Medical Data** — classification using structured breast-cancer medical measurements.

The supplied CodeAlpha instructions require completion of a minimum of two or three tasks for internship completion and require source code to be uploaded to GitHub in a repository named `CodeAlpha_ProjectName`.

## Project Structure
- `task1_credit_scoring/`
- `task4_disease_prediction/`
- `requirements.txt`

## Quick Start
```bash
pip install -r requirements.txt
cd task1_credit_scoring
python train_model.py
python predict.py

cd ../task4_disease_prediction
python train_model.py
python predict.py
```

## Notes
All metrics are generated from reproducible train/test splits (`random_state=42`). The credit dataset is synthetic and is included for educational demonstration. The disease-prediction task is strictly an ML demonstration and not a medical diagnostic tool.
