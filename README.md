# Sydney Housing Price Prediction and Decision Support Project

This workspace contains a complete starter implementation for the Sydney housing price prediction assignment based on the provided dataset.

## Project structure

- `data/raw/` — raw property dataset
- `data/processed/` — cleaned dataset and generated outputs
- `notebooks/` — analysis notebook
- `src/` — preprocessing and model training scripts
- `app/` — Streamlit web application
- `report/` — model evaluation and error analysis outputs

## Included files

- `data/raw/8D_Housing_Data_Template (1).csv` — base dataset
- `src/housing_model.py` — data preprocessing, feature engineering, cross-validation, and model evaluation
- `app/sydney_housing_app.py` — prediction web app
- `notebooks/sydney_housing_analysis.ipynb` — notebook template for exploration
- `report/model_cv_results.csv` — CV summary by model
- `report/largest_prediction_errors.csv` — top 5 prediction failures

## Local setup

1. Open a terminal in the project folder.
2. Install dependencies:
   `py -m pip install -r requirements.txt`
3. Run the model script:
   `py src\housing_model.py`
4. Launch the app:
   `streamlit run app\sydney_housing_app.py`

## Notes

- The dataset contains three materially different Sydney markets: Campsie, Manly, and Parramatta.
- The project reflects a realistic real-estate ML workflow with missing values, feature engineering, model comparison, and decision support.
- The final app should be treated as a valuation aid rather than a substitute for professional appraisal.
