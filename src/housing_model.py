from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(r"C:\Users\Hp_D\.copilot\session-state\61ab8fea-708f-4592-89ca-ebda704d8ef5\files\project\sydney-housing\data\raw\8D_Housing_Data_Template (1).csv")
OUTPUT_DIR = Path(r"C:\Users\Hp_D\.copilot\session-state\61ab8fea-708f-4592-89ca-ebda704d8ef5\files\project\sydney-housing\data\processed")
REPORT_DIR = Path(r"C:\Users\Hp_D\.copilot\session-state\61ab8fea-708f-4592-89ca-ebda704d8ef5\files\project\sydney-housing\report")


def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["property_id", "suburb", "address"]).copy()
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df = df.dropna(subset=["sale_price"]).copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_month"] = df["sale_date"].dt.month

    numeric_cols = [
        "bedrooms",
        "bathrooms",
        "parking_spaces",
        "land_size_sqm",
        "floor_area_sqm",
        "distance_to_cbd_km",
        "distance_to_station_km",
        "year_built",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year_built"] = df["year_built"].fillna(df["year_built"].median())
    df["property_type"] = df["property_type"].fillna("Unknown")
    df["suburb"] = df["suburb"].fillna("Unknown")
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
    df["parking_spaces"] = df["parking_spaces"].fillna(df["parking_spaces"].median())
    df["land_size_sqm"] = df["land_size_sqm"].fillna(df["land_size_sqm"].median())
    df["floor_area_sqm"] = df["floor_area_sqm"].fillna(df["floor_area_sqm"].median())
    df["distance_to_cbd_km"] = df["distance_to_cbd_km"].fillna(df["distance_to_cbd_km"].median())
    df["distance_to_station_km"] = df["distance_to_station_km"].fillna(df["distance_to_station_km"].median())

    df["price_per_sqm"] = df["sale_price"] / df["floor_area_sqm"].replace(0, np.nan)
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
    df["parking_to_bed_ratio"] = df["parking_spaces"] / df["bedrooms"].replace(0, np.nan)
    df["sale_year"] = df["sale_date"].dt.year

    return df


def make_features(df: pd.DataFrame):
    feature_cols = [
        "suburb",
        "property_type",
        "bedrooms",
        "bathrooms",
        "parking_spaces",
        "land_size_sqm",
        "floor_area_sqm",
        "sale_month",
        "price_per_sqm",
        "bed_bath_ratio",
        "parking_to_bed_ratio",
    ]
    X = df[feature_cols].copy()
    y = df["sale_price"].astype(float)
    return X, y


def build_model_pipeline(model):
    numeric_cols = [
        "bedrooms",
        "bathrooms",
        "parking_spaces",
        "land_size_sqm",
        "floor_area_sqm",
        "sale_month",
        "price_per_sqm",
        "bed_bath_ratio",
        "parking_to_bed_ratio",
    ]
    categorical_cols = ["suburb", "property_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_models():
    df = load_and_prepare_data()
    X, y = make_features(df)
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=500, min_samples_leaf=1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=3),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, model in models.items():
        pipe = build_model_pipeline(model)
        rmse_scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        mae_scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
        r2_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        results.append(
            {
                "model": name,
                "rmse_mean": rmse_scores.mean(),
                "rmse_std": rmse_scores.std(),
                "mae_mean": -mae_scores.mean(),
                "mae_std": mae_scores.std(),
                "r2_mean": r2_scores.mean(),
                "r2_std": r2_scores.std(),
            }
        )

    results_df = pd.DataFrame(results).sort_values("rmse_mean")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_DIR / "model_cv_results.csv", index=False)
    return df, X, y, results_df


def train_best_model_and_error_analysis():
    df, X, y, results = evaluate_models()
    best_model_name = results.iloc[0]["model"]
    model_map = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=500, min_samples_leaf=1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=3),
    }
    best_model = build_model_pipeline(model_map[best_model_name])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    test_df = X_test.copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = preds
    test_df["absolute_error"] = np.abs(test_df["actual"] - test_df["predicted"])
    test_df["relative_error_pct"] = (test_df["absolute_error"] / test_df["actual"]) * 100
    largest_errors = test_df.sort_values("absolute_error", ascending=False).head(5).copy()

    df_test = df.loc[X_test.index, ["property_id", "suburb", "address", "property_type", "sale_price", "sale_date", "bedrooms", "bathrooms", "parking_spaces", "land_size_sqm", "floor_area_sqm", "distance_to_cbd_km", "distance_to_station_km", "year_built"]].copy()
    df_test["actual"] = y_test.values
    df_test["predicted"] = preds
    df_test["absolute_error"] = np.abs(df_test["actual"] - df_test["predicted"])
    df_test["relative_error_pct"] = (df_test["absolute_error"] / df_test["actual"]) * 100
    df_test = df_test.sort_values("absolute_error", ascending=False).head(5)
    df_test.to_csv(REPORT_DIR / "largest_prediction_errors.csv", index=False)

    return best_model_name, y_test, preds, df_test


if __name__ == "__main__":
    df, X, y, results = evaluate_models()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "cleaned_housing_data.csv", index=False)
    best_model_name, y_test, preds, errors = train_best_model_and_error_analysis()
    print(results.round(4).to_string(index=False))
    print(f"\nBest model by cross-validated RMSE: {best_model_name}")
    print(f"Test MAE: {mean_absolute_error(y_test, preds):,.0f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, preds)):,.0f}")
    print(f"Test R²: {r2_score(y_test, preds):.3f}")
    print("\nTop 5 prediction error cases saved to report/largest_prediction_errors.csv")
