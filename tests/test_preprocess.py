import pandas as pd

from src.preprocess import FEATURE_COLUMNS, TARGET_COLUMN, build_model_pipeline, prepare_features


def sample_dataframe():
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3, 4, 5, 6],
            "Product ID": ["M14860", "L47181", "L47182", "M14863", "H29424", "L47185"],
            "Type": ["M", "L", "L", "M", "H", "L"],
            "Air temperature [K]": [298.1, 298.2, 298.1, 298.2, 298.3, 298.4],
            "Process temperature [K]": [308.6, 308.7, 308.5, 308.6, 308.7, 308.9],
            "Rotational speed [rpm]": [1551, 1408, 1498, 1433, 1550, 1600],
            "Torque [Nm]": [42.8, 46.3, 49.4, 39.5, 35.0, 31.0],
            "Tool wear [min]": [0, 3, 5, 7, 9, 11],
            "Machine failure": [0, 0, 0, 1, 0, 1],
            "TWF": [0, 0, 0, 1, 0, 0],
            "HDF": [0, 0, 0, 0, 0, 1],
            "PWF": [0, 0, 0, 0, 0, 0],
            "OSF": [0, 0, 0, 0, 0, 0],
            "RNF": [0, 0, 0, 0, 0, 0],
        }
    )


def test_prepare_features_uses_expected_columns_only():
    X, y = prepare_features(sample_dataframe())

    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN
    assert "TWF" not in X.columns
    assert "Product ID" not in X.columns


def test_model_pipeline_can_fit_sample_data():
    X, y = prepare_features(sample_dataframe())
    pipeline = build_model_pipeline(random_state=42)
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)
