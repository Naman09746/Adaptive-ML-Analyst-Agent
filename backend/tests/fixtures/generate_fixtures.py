# ama2/backend/tests/fixtures/generate_fixtures.py

import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris, make_classification

def main():
    fixture_dir = Path(__file__).parent
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean Iris Classification
    iris = load_iris()
    df_iris = pd.DataFrame(data=iris.data, columns=[c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names])
    df_iris["species"] = iris.target
    iris_path = fixture_dir / "iris.csv"
    df_iris.to_csv(iris_path, index=False)
    print(f"Generated clean classification dataset at: {iris_path}")

    # 2. Messy Dataset (with >20% nulls in some columns to test imputation)
    X, y = make_classification(n_samples=150, n_features=5, n_classes=2, random_state=42)
    df_messy = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    
    # Introduce nulls in feat_0 (~10%) and feat_1 (~15%)
    df_messy.loc[df_messy.sample(frac=0.1, random_state=42).index, "feat_0"] = None
    df_messy.loc[df_messy.sample(frac=0.15, random_state=42).index, "feat_1"] = None
    df_messy["target"] = y
    
    messy_path = fixture_dir / "messy.csv"
    df_messy.to_csv(messy_path, index=False)
    print(f"Generated messy classification dataset at: {messy_path}")

if __name__ == "__main__":
    main()
