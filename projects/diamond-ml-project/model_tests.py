import pickle
import pandas as pd
import numpy as np


def main():
    with open("diamond-model-complete.pkl", "rb") as f:
        saved_data = pickle.load(f)

    model = saved_data["model"]

    X_test = pd.read_csv("diamond-testdata-scaled.csv")

    y_pred = model.predict(X_test)

    print(y_pred)


if __name__ == "__main__":
    main()
