import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder
from sklearn.datasets import load_breast_cancer, load_wine

try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False


class DataLoader:
    """
    Utility class for loading and preprocessing datasets.
    """
    @staticmethod
    def load_breast_cancer_data(n_bins=5):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_binned = est.fit_transform(X)

        return X_binned.astype(int), y

    @staticmethod
    def load_wine_data(n_bins=5):
        data = load_wine()
        X = data.data
        y = data.target

        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_binned = est.fit_transform(X)

        return X_binned.astype(int), y

    @staticmethod
    def load_mushroom_data(filepath='data/mushrooms.csv'):
        df = None

        if OPENML_AVAILABLE:
            try:
                dataset = openml.datasets.get_dataset(24, download_data=True)
                df = dataset.get_data(dataset_format='dataframe')[0]
                print("Mushroom dataset downloaded from OpenML.")
            except Exception as e:
                print(f"Error downloading Mushroom dataset from OpenML: {e}")

        if df is None:
            try:
                df = pd.read_csv(filepath)
            except FileNotFoundError:
                raise Exception("File mushroom.csv not found locally and OpenML is unavailable.")

        if 'class' in df.columns:
            y_raw = df['class'].values
            X_raw = df.drop('class', axis=1).values
        else:
            y_raw = df.iloc[:, 0].values
            X_raw = df.iloc[:, 1:].values

        enc = OrdinalEncoder()
        X_encoded = enc.fit_transform(X_raw)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)

        return X_encoded.astype(int), y_encoded
