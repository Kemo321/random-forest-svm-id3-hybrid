import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.datasets import load_breast_cancer, load_wine
import openml


class DataLoader:
    @staticmethod
    def load_breast_cancer_data(n_bins=5):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_id3 = est.fit_transform(X).astype(int)

        X_svm = X

        return X_id3, X_svm, y

    @staticmethod
    def load_wine_data(n_bins=5):
        data = load_wine()
        X = data.data
        y = data.target

        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_id3 = est.fit_transform(X).astype(int)

        X_svm = X

        return X_id3, X_svm, y

    @staticmethod
    def load_mushroom_data(filepath='data/mushrooms.csv'):
        try:
            dataset = openml.datasets.get_dataset(24, download_data=True)
            df = dataset.get_data(dataset_format='dataframe')[0]
            print("Mushroom dataset downloaded from OpenML.")
        except Exception as e:
            raise RuntimeError(f"Failed to download Mushroom dataset: {e}")

        df = df.astype('object')
        df = df.replace('?', 'missing')
        df = df.fillna('missing')

        if 'class' in df.columns:
            y_raw = df['class'].values
            X_raw = df.drop('class', axis=1).values
        else:
            y_raw = df.iloc[:, 0].values
            X_raw = df.iloc[:, 1:].values

        enc_ord = OrdinalEncoder()
        X_id3 = enc_ord.fit_transform(X_raw).astype(int)

        enc_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_svm = enc_ohe.fit_transform(X_raw)

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        return X_id3, X_svm, y
