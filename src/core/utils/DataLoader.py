from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.datasets import load_breast_cancer
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
    def load_wine_quality_red_data(n_bins=5):
        try:
            dataset = openml.datasets.get_dataset(287, download_data=True)
            df, _, _, _ = dataset.get_data(dataset_format='dataframe')
            print("Wine Quality Red dataset downloaded from OpenML.")
        except Exception as e:
            raise RuntimeError(f"Failed to download Wine Quality Red dataset: {e}")

        if 'class' in df.columns:
            y_raw = df['class'].values
            X = df.drop('class', axis=1).values
        elif 'quality' in df.columns:
            y_raw = df['quality'].values
            X = df.drop('quality', axis=1).values
        else:
            y_raw = df.iloc[:, -1].values
            X = df.iloc[:, :-1].values

        y = (y_raw.astype(float) > 5).astype(int)

        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_id3 = est.fit_transform(X).astype(int)

        X_svm = X.astype(float)

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

    @staticmethod
    def load_car_data():
        try:
            dataset = openml.datasets.get_dataset(21, download_data=True)
            df, _, _, _ = dataset.get_data(dataset_format='dataframe')
            print("Car Evaluation dataset downloaded from OpenML.")
        except Exception as e:
            raise RuntimeError(f"Failed to download Car dataset: {e}")

        X_raw = df.drop('class', axis=1).values
        y_raw = df['class'].values

        X_id3 = OrdinalEncoder().fit_transform(X_raw).astype(int)
        X_svm = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(X_raw)
        y = LabelEncoder().fit_transform(y_raw)

        return X_id3, X_svm, y
