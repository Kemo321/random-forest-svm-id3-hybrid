from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.datasets import load_breast_cancer
import openml
import numpy as np


class DataLoader:
    @staticmethod
    def load_breast_cancer_data(n_bins=5):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        print(f"DEBUG - Dataset name: Wisconsin Breast Cancer")
        print(f"DEBUG - Dataset description: {data.DESCR[:300] if data.DESCR else 'No description'}")
        print(f"DEBUG - Number of instances: {X.shape[0]}")
        print(f"DEBUG - Number of features: {X.shape[1]}")
        print(f"DEBUG - Columns: {list(data.feature_names)}")
        unique, counts = np.unique(y, return_counts=True)
        class_names = dict(zip(data.target_names, range(len(data.target_names))))
        print(f"DEBUG - Class distribution: {', '.join([f'{data.target_names[k]} ({k}): {v}' for k, v in zip(unique, counts)])}")

        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_id3 = est.fit_transform(X).astype(int)

        X_svm = X

        return X_id3, X_svm, y

    @staticmethod
    def load_wine_quality_red_data(n_bins=5):
        try:
            dataset = openml.datasets.get_dataset(40691, download_data=True)
            df, _, _, _ = dataset.get_data(dataset_format='dataframe')
            print("Wine Quality Red dataset downloaded from OpenML.")
            print(f"DEBUG - Dataset name: {dataset.name}")
            print(f"DEBUG - Dataset description: {dataset.description[:300] if dataset.description else 'No description'}")
            print(f"DEBUG - Number of instances: {df.shape[0]}")
            print(f"DEBUG - Number of features: {df.shape[1]}")
            print(f"DEBUG - Columns: {list(df.columns)}")
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
        print(f"DEBUG - Class distribution: Low quality (0): {(y == 0).sum()}, Good quality (1): {(y == 1).sum()}")

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
            print(f"DEBUG - Dataset name: {dataset.name}")
            print(f"DEBUG - Dataset description: {dataset.description[:300] if dataset.description else 'No description'}")
            print(f"DEBUG - Number of instances: {df.shape[0]}")
            print(f"DEBUG - Number of features: {df.shape[1]}")
            print(f"DEBUG - Columns: {list(df.columns)}")
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

        unique, counts = np.unique(y, return_counts=True)
        class_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        print(f"DEBUG - Class distribution: {', '.join([f'{class_mapping[k]} ({k}): {v}' for k, v in zip(unique, counts)])}")

        return X_id3, X_svm, y

    @staticmethod
    def load_car_data():
        try:
            dataset = openml.datasets.get_dataset(21, download_data=True)
            df, _, _, _ = dataset.get_data(dataset_format='dataframe')
            print("Car Evaluation dataset downloaded from OpenML.")
            print(f"DEBUG - Dataset name: {dataset.name}")
            print(f"DEBUG - Dataset description: {dataset.description[:300] if dataset.description else 'No description'}")
            print(f"DEBUG - Number of instances: {df.shape[0]}")
            print(f"DEBUG - Number of features: {df.shape[1]}")
            print(f"DEBUG - Columns: {list(df.columns)}")
        except Exception as e:
            raise RuntimeError(f"Failed to download Car dataset: {e}")

        X_raw = df.drop('class', axis=1).values
        y_raw = df['class'].values

        X_id3 = OrdinalEncoder().fit_transform(X_raw).astype(int)
        X_svm = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(X_raw)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        unique, counts = np.unique(y, return_counts=True)
        class_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        print(f"DEBUG - Class distribution: {', '.join([f'{class_mapping[k]} ({k}): {v}' for k, v in zip(unique, counts)])}")

        return X_id3, X_svm, y


if __name__ == "__main__":
    # wine quality red
    X_id3, X_svm, y = DataLoader.load_mushroom_data()
    X_id3, X_svm, y = DataLoader.load_breast_cancer_data()
    X_id3, X_svm, y = DataLoader.load_wine_quality_red_data()
    X_id3, X_svm, y = DataLoader.load_car_data()