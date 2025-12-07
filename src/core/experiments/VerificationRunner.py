import pandas as pd
from typing import Any, Callable, Dict, List, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from core.models.ID3Classifier import ID3Classifier

Loader = Callable[[], Union[Tuple[Any, Any], Tuple[Any, Any, Any]]]
DatasetConfig = Dict[str, Any]


class VerificationRunner:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state: int = random_state

    def run(self, datasets_config: List[DatasetConfig]) -> None:
        print("\n" + "=" * 50)
        print("VERIFICATION EXPERIMENT: Custom ID3 vs Sklearn Tree")
        print("=" * 50)

        results: List[Dict[str, str]] = []

        for ds in datasets_config:
            ds_name: str = ds["name"]
            print(f"Verifying on {ds_name}...")
            try:
                loader: Loader = ds["loader"]
                loaded_data = loader()
                if len(loaded_data) == 3:
                    X_id3, _, y = loaded_data
                else:
                    X_id3, y = loaded_data

                X_train, X_test, y_train, y_test = train_test_split(
                    X_id3, y, test_size=0.3, random_state=self.random_state, stratify=y
                )

                id3: ID3Classifier = ID3Classifier()
                id3.fit(X_train, y_train)
                pred_id3 = id3.predict(X_test)
                acc_id3 = accuracy_score(y_test, pred_id3)

                dt: DecisionTreeClassifier = DecisionTreeClassifier(
                    criterion="entropy", random_state=self.random_state
                )
                dt.fit(X_train, y_train)
                pred_dt = dt.predict(X_test)
                acc_dt = accuracy_score(y_test, pred_dt)

                results.append(
                    {
                        "Dataset": ds_name,
                        "Custom ID3 Acc": f"{acc_id3:.4f}",
                        "Sklearn Tree Acc": f"{acc_dt:.4f}",
                        "Diff": f"{acc_id3 - acc_dt:.4f}",
                    }
                )

            except Exception as e:
                print(f"Error in verification for {ds_name}: {e}")

        df_ver = pd.DataFrame(results)
        print(df_ver.to_string(index=False))
