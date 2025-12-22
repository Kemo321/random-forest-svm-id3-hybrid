import pandas as pd
from typing import Any, Callable, Dict, List, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from core.models.ID3Classifier import ID3Classifier
from core.models.HybridSVMForest import HybridSVMForest
import os


Loader = Callable[[], Union[Tuple[Any, Any], Tuple[Any, Any, Any]]]
DatasetConfig = Dict[str, Any]


class VerificationRunner:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state: int = random_state

    def run(self, datasets_config: List[DatasetConfig]) -> None:
        print("\n" + "=" * 90)
        print(f"{'VERIFICATION EXPERIMENT: Decision Trees & Random Forests':^90}")
        print("=" * 90)

        results: List[Dict[str, str]] = []

        for ds in datasets_config:
            ds_name: str = ds["name"]
            print(f"Verifying on {ds_name}...")
            try:
                loader: Loader = ds["loader"]
                loaded_data = loader()

                if len(loaded_data) == 3:
                    X_id3, X_svm, y = loaded_data
                else:
                    X_id3, y = loaded_data
                    X_svm = X_id3

                X_id3_tr, X_id3_te, y_train, y_test = train_test_split(
                    X_id3, y, test_size=0.3, random_state=self.random_state, stratify=y
                )
                X_svm_tr, X_svm_te, _, _ = train_test_split(
                    X_svm, y, test_size=0.3, random_state=self.random_state, stratify=y
                )

                id3 = ID3Classifier()
                id3.fit(X_id3_tr, y_train)
                acc_id3 = accuracy_score(y_test, id3.predict(X_id3_te))

                dt = DecisionTreeClassifier(criterion="entropy", random_state=self.random_state)
                dt.fit(X_id3_tr, y_train)
                acc_dt = accuracy_score(y_test, dt.predict(X_id3_te))

                hybrid = HybridSVMForest(
                    estimator_count=50,
                    p_svm=0.5,
                    random_state=self.random_state
                )
                hybrid.fit((X_id3_tr, X_svm_tr), y_train)
                acc_hybrid = accuracy_score(y_test, hybrid.predict((X_id3_te, X_svm_te)))

                rf_sk = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                rf_sk.fit(X_id3_tr, y_train)
                acc_rf = accuracy_score(y_test, rf_sk.predict(X_id3_te))

                results.append({
                    "Dataset": ds_name,
                    "ID3": f"{acc_id3:.4f}",
                    "SkTree": f"{acc_dt:.4f}",
                    "Hybrid": f"{acc_hybrid:.4f}",
                    "SkRF": f"{acc_rf:.4f}",
                    "H-RF Diff": f"{acc_hybrid - acc_rf:.4f}"
                })

            except Exception as e:
                print(f"Error in verification for {ds_name}: {e}")
                import traceback
                traceback.print_exc()

        df_ver = pd.DataFrame(results)
        print("\n" + df_ver.to_string(index=False))

        csv_path = os.path.join('results', "verification_results.csv")
        df_ver.to_csv(csv_path, index=False)
        print(f"\nVerification results saved to {csv_path}\n")
