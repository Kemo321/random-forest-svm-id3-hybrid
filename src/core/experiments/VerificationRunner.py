# Authors: Jan Szwagierczak

import pandas as pd
from typing import Any, Callable, Dict, List, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from core.models.ID3Classifier import ID3Classifier
from core.models.HybridSVMForest import HybridSVMForest
import os
import numpy as np


Loader = Callable[[], Union[Tuple[Any, Any], Tuple[Any, Any, Any]]]
DatasetConfig = Dict[str, Any]


class VerificationRunner:
    def __init__(self, random_state: int = 42, results_dir: str = "./results", n_repeats: int = 25) -> None:
        self.random_state: int = random_state
        self.results_dir: str = results_dir
        self.n_repeats: int = n_repeats

    def run(self, datasets_config: List[DatasetConfig]) -> None:
        print("\n" + "=" * 90)
        print(f"{'VERIFICATION EXPERIMENT: Decision Trees & Random Forests':^90}")
        print(f"{'(Averaged over ' + str(self.n_repeats) + ' runs)':^90}")
        print("=" * 90)

        results: List[Dict[str, str]] = []

        for ds in datasets_config:
            ds_name: str = ds["name"]
            print(f"Verifying on {ds_name}...")
            try:
                loader: Loader = ds["loader"]
                loaded_data = loader()

                if len(loaded_data) == 3:
                    X_id3_orig, X_svm_orig, y_orig = loaded_data
                else:
                    X_id3_orig, y_orig = loaded_data
                    X_svm_orig = X_id3_orig

                scores_id3 = []
                scores_dt = []
                scores_rf = []
                scores_rf_sk = []
                scores_hybrid = []

                for i in range(self.n_repeats):
                    current_seed = self.random_state + i

                    X_id3_tr, X_id3_te, y_train, y_test = train_test_split(
                        X_id3_orig, y_orig, test_size=0.3, random_state=current_seed, stratify=y_orig
                    )
                    X_svm_tr, X_svm_te, _, _ = train_test_split(
                        X_svm_orig, y_orig, test_size=0.3, random_state=current_seed, stratify=y_orig
                    )

                    id3 = ID3Classifier()
                    id3.fit(X_id3_tr, y_train)
                    scores_id3.append(accuracy_score(y_test, id3.predict(X_id3_te)))

                    dt = DecisionTreeClassifier(criterion="entropy", random_state=current_seed)
                    dt.fit(X_id3_tr, y_train)
                    scores_dt.append(accuracy_score(y_test, dt.predict(X_id3_te)))

                    rf_my = HybridSVMForest(
                        estimator_count=50,
                        p_svm=0.0,
                        random_state=current_seed
                    )
                    rf_my.fit((X_id3_tr, X_svm_tr), y_train)
                    scores_rf.append(accuracy_score(y_test, rf_my.predict((X_id3_te, X_svm_te))))

                    rf_sk = RandomForestClassifier(n_estimators=50, random_state=current_seed)
                    rf_sk.fit(X_id3_tr, y_train)
                    scores_rf_sk.append(accuracy_score(y_test, rf_sk.predict(X_id3_te)))

                    hybrid = HybridSVMForest(
                        estimator_count=50,
                        p_svm=0.5,
                        C=10.0,
                        random_state=current_seed
                    )
                    hybrid.fit((X_id3_tr, X_svm_tr), y_train)
                    scores_hybrid.append(accuracy_score(y_test, hybrid.predict((X_id3_te, X_svm_te))))

                mean_id3 = np.mean(scores_id3)
                mean_dt = np.mean(scores_dt)
                mean_rf = np.mean(scores_rf)
                mean_rf_sk = np.mean(scores_rf_sk)
                mean_hybrid = np.mean(scores_hybrid)

                results.append({
                    "Dataset": ds_name,
                    "ID3": f"{mean_id3:.4f}",
                    "SkTree": f"{mean_dt:.4f}",
                    "RF": f"{mean_rf:.4f}",
                    "SkRF": f"{mean_rf_sk:.4f}",
                    "Hybrid": f"{mean_hybrid:.4f}",
                    "H-RF Diff": f"{mean_hybrid - mean_rf:.4f}"
                })

            except Exception as e:
                print(f"Error in verification for {ds_name}: {e}")
                import traceback
                traceback.print_exc()

        df_ver = pd.DataFrame(results)
        cols = ["Dataset", "ID3", "SkTree", "RF", "SkRF", "Hybrid", "H-RF Diff"]
        df_ver = df_ver[cols]

        print("\n" + df_ver.to_string(index=False))

        csv_path = os.path.join(self.results_dir, "verification_results.csv")
        df_ver.to_csv(csv_path, index=False)
        print(f"\nVerification results saved to {csv_path}\n")
