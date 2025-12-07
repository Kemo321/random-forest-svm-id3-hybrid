import os

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    output_dir: str

    def __init__(self, output_dir: str = "./plots") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_experiment(self, df_results: pd.DataFrame, x_col: str, x_label: str, title_suffix: str) -> None:
        if df_results.empty:
            return

        datasets = df_results["dataset"].unique()

        for ds_name in datasets:
            subset = df_results[df_results["dataset"] == ds_name]
            subset = subset.sort_values(x_col)

            plt.figure(figsize=(10, 6))

            hybrid_data = subset[subset["model_type"] == "HybridSVMForest"]
            if not hybrid_data.empty:
                plt.errorbar(
                    hybrid_data[x_col],
                    hybrid_data["mean_acc"],
                    yerr=hybrid_data["std_acc"],
                    fmt='-o',
                    label='Hybrid SVM Forest',
                    capsize=5,
                    linewidth=2
                )

            plt.title(f"{title_suffix} - {ds_name}")
            plt.xlabel(x_label)
            plt.ylabel("Mean Accuracy (CV)")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()

            safe_name: str = f"{ds_name}_{x_col}".replace(" ", "_").replace("-", "").lower()
            filename: str = os.path.join(self.output_dir, f"exp_{safe_name}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Plot saved: {filename}")
