
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

model_files = {
    "Naive Baseline": "baseline_validation.csv",
    "XGBoost": "xgboost_validation.csv",
    "LSTM": "lstm_validation.csv",
    "SARIMAX": "sarimax_validation.csv",
}

MODEL_ORDER = [
    "Naive Baseline",
    "XGBoost",
    "LSTM",
    "SARIMAX"
]

# Consistent color for each model across all charts
MODEL_COLORS = {
    "Naive Baseline": "#7F8C8D",
    "XGBoost": "#3498DB",
    "LSTM": "#2ECC71",
    "SARIMAX": "#E67E22"
}


# ==================================================
# 2. LOAD AND COMBINE VALIDATION RESULTS
# ==================================================

all_results = []

for model_name, filename in model_files.items():

    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing results file: {file_path}"
        )

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    required_columns = [
        "Target",
        "MAE",
        "RMSE",
        "R2"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filename} is missing columns: {missing_columns}"
        )

    df = df[required_columns].copy()
    df["Model"] = model_name

    all_results.append(df)


comparison_df = pd.concat(
    all_results,
    ignore_index=True
)

comparison_df = comparison_df[
    ["Target", "Model", "MAE", "RMSE", "R2"]
]

# Convert metric columns to numeric values.
for metric in ["MAE", "RMSE", "R2"]:
    comparison_df[metric] = pd.to_numeric(
        comparison_df[metric],
        errors="coerce"
    )

comparison_df["Model"] = pd.Categorical(
    comparison_df["Model"],
    categories=MODEL_ORDER,
    ordered=True
)

comparison_df = comparison_df.sort_values(
    ["Target", "Model"]
).reset_index(drop=True)


# ==================================================
# 3. SAVE CONSOLIDATED COMPARISON
# ==================================================

comparison_path = (
    RESULTS_DIR / "model_comparison_validation.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\n" + "=" * 90)
print("COMPLETE MODEL VALIDATION COMPARISON")
print("=" * 90)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print(f"\nComparison saved: {comparison_path}")


# ==================================================
# 4. BEST MODEL BY MAE AND RMSE
# ==================================================

print("\n" + "=" * 90)
print("BEST MODEL PER TARGET")
print("=" * 90)

for target in comparison_df["Target"].unique():

    target_results = comparison_df[
        comparison_df["Target"] == target
    ]

    valid_mae = target_results.dropna(subset=["MAE"])
    valid_rmse = target_results.dropna(subset=["RMSE"])

    print(f"\nTarget: {target}")

    if not valid_mae.empty:

        best_mae = valid_mae.loc[
            valid_mae["MAE"].idxmin()
        ]

        print(
            f"  Best MAE : {best_mae['Model']} "
            f"({best_mae['MAE']:.4f})"
        )

    if not valid_rmse.empty:

        best_rmse = valid_rmse.loc[
            valid_rmse["RMSE"].idxmin()
        ]

        print(
            f"  Best RMSE: {best_rmse['Model']} "
            f"({best_rmse['RMSE']:.4f})"
        )


# ==================================================
# 5. SHARED PLOTTING FUNCTION
# MAE, RMSE, AND R2 COMPARISONS
# ==================================================

def plot_metric_comparison(metric, filename, title):

    plot_df = comparison_df.dropna(
        subset=[metric]
    )

    targets = plot_df["Target"].unique()

    n_targets = len(targets)
    n_cols = 2
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4.5 * n_rows)
    )

    axes = axes.flatten()

    for index, target in enumerate(targets):

        ax = axes[index]

        target_df = plot_df[
            plot_df["Target"] == target
        ].sort_values("Model")

        colors = [
            MODEL_COLORS[str(model)]
            for model in target_df["Model"]
        ]

        bars = ax.bar(
            target_df["Model"].astype(str),
            target_df[metric],
            color=colors,
            edgecolor="black",
            linewidth=0.5
        )

        ax.set_title(
            target.replace("_", " ").title(),
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel(metric)

        ax.tick_params(
            axis="x",
            rotation=20
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        # Display metric values above bars.
        for bar in bars:

            height = bar.get_height()

            ax.annotate(
                f"{height:.3f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

        # R2 can be negative.
        if metric == "R2":

            ax.axhline(
                0,
                linestyle="--",
                linewidth=1,
                color="black"
            )

    # Hide unused subplot panels.
    for index in range(n_targets, len(axes)):
        axes[index].set_visible(False)

    # Shared legend.
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=MODEL_COLORS[model],
            label=model
        )
        for model in MODEL_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False
    )

    fig.suptitle(
        title,
        fontsize=17,
        fontweight="bold",
        y=1.04
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_file = FIGURES_DIR / filename

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved visualization: {output_file}")


# ==================================================
# 6. GENERATE METRIC COMPARISON CHARTS
# ==================================================

plot_metric_comparison(
    metric="MAE",
    filename="mae_comparison.png",
    title="Model Comparison Across Forecasting Targets - MAE"
)

plot_metric_comparison(
    metric="RMSE",
    filename="rmse_comparison.png",
    title="Model Comparison Across Forecasting Targets - RMSE"
)

plot_metric_comparison(
    metric="R2",
    filename="r2_comparison.png",
    title="Model Comparison Across Forecasting Targets - R²"
)


# ==================================================
# 7. BEST MODEL PER TARGET
# PRIMARY CRITERION: LOWEST VALIDATION RMSE
# ==================================================

# Select the model with the lowest RMSE per target.
valid_rmse_df = comparison_df.dropna(
    subset=["RMSE"]
)

best_indices = (
    valid_rmse_df.groupby("Target")["RMSE"].idxmin()
)

best_models = valid_rmse_df.loc[
    best_indices
].copy()

best_models = best_models.sort_values(
    "RMSE",
    ascending=True
).reset_index(drop=True)


# Create horizontal winner chart.
fig, ax = plt.subplots(
    figsize=(13, 7)
)

winner_colors = [
    MODEL_COLORS[str(model)]
    for model in best_models["Model"]
]

bars = ax.barh(
    best_models["Target"].str.replace(
        "_", " "
    ).str.title(),
    best_models["RMSE"],
    color=winner_colors,
    edgecolor="black",
    linewidth=0.5
)

ax.set_title(
    "Best Performing Model for Each Forecasting Target",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Validation RMSE (lower is better)"
)

ax.set_ylabel(
    "Forecasting Target"
)

ax.invert_yaxis()

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

# Annotate bars with model name and RMSE.
for bar, (_, row) in zip(
    bars,
    best_models.iterrows()
):

    ax.annotate(
        f"{row['Model']}  |  RMSE: {row['RMSE']:.3f}",
        xy=(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2
        ),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9
    )


# Legend for winner chart.
legend_handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=MODEL_COLORS[model],
        label=model
    )
    for model in MODEL_ORDER
]

ax.legend(
    handles=legend_handles,
    title="Model",
    loc="lower right"
)

plt.tight_layout()

winner_chart_path = (
    FIGURES_DIR / "best_model_per_target.png"
)

plt.savefig(
    winner_chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 8. SAVE BEST MODELS TABLE
# ==================================================

best_models_path = (
    RESULTS_DIR / "best_models_per_target.csv"
)

best_models.to_csv(
    best_models_path,
    index=False
)


print("\n" + "=" * 90)
print("FINAL BEST MODEL SELECTION")
print("=" * 90)

print(
    best_models[
        ["Target", "Model", "MAE", "RMSE", "R2"]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print(f"\nWinner chart saved: {winner_chart_path}")
print(f"Best models CSV saved: {best_models_path}")


# ==================================================
# 9. COMPLETION SUMMARY
# ==================================================

print("\n" + "=" * 90)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 90)

print("\nGenerated files:")

print(f"1. {comparison_path}")
print(f"2. {best_models_path}")
print(f"3. {FIGURES_DIR / 'mae_comparison.png'}")
print(f"4. {FIGURES_DIR / 'rmse_comparison.png'}")
print(f"5. {FIGURES_DIR / 'r2_comparison.png'}")
print(f"6. {winner_chart_path}")

print("=" * 90)
