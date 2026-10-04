import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ==========================================================
# Load Dataset
# ==========================================================
df = pd.read_csv("telco.csv")

sns.set_theme(style="whitegrid")

print(f"Dataset Shape: {df.shape}")
print("=" * 60)


# ==========================================================
# Missing Values and Structural Missingness
# ==========================================================
print("\n========== Missing Values Summary ==========")

missing_df = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing %": (
        df.isnull().sum() / len(df) * 100
    ).round(2)
})

missing_df = (
    missing_df[missing_df["Missing Values"] > 0]
    .sort_values(by="Missing %", ascending=False)
)

print(missing_df)


# ==========================================================
# Class Imbalance
# ==========================================================
print("\n========== Class Imbalance ==========")

class_summary = pd.DataFrame({
    "Count": df["Churn Label"].value_counts(),
    "Percentage (%)": (
        df["Churn Label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
})

print(class_summary)

plt.figure(figsize=(6, 4))

ax = sns.countplot(
    data=df,
    x="Churn Label",
    hue="Churn Label",
    legend=False
)

plt.title("Churn Label Distribution")
plt.xlabel("Churn Label")
plt.ylabel("Count")

total = len(df)

for bar in ax.patches:
    percentage = (
        100 * bar.get_height() / total
    )

    ax.annotate(
        f"{percentage:.2f}%",
        (
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()
        ),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ==========================================================
# Correlation and Multicollinearity
# ==========================================================
print(
    "\n========== Correlation and "
    "Multicollinearity =========="
)

selected_num_cols = [
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Total Revenue"
]

numeric_data = df[selected_num_cols].dropna()

correlation_matrix = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))


# Correlation Heatmap
plt.figure(figsize=(6, 5))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title(
    "Correlation Heatmap of Selected Numerical Features"
)

plt.tight_layout()
plt.show()


# VIF Analysis
# Intercept is included for standard VIF calculation.
vif_input = add_constant(numeric_data)

vif_df = pd.DataFrame({
    "Feature": vif_input.columns,
    "VIF Score": [
        variance_inflation_factor(
            vif_input.values,
            i
        )
        for i in range(vif_input.shape[1])
    ]
})

vif_df = (
    vif_df[vif_df["Feature"] != "const"]
    .reset_index(drop=True)
)

vif_df["VIF Score"] = (
    vif_df["VIF Score"].round(2)
)

print("\nVIF Multicollinearity Analysis:")
print(vif_df)


# ==========================================================
# Skewness and Potential Outliers
# ==========================================================
print("\n========== Skewness Summary ==========")

skew_cols = [
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Avg Monthly GB Download",
    "Total Revenue"
]

skew_df = pd.DataFrame({
    "Skewness": (
        df[skew_cols]
        .skew()
        .round(2)
    )
}).sort_values(
    by="Skewness",
    ascending=False
)

print(skew_df)


# Histogram + Boxplot
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4)
)

sns.histplot(
    df["Total Revenue"].dropna(),
    bins=30,
    kde=True,
    ax=axes[0]
)

axes[0].set_title(
    "Histogram of Total Revenue"
)

sns.boxplot(
    x=df["Total Revenue"].dropna(),
    ax=axes[1]
)

axes[1].set_title(
    "Boxplot of Total Revenue"
)

plt.suptitle(
    "Distribution and Outliers of Total Revenue"
)

plt.tight_layout()
plt.show()


# ==========================================================
# Duplicate Records
# ==========================================================
print("\n========== Duplicate Records ==========")

duplicate_count = df.duplicated().sum()

print(
    f"Total duplicate rows found: "
    f"{duplicate_count}"
)


# ==========================================================
# Basic Data Consistency Checks
# ==========================================================
print(
    "\n========== Data Consistency Checks =========="
)

categorical_cols = (
    df.select_dtypes(include=["object"])
    .columns
)

whitespace_issues = {}

for col in categorical_cols:

    values = (
        df[col]
        .dropna()
        .astype(str)
    )

    issue_count = (
        values != values.str.strip()
    ).sum()

    if issue_count > 0:
        whitespace_issues[col] = issue_count


if whitespace_issues:
    print(
        "Leading/trailing whitespace issues:"
    )
    print(whitespace_issues)
else:
    print(
        "No leading/trailing whitespace "
        "inconsistencies detected."
    )


# Variables that should logically be non-negative
non_negative_cols = [
    "Age",
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue"
]

print("\nNegative-value checks:")

for col in non_negative_cols:

    if col in df.columns:

        negative_count = (
            df[col]
            .dropna()
            .lt(0)
            .sum()
        )

        print(
            f"{col}: "
            f"{negative_count} negative values"
        )