from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, ttest_ind
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plotnine import (
    aes,
    element_text,
    geom_boxplot,
    geom_point,
    ggplot,
    labs,
    scale_fill_gradient2,
    scale_y_continuous,
    scale_x_continuous,
    scale_color_discrete,
    scale_fill_discrete,
    theme,
    theme_bw,
    geom_density,
    geom_tile,
    facet_wrap,
    coord_fixed,
    theme_minimal,
    coord_cartesian,
    scale_color_brewer,
)

def load_pima_data() -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes CSV (which already has a header row),
    coerce feature columns to numeric, and return the DataFrame.
    """
    # 1) Read the CSV, letting pandas use the first line as headers
    df = pd.read_csv("data/01_raw/pima_indians_diabetes.csv", header=0)

    # 2) Convert feature columns to numeric (in case of any stray strings)
    feature_cols = [c for c in df.columns if c != "outcome"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 3) Convert 'outcome' to integer (it should already be int, but just to be safe)
    df["outcome"] = df["outcome"].astype(int)

    return df

def rename_columns(
    df: pd.DataFrame,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """
    Rename columns in `df` according to `mapping`.
    Only columns present as keys in `mapping` will be renamed;
    all others remain untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame whose columns we want to rename.
    mapping : dict
        A dict of {old_name: new_name} for all columns you want to rename.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with columns renamed.
    """
    # Make a copy so we don’t modify the original in‐memory DataFrame
    df_out = df.copy()

    # Use pandas’ built‐in .rename() method
    df_out = df_out.rename(columns=mapping)

    return df_out

def calculate_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw Pima DataFrame and return a descriptive summary with variable
    names as a column, median, skewness, kurtosis, and missing counts included,
    sorted by kurtosis descending.
    """
    # get the default describe() table (count, mean, std, min, 25%, 50%, 75%, max)
    summary = df.describe().T

    # compute median, skew and kurtosis, and count missing values
    summary["median"] = df.median()
    summary["skew"] = df.skew()
    summary["kurt"] = df.kurtosis()
    summary["missing"] = df.isnull().sum()

    # reorder columns
    cols = [
        "count",
        "missing",
        "mean",
        "median",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "skew",
        "kurt"
    ]
    summary = summary[cols]

    # reset index to turn variable names into a column called "variable"
    summary = summary.reset_index().rename(columns={"index": "variable"})

    # sort by kurtosis, descending
    summary = summary.sort_values("kurt", ascending=False).reset_index(drop=True)

    return summary

def replace_zeroes_with_na(df: pd.DataFrame, cols_with_zeroes: list) -> pd.DataFrame:
    """
    Replaces 0 values with np.nan in the specified columns of a DataFrame.
    This is particularly useful for datasets like the Pima Indians Diabetes dataset
    where a zero value in certain physiological measurements (e.g., Glucose,
    BloodPressure, SkinThickness, Insulin, BMI) actually indicates a missing value.

    Args:
        df: The input pandas DataFrame.
        cols_with_zeroes: A list of column names in which zero values
                          should be replaced with np.nan.

    Returns:
        A pandas DataFrame with 0s replaced by np.nan in the specified columns.
    """
    # Work on a copy to avoid modifying the original DataFrame passed to the function,
    # which is good practice for Kedro nodes.
    df_processed = df.copy()

    for col_name in cols_with_zeroes:
        if col_name in df_processed.columns:
            # Replace all occurrences of 0 with np.nan in the current column
            df_processed[col_name] = df_processed[col_name].replace(0, np.nan)
        else:
            # Optionally, you might want to log a warning or raise an error
            # if a specified column doesn't exist in the DataFrame.
            print(f"Warning: Column '{col_name}' specified in cols_with_zeroes "
                  f"was not found in the DataFrame. This column will be skipped.")
            
    return df_processed

def plot_zscore_boxplots(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    title: Optional[str] = "Boxplots of Features (Z-scored)",
    highlight_outliers: bool = True,
    outlier_color: str = "steelblue"
) -> "ggplot":
    """
    Z-score all columns except `outcome_col` (and any metadata columns like "flag_imp"),
    then plot boxplots with y-limits [-6,6]. If highlight_outliers is True, color outliers
    in `outlier_color`.

    This version explicitly converts feature columns to numeric, drops entirely-NaN or
    zero-variance features, and raises if there is no data to plot.
    """

    # 1) Decide which columns to treat as “features”:
    #    - Exclude outcome_col
    #    - Exclude any metadata columns (e.g. “flag_imp”)
    features = [
        c
        for c in df.columns
        if c not in {outcome_col, "flag_imp"}
    ]

    # 2) Make a local copy so we don’t modify the original DataFrame
    df_numeric = df.copy()

    # 3) Explicitly coerce each feature column to numeric (strings → NaN if they fail)
    for col in features:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    # 4) Drop any feature column that is entirely NaN or has zero variance
    numeric_features: List[str] = []
    for col in features:
        col_data = df_numeric[col]
        if col_data.isna().all():
            # skip columns with no numeric data
            continue
        if col_data.std(skipna=True) == 0:
            # skip columns with zero variance
            continue
        numeric_features.append(col)

    if not numeric_features:
        raise ValueError(
            f"No valid numeric features left after dropping outcome and metadata. "
            f"Checked columns (excluding '{outcome_col}' and 'flag_imp'): {features}"
        )

    # 5) Compute Z-scores on the remaining numeric features
    df_z = df_numeric[numeric_features].apply(zscore, nan_policy="omit")

    # 6) Melt into long‐form for Plotnine
    df_melted = df_z.melt(var_name="Feature", value_name="Z_score")

    # 7) If there are no finite Z-scores, we cannot draw a boxplot
    if df_melted["Z_score"].dropna().empty:
        raise ValueError(
            "After computing z‐scores, no finite (non‐NaN) values remain. Cannot draw boxplots."
        )

    # 8) Choose the boxplot layer (with or without colored outliers)
    if highlight_outliers:
        box_layer = geom_boxplot(na_rm=True, outlier_color=outlier_color, notch=True, notchwidth = .25)
    else:
        box_layer = geom_boxplot(na_rm=True, notch=True, notchwidth = .25)

    # 9) Assemble the final plot
    p = (
        ggplot(df_melted, aes(x="Feature", y="Z_score"))
        + box_layer
        + theme_bw()
        + labs(title=title, y="Z-score", x="")
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
        + scale_y_continuous(limits=(-6, 6))
    )

    return p

def replace_outliers_with_na(
    df: pd.DataFrame,
    threshold: float,
    cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Identify values whose z‐score exceeds `threshold` (in absolute value)
    and replace them with NaN. Add a new column 'flag_imp' that is 1 if any
    outlier was replaced in that row, else 0.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame (numeric columns expected for outlier detection).
    threshold : float
        The z‐score cutoff. Any |z| > threshold is considered an outlier.
    cols : Optional[List[str]]
        List of column names to check for outliers. If None, all numeric
        columns in `df` will be used.

    Returns
    -------
    df_clean : pd.DataFrame
        A copy of `df` where each outlier (|z| > threshold) has been set to np.nan,
        plus a new column 'flag_imp' (0/1).
    """
    # 1) Decide which columns to examine
    if cols is None:
        cols_to_check = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols_to_check = cols

    # 2) Compute z‐scores on the selected columns
    df_z = df[cols_to_check].apply(zscore, nan_policy="omit")

    # 3) Identify outliers: a boolean DataFrame where True = |z| > threshold
    is_outlier = df_z.abs() > threshold

    # 4) Create a cleaned copy of the original DataFrame
    df_clean = df.copy()

    # 5) For each column, set rows where |z| > threshold to NaN
    for col in cols_to_check:
        df_clean.loc[is_outlier[col], col] = np.nan

    # 6) Create 'flag_imp': 1 if any column in that row was an outlier, else 0
    #    (axis=1 to check per‐row, then convert boolean → int)
    row_flag = is_outlier.any(axis=1).astype(int)
    df_clean["flag_imp"] = row_flag

    return df_clean

def impute_missing_values(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: Optional[int] = 0
) -> pd.DataFrame:
    """
    Use IterativeImputer to fill in missing values in all feature columns of the Pima dataset,
    then reattach the "outcome" column unchanged and add a Boolean 'flag_imp' column that is
    True for any row that contained at least one NaN in a feature before imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, which must contain an "outcome" column and may contain NaNs
        among the other (feature) columns.

    max_iter : int
        Number of imputation rounds. Passed directly to IterativeImputer(max_iter=...).

    random_state : Optional[int]
        Random seed for reproducibility (if None, draws will be non-deterministic).

    Returns
    -------
    pd.DataFrame
        A new DataFrame in which all NaNs in the feature columns have been replaced by
        the IterativeImputer. The "outcome" column is copied over unchanged, and a new
        Boolean column "flag_imp" is True if the original row had any missing feature.
    """
    # 1) Verify "outcome" exists
    if "outcome" not in df.columns:
        raise KeyError("Input DataFrame must contain an 'outcome' column.")

    # 2) Identify feature columns (everything except "outcome")
    features: List[str] = [c for c in df.columns if c != "outcome"]

    # 3) Determine which rows have at least one NaN in any feature
    original_na_mask = df[features].isna().any(axis=1)

    # 4) Split into X (features) and y (outcome)
    X = df[features]
    y = df["outcome"]

    # 5) Initialize and run the IterativeImputer
    imp = IterativeImputer(
        max_iter=max_iter,
        sample_posterior=True,
        random_state=random_state
    )
    X_imputed_array = imp.fit_transform(X)

    # 6) Reconstruct a DataFrame with the same feature names and original index
    df_imputed = pd.DataFrame(
        X_imputed_array,
        columns=features,
        index=df.index
    )

    # 7) Reattach the "outcome" column unchanged
    df_imputed["outcome"] = y.values

    # 8) Add the Boolean flag_imp column
    df_imputed["flag_imp"] = original_na_mask.values

    return df_imputed

def plot_density_by_imputation(

    df: pd.DataFrame,
    flag_col: str = "flag_imp",
    outcome_col: str = "outcome",
    title: Optional[str] = "Density of Scaled Features by Imputation Flag"
) -> "ggplot":
    """
    Take a DataFrame (with one “flag_imp” column and one “outcome” column),
    scale each numeric feature to a z‐score, then draw density plots (one facet
    per feature) grouped by whether a row was imputed (flag_imp == 1) or not
    (flag_imp == 0).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
          - One column named `flag_col` (0/1 indicator for imputed rows)
          - One column named `outcome_col` (left out of features)
          - Eight numeric feature columns (e.g. "pregnant", "glucose", etc.)
    flag_col : str
        Name of the binary column indicating if any value in that row was imputed.
    outcome_col : str
        Name of the “outcome” column to exclude from plotting.
    title : Optional[str]
        Title for the overall ggplot object.

    Returns
    -------
    ggplot
        A Plotnine ggplot object with one density curve per feature facet,
        colored/fill‐shaded by `flag_col`.
    """

    # 1) Identify feature columns (exclude outcome and flag columns)
    features = [c for c in df.columns if c not in {outcome_col, flag_col}]

    # 2) Create a copy and coerce each feature to numeric (strings → NaN)
    df_numeric = df.copy()
    for col in features:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    # 3) Compute z‐scores for each feature (NaNs are allowed)
    df_z = df_numeric[features].apply(zscore, nan_policy="omit")

    # 4) Attach the flag column (0/1) to the scaled frame
    df_z[flag_col] = df_numeric[flag_col]

    # 5) Melt into long form: one row per (Feature, ScaledValue, flag_imp)
    df_melted = pd.melt(
        df_z,
        id_vars=flag_col,
        var_name="Feature",
        value_name="ScaledValue"
    )

    # 6) Drop any rows where ScaledValue is NaN
    df_melted = df_melted.dropna(subset=["ScaledValue"])

    # 7) Build the density plot: one facet per Feature, group by flag_imp
    p = (
        ggplot(
            df_melted,
            aes(
                x="ScaledValue",
                color=f"factor({flag_col})",
                fill=f"factor({flag_col})"
            )
        )
        + geom_density(alpha=0.3)
        + facet_wrap("~Feature", scales="fixed")
        + coord_cartesian(xlim=(-4, 4))
        + theme_bw()
        + labs(
            title=title,
            x="Z‐score (scaled value)",
            color="Imputed",
            fill="Imputed"
        ) 
    )

    return p

def plot_density_by_outcome(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    title: Optional[str] = "Density of Scaled Features by Outcome"
) -> "ggplot":
    """
    Take a DataFrame (with one “outcome” column),
    scale each numeric feature to a z‐score, then draw density plots (one facet
    per feature) grouped by outcome (outcome == 1 vs outcome == 0).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
          - One column named `outcome_col` (binary indicator for outcome)
          - Several numeric feature columns
    outcome_col : str
        Name of the binary column indicating the outcome.
    title : Optional[str]
        Title for the overall ggplot object.

    Returns
    -------
    ggplot
        A Plotnine ggplot object with one density curve per feature facet,
        colored/fill‐shaded by `outcome_col`, x‐axis fixed from −4 to 4.
    """

    # 1) Identify feature columns (exclude outcome column)
    features = [c for c in df.columns if c not in {outcome_col, "flag_imp"}]

    # 2) Create a copy and coerce each feature to numeric (strings → NaN)
    df_numeric = df.copy()
    for col in features:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    # 3) Compute z‐scores for each feature (NaNs are allowed)
    df_z = df_numeric[features].apply(zscore, nan_policy="omit")

    # 4) Attach the outcome column to the scaled frame
    df_z[outcome_col] = df_numeric[outcome_col]

    # 5) Melt into long form: one row per (Feature, ScaledValue, outcome)
    df_melted = pd.melt(
        df_z,
        id_vars=outcome_col,
        var_name="Feature",
        value_name="ScaledValue"
    )

    # 6) Drop any rows where ScaledValue is NaN
    df_melted = df_melted.dropna(subset=["ScaledValue"])

    # 7) Build the density plot: one facet per Feature, group by outcome
    p = (
        ggplot(
            df_melted,
            aes(
                x="ScaledValue",
                color=f"factor({outcome_col})",
                fill=f"factor({outcome_col})"
            )
        )
        + geom_density(alpha=0.3)
        + scale_color_discrete(labels=["negative", "positive"])
        + scale_fill_discrete(labels=["negative", "positive"])
        + facet_wrap("~Feature", scales="fixed")
        + coord_cartesian(xlim=(-4, 4))
        + theme_bw()
        + labs(
            title=title,
            x="Z‐score (scaled value)",
            color="Outcome",
            fill="Outcome"
        )
    )

    return p

def compute_t_test(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    flag_col: str = "flag_imp",
    alpha: float = 0.1
) -> pd.DataFrame:
    """
    Compute group means and independent‐samples t‐tests for each numeric feature
    in `df`, comparing outcome == 0 vs. outcome == 1. Excludes `outcome_col` and
    `flag_col` from the feature set. Adds a column "significant" indicating
    whether p_value < alpha.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
          - A binary `outcome_col`
          - A binary `flag_col` (to be excluded)
          - Several numeric feature columns
    outcome_col : str
        Name of the binary outcome column.
    flag_col : str
        Name of the binary flag column to exclude.
    alpha : float
        Significance threshold for p‐value (default 0.05).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
          - feature
          - mean_outcome0
          - mean_outcome1
          - t_stat
          - p_value
          - significant (True if p_value < alpha, else False)
    """
    # 1) Identify feature columns (exclude outcome and flag)
    features = [c for c in df.columns if c not in {outcome_col, flag_col}]

    rows = []
    for feature in features:
        grp0 = df.loc[df[outcome_col] == 0, feature].dropna().astype(float)
        grp1 = df.loc[df[outcome_col] == 1, feature].dropna().astype(float)

        mean0 = grp0.mean()
        mean1 = grp1.mean()

        tstat, pval = ttest_ind(grp0, grp1, equal_var=False)

        rows.append({
            "feature": feature,
            "mean_outcome0": mean0,
            "mean_outcome1": mean1,
            "t_stat": tstat,
            "p_value": pval,
            "significant": pval < alpha
        })
    result_df = pd.DataFrame(rows)
    # Sort by p_value ascending
    result_df = result_df.sort_values("p_value").reset_index(drop=True)
    return result_df

def plot_correlogram(
    df: pd.DataFrame, 
    exclude_cols: Optional[List[str]] = None, 
    title: str = "Correlogram"
) -> plt.Figure:
    """
    Draw a correlogram (correlation‐matrix heatmap) using pure Matplotlib,
    with correlation values overlaid on each tile.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric columns (including “outcome” if desired).
    exclude_cols : Optional[List[str]]
        Columns to drop from the correlation (e.g. ["flag_imp"]).
        If None, defaults to excluding only "flag_imp" if present.
    title : str
        Figure title.

    Returns
    -------
    fig : plt.Figure
        The Matplotlib Figure containing the correlogram.
    """
    if exclude_cols is None:
        exclude_cols = []
    # Automatically drop "flag_imp" if present, but do NOT drop "outcome"
    if "flag_imp" in df.columns and "flag_imp" not in exclude_cols:
        exclude_cols.append("flag_imp")

    # 1) Select numeric columns, excluding unwanted ones
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns 
        if c not in exclude_cols
    ]
    if not numeric_cols:
        raise ValueError(f"No numeric columns left after excluding {exclude_cols}.")

    # 2) Compute Pearson correlation matrix
    corr = df[numeric_cols].corr()

    # 3) Build a mask for the upper triangle (k=1 excludes diagonal too)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # 4) Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.close(fig)

    # 5) Plot the lower triangle of corr (mask out upper triangle)
    corr_masked = corr.mask(mask)
    im = ax.imshow(
        corr_masked.values,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        interpolation="nearest",
        aspect="equal",
        origin="lower"
    )

    # 6) Overlay correlation values on each visible tile
    n = len(numeric_cols)
    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                value = corr_masked.iloc[i, j]
                ax.text(
                    j, i, 
                    f"{value:.2f}", 
                    ha="center", va="center", 
                    color="black", 
                    fontsize=8
                )

    # 7) Add a colorbar beside the heatmap
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson $r$", rotation=270, labelpad=15)

    # 8) Set ticks to be at cell centers, and label them with column names
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)

    # 9) Draw grid lines between cells
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 10) Title and layout
    ax.set_title(title)
    fig.tight_layout()

    return fig

def plot_pca_by_outcome(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    title: str = "PCA Scatterplot (PC1 vs PC2) by Outcome"
) -> "ggplot":
    """
    Run PCA on all columns except `outcome_col` and `flag_imp`, then
    return a Plotnine scatterplot of PC1 vs PC2, colored by outcome label
    (0 → 'healthy', 1 → 'diabetic') with axes fixed to [-4,4].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
          - All numeric feature columns
          - One column named `outcome_col` (binary 0/1)
          - An optional 'flag_imp' column (will be dropped)
    outcome_col : str
        Name of the binary outcome column.
    title : str
        Title for the plot.

    Returns
    -------
    ggplot
        A Plotnine scatterplot with PC2 on the x-axis, PC1 on the y-axis,
        colored by 'healthy' vs 'diabetic'. Axes are limited to [-4, 4].
    """
    # 1) Identify feature columns (drop outcome and flag_imp if present)
    drop_cols = {outcome_col, "flag_imp"}
    features = [c for c in df.columns if c not in drop_cols]

    # 2) Standardize those features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # 3) Compute PCA (keep first two components)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    # 4) Build a DataFrame for plotting
    pca_df = pd.DataFrame({
        "pca_one": components[:, 0],
        "pca_two": components[:, 1],
        "outcome_label": df[outcome_col]
            .map({0: "healthy", 1: "diabetic"})
            .astype("category")
    })

    # 5) Create and return the Plotnine scatterplot
    p = (
        ggplot(pca_df, aes(x="pca_two", y="pca_one", color="outcome_label"))
        + geom_point(size=2, alpha=0.5)
        + scale_color_brewer(type="qual", palette="Set1")
        + coord_fixed(ratio=1, xlim=(-4, 4), ylim=(-4, 4))
        + theme_bw()
        + labs(
            x="PC2",
            y="PC1",
            color="Outcome",
            title=title
        )
    )
    return p

def create_train_test_splits(
    df: pd.DataFrame,
    outcome_col: str,
    test_size: float,
    random_state: int,
    stratify: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Given the fully preprocessed DataFrame (features + outcome),
    split into train/test and return X_train, X_test, y_train, y_test.

    Parameters
    ----------
    df : pd.DataFrame
        The processed DataFrame containing all feature columns plus `outcome`.
    outcome_col : str
        Name of the target column to separate off.
    test_size : float
        Fraction of data to reserve for the test set.
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        If True, stratify by the outcome column, else do not stratify.

    Returns
    -------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
    y_test  : pd.Series
    """
    X = df.drop(columns=[outcome_col])
    y = df[outcome_col]

    stratify_vals = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vals
    )
    return X_train, X_test, y_train, y_test

