import pandas as pd

def format_table(
    df: pd.DataFrame,
    decimals: int = 2,
    hide_index: bool = True
):
    """
    Return a pandas Styler for `df` with consistent formatting and
    optional index hiding. Floats are displayed to `decimals` places.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame you want to display.
    decimals : int
        Number of decimal places for all float columns.
    hide_index : bool
        If True, hide the DataFrame index (row numbers).

    Returns
    -------
    Styler
        A pandas Styler object with Bootstrap‐style classes, padding,
        header background, and optional index hiding.
    """
    # Identify numeric columns (so we don’t try to apply a float format to strings)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    float_fmt = f"{{:.{decimals}f}}"
    fmt_dict = {col: float_fmt for col in numeric_cols}

    # Start styling
    styler = df.style.format(fmt_dict)

    # Apply Bootstrap-like classes for striped rows and hover effect
    styler = styler.set_table_attributes(
        'class="dataframe table table-striped table-hover" '
        'style="border-collapse:collapse; font-size:0.9em;"'
    )

    # Style the header cells (th)
    styler = styler.set_table_styles([{
        "selector": "th",
        "props": [
            ("background-color", "#f2f2f2"),
            ("color", "#333"),
            ("font-weight", "bold"),
            ("padding", "0.5em")
        ]
    }], overwrite=False)

    # Style the data cells (td)
    styler = styler.set_table_styles([{
        "selector": "td",
        "props": [
            ("padding", "0.5em"),
            ("text-align", "right")
        ]
    }], overwrite=False)

    if hide_index:
        try:
            # for pandas >= 1.3
            styler = styler.hide_index()
        except AttributeError:
            # fallback for older pandas versions
            styler = styler.hide(axis="index")

    return styler
