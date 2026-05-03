import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(filepath: str):
    """
    Returns
    -------
    X_scaled     : np.ndarray, shape (n_samples, n_features)
    feature_names: list[str]
    scaler       : fitted StandardScaler
    y_true       : np.ndarray | None, reference labels for accuracy checks
    target_name  : str | None, name of the preserved reference label column
    """
    try:
        df = pd.read_csv(filepath, sep='\t')  # type: ignore[call-overload]
        if df.shape[1] < 5:
            df = pd.read_csv(filepath, sep=',')  # type: ignore[call-overload]
    except Exception:
        df = pd.read_csv(filepath, sep=',')  # type: ignore[call-overload]

    y_true = None
    target_name = None
    if 'Response' in df.columns:
        target_name = 'Response'
        df['__reference_target__'] = df['Response']

    drop_cols = ['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue',
                 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())

    if 'Year_Birth' in df.columns:
        df['Age'] = 2024 - df['Year_Birth']
        df = df.drop(columns=['Year_Birth'])

    mnt_cols = [c for c in df.columns if c.startswith('Mnt')]
    if mnt_cols:
        df['TotalSpend'] = df[mnt_cols].sum(axis=1)

    purchase_cols = [c for c in df.columns if c.startswith('Num') and 'Purchases' in c]
    if purchase_cols:
        df['NumPurchases'] = df[purchase_cols].sum(axis=1)

    if 'Age' in df.columns:
        df = df[df['Age'] < 100].copy()

    if 'Income' in df.columns:
        income_cap = df['Income'].quantile(0.99)
        df['Income'] = df['Income'].clip(upper=income_cap)

    cat_cols = [c for c in ['Education', 'Marital_Status'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    reference_target = df.pop('__reference_target__') if '__reference_target__' in df.columns else None

    df = df.astype(float).dropna()

    if reference_target is not None:
        reference_target = reference_target.loc[df.index]
        if reference_target.dtype == object:
            reference_target = reference_target.astype(str).str.strip().str.lower().map({
                'yes': 1, '1': 1, 'true': 1,
                'no': 0, '0': 0, 'false': 0,
            }).fillna(reference_target)
        y_true = pd.factorize(reference_target)[0].astype(int)

    feature_names = list(df.columns)
    X_raw = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feature_names, scaler, y_true, target_name
