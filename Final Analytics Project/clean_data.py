def clean_data(data, target, drop=None, onehot=False, standardize=False):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    if drop is None:
        drop = []
    y = data[target].copy()
    X = data.drop(columns=[target] + drop, errors='ignore')
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    transformers = []
    if standardize and num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    elif num_cols:
        transformers.append(('num', 'passthrough', num_cols))
    if onehot and cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols))
    elif cat_cols:
        transformers.append(('cat', 'passthrough', cat_cols))
    ct = ColumnTransformer(transformers=transformers)
    X_transformed = ct.fit_transform(X)
    cols = ct.get_feature_names_out()
    X_transformed = pd.DataFrame(X_transformed, columns=cols, index=X.index)
    X_transformed.columns = [c.split('__')[-1] for c in X_transformed.columns]
    return X_transformed, y