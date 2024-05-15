X.iloc[1, 0] = np.nan
    print(X.isna().sum())