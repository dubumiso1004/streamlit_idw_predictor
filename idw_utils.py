import numpy as np

def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, str(dms_str).split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

def idw_interpolation(df, lat_col, lon_col, value_col, lat, lon, power=2):
    df_valid = df[[lat_col, lon_col, value_col]].dropna()
    distances = np.sqrt((df_valid[lat_col] - lat)**2 + (df_valid[lon_col] - lon)**2)
    weights = 1 / (distances**power + 1e-8)
    values = df_valid[value_col]
    return np.sum(weights * values) / np.sum(weights)
