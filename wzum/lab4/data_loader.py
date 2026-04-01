def load_data():
    from sklearn.datasets import fetch_openml
    import pandas as pd
    import numpy as np

    data = fetch_openml(name='diabetes', version=1, as_frame=True)
    df = data.frame

    cols = ['plas', 'pres', 'skin', 'insu', 'mass']
    df[cols] = df[cols].replace(0, np.nan)

    return df