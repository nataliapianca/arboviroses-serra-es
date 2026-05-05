# src/mock_data.py

import pandas as pd
import numpy as np

DOENCAS = ["dengue", "zika", "chikungunya"]

CORES = {
    "dengue": "rgba(255, 99, 132, 0.8)",
    "zika": "rgba(54, 162, 235, 0.8)",
    "chikungunya": "rgba(255, 206, 86, 0.8)",
}

def get_serie_temporal():
    datas = pd.date_range(start="2015-01-01", end="2023-12-01", freq="MS")

    np.random.seed(42)

    df = pd.DataFrame({
        "data": datas,
        "dengue": np.random.randint(50, 500, len(datas)),
        "zika": np.random.randint(10, 150, len(datas)),
        "chikungunya": np.random.randint(5, 100, len(datas)),
    })

    return df