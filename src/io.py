from astropy.io import fits
import numpy as np
import pandas as pd
import cv2


def load_fits(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
    return data, header


def header_to_dataframe(header: fits.Header) -> pd.DataFrame:
    rows = []

    for key in header.keys():
        rows.append({
            "Keyword": key,
            "Value": header[key],
            "Comment": header.comments[key]
        })

    return pd.DataFrame(rows)


def load_image(path, grayscale=True):
    if grayscale:
        data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        data = cv2.imread(str(path), cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    return data.astype(np.float32)
