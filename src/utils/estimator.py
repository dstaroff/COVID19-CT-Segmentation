import numpy as np

import src.segmentation.image.preprocessing as Preprocessor


def content(minor: np.ndarray, major: np.ndarray) -> float:
    m_1 = Preprocessor.binarize(minor)
    m_2 = Preprocessor.binarize(major)
    intersection = np.where((m_1 * m_2) == 1, 1, 0)

    return np.count_nonzero(intersection) / np.count_nonzero(m_2)
