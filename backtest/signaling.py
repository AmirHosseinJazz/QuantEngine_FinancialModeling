import numpy as np


def convert_predictions_to_signals(predictions, threshold=0.5):
    signals = np.zeros(len(predictions))
    signals[predictions > threshold] = 1
    return signals
