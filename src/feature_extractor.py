import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass
        
    def extract(self, segment: np.ndarray) -> np.ndarray:
        features = []
        for channel in segment:
            mean = np.mean(channel)
            var = np.var(channel)
            median = np.median(channel)
            crossings = np.sum((channel[:-1] < median) & (channel[1:] >= median)) + \
                       np.sum((channel[:-1] > median) & (channel[1:] <= median))
            crossings_norm = crossings / len(channel)
            features.extend([mean, var, crossings_norm])
        return np.array(features)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.extract(seg) for seg in X])