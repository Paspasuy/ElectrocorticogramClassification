import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass
        
    def extract(self, segment: np.ndarray, partitions: int) -> np.ndarray:
        features = []
        segment_len = segment.shape[1]
        part_size = segment_len // partitions
        
        for channel in segment:
            for i in range(4):
                start = i * part_size
                end = (i + 1) * part_size if i < 3 else segment_len
                part = channel[start:end]
                
                mean = np.mean(part)
                var = np.var(part)
                mean = (part.max() - part.min()) / 2
                crossings = np.sum((part[:-1] < mean) & (part[1:] >= mean)) + \
                           np.sum((part[:-1] > mean) & (part[1:] <= mean))
                crossings_norm = crossings / len(part)
                features.extend([mean, var, crossings_norm])
        return np.array(features)
        
    def transform(self, X: np.ndarray, partitions: int) -> np.ndarray:
        return np.array([self.extract(seg, partitions) for seg in X])
    
    
class DummyFeatureExtractor:
    def __init__(self):
        pass
        
    def extract(self, segment: np.ndarray, partitions: int) -> np.ndarray:
        return segment
        
    def transform(self, X: np.ndarray, partitions: int) -> np.ndarray:
        return self.extract(X, partitions)
