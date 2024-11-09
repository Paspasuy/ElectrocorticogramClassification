import numpy as np
import pywt
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        pass
        
    def extract(self, segment: np.ndarray, partitions: int) -> np.ndarray:
        features = []
        segment_len = segment.shape[1]
        part_size = segment_len // partitions
        
        for channel in segment:
            for i in range(partitions):
                start = i * part_size
                end = (i + 1) * part_size
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
    
class WaveletFeatureExtractor:
    def __init__(self, scales=None):
        if scales is None:
            scales = list(range(30, 50))
        self.scales = scales
        pass
        
    def extract(self, segment: np.ndarray) -> np.ndarray:
        features = []
        segment_len = segment.shape[1]
        
        coefs = pywt.cwt(segment, self.scales, 'morl')[0] # len(scales) x channels x segment_len 
        coefs = coefs.transpose(1, 0, 2) # channels x len(scales) x segment_len
        
        # Concat with original segment (channels x segment_len)
        coefs = np.concatenate([segment[:, np.newaxis, :], coefs], axis=1)
        
        return coefs
            
        
    def transform(self, X: np.ndarray, partitions: int) -> np.ndarray:
        return np.array([self.extract(seg) for seg in tqdm(X, total=len(X))])
