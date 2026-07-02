import numpy as np
from scipy.optimize import curve_fit
import warnings

def _constant(x, c):
    return np.full_like(x, c)

def _linear(x, a, b):
    return a * x + b

def _quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def _cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def _sine(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def _piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0, x >= x0],
                        [lambda x: k1*(x-x0) + y0, 
                         lambda x: k2*(x-x0) + y0])

def _exp(x, a, b):
    return a * np.exp(x) + b

def _log(x, a, b):
    # Ensure x+b > 0 to avoid warnings
    val = x + b
    val = np.maximum(val, 1e-5)
    return a * np.log(val)

class HypothesisDiscriminator:
    def __init__(self):
        self.models = {
            'constant': _constant,
            'linear': _linear,
            'quadratic': _quadratic,
            'cubic': _cubic,
            'sine': _sine,
            'piecewise_linear': _piecewise_linear,
            'exp': _exp,
            'log': _log
        }
        
    def fit_and_predict(self, X_train, Y_train, X_candidates):
        """
        X_train: (N,)
        Y_train: (N,)
        X_candidates: (M,)
        Returns: predictions array (num_successful_models, M)
        """
        predictions = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, func in self.models.items():
                try:
                    p0 = None
                    if name == 'sine':
                        p0 = [1.0, 3*np.pi, 0, 0]
                    elif name == 'piecewise_linear':
                        p0 = [0, 0, -1, 1]
                        
                    popt, _ = curve_fit(func, X_train, Y_train, p0=p0, maxfev=2000)
                    preds = func(X_candidates, *popt)
                    predictions.append(preds)
                except Exception:
                    pass
                    
        if len(predictions) == 0:
            return None
        return np.array(predictions)
