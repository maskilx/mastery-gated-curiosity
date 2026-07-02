import numpy as np

class Environment:
    def __init__(self, num_regions=5, max_regions=10, noise_learnable=0.03, noise_unlearnable=1.0, val_samples_per_region=100, seed=42):
        self.num_regions = num_regions
        self.max_regions = max_regions
        self.noise_learnable = noise_learnable
        self.noise_unlearnable = noise_unlearnable
        self.val_samples_per_region = val_samples_per_region
        self.rng = np.random.default_rng(seed)
        
        self.active_mask = np.zeros(self.max_regions, dtype=np.float32)
        self.active_mask[:self.num_regions] = 1.0
        
        # Fixed validation x-values for each region
        self.val_x = {}
        for r in range(self.max_regions):
            self.val_x[r] = self.rng.uniform(-1, 1, size=self.val_samples_per_region)
            
    def _true_function(self, region_id, x):
        if region_id == 0: # linear
            return 2 * x + 0.5
        elif region_id == 1: # quadratic
            return x**2 - 0.3 * x + 0.2
        elif region_id == 2: # periodic
            return np.sin(3 * np.pi * x)
        elif region_id == 3: # noisy
            return 0.0
        elif region_id == 4: # piecewise learnable
            # Handle both scalar and array inputs
            if isinstance(x, np.ndarray):
                return np.where(x < 0, -x, 0.5 * x + 0.2)
            else:
                return -x if x < 0 else 0.5 * x + 0.2
        else:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0

    def measure(self, region_id, x):
        """Returns y = f(x) + noise"""
        y_true = self._true_function(region_id, x)
        if region_id == 3:
            return y_true + self.rng.normal(0, self.noise_unlearnable, size=np.shape(x))
        else:
            return y_true + self.rng.normal(0, self.noise_learnable, size=np.shape(x))
            
    def get_validation_set(self, region_id):
        """Returns x, y validation points.
        For region 3, it generates fresh noise at the fixed val_x.
        For other regions, it returns the noiseless true function to measure true generalization.
        Wait, the prompt says 'For the noisy region, estimate error using repeated measurements.'
        So for region 3 we return freshly sampled noisy targets.
        """
        x = self.val_x[region_id]
        if region_id == 3:
            y = self.measure(region_id, x)
        else:
            y = self._true_function(region_id, x)
        return x, y
