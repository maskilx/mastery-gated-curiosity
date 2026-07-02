import numpy as np
from collections import deque

class RegionTracker:
    def __init__(self, num_regions, config):
        self.num_regions = num_regions
        self.config = config
        
        # Tracking metrics per region
        self.num_samples = {r: 0 for r in range(num_regions)}
        self.val_error = {r: 1.0 for r in range(num_regions)} # High initial
        self.preq_error_ema = {r: 1.0 for r in range(num_regions)}
        self.uncertainty = {r: 1.0 for r in range(num_regions)}
        self.learning_progress = {r: config.get('initial_lp_prior', 0.1) for r in range(num_regions)}
        self.noise_risk = {r: 0.0 for r in range(num_regions)}
        
        self.status = {r: 'unexplored' for r in range(num_regions)}
        
        # History for learning progress
        self.val_error_history = {r: deque(maxlen=5) for r in range(num_regions)}
        self.preq_error_history = {r: deque(maxlen=5) for r in range(num_regions)}
        
        # For estimating noise risk
        self.repeated_measurements = {r: {} for r in range(num_regions)}
        
    def update_metrics(self, region_id, val_error, uncertainty, preq_error=None):
        self.val_error_history[region_id].append(val_error)
        
        self.val_error[region_id] = val_error
        
        if preq_error is not None:
            self.preq_error_ema[region_id] = 0.9 * self.preq_error_ema[region_id] + 0.1 * preq_error
        else:
            self.preq_error_ema[region_id] = val_error # fallback
            
        self.preq_error_history[region_id].append(self.preq_error_ema[region_id])
        
        # Calculate learning progress
        if len(self.preq_error_history[region_id]) > 1:
            # LP = error_past - error_current
            past_error = self.preq_error_history[region_id][0]
            self.learning_progress[region_id] = past_error - self.preq_error_ema[region_id]
        else:
            self.learning_progress[region_id] = self.config.get('initial_lp_prior', 0.1)
            
        self.uncertainty[region_id] = uncertainty
        
        if self.status[region_id] == 'unexplored' and self.num_samples[region_id] > 0:
            self.status[region_id] = 'learning'
            
        self._update_status(region_id)
        
    def record_sample(self, region_id, x, y):
        self.num_samples[region_id] += 1
        
        # Track repeated measurements for noise risk calculation
        x_rounded = round(float(x), 4)
        if x_rounded not in self.repeated_measurements[region_id]:
            self.repeated_measurements[region_id][x_rounded] = []
        self.repeated_measurements[region_id][x_rounded].append(y)
        
        self._update_noise_risk(region_id)
        
    def _update_noise_risk(self, region_id):
        variances = []
        for x, y_list in self.repeated_measurements[region_id].items():
            if len(y_list) > 1:
                variances.append(np.var(y_list))
                
        if variances:
            self.noise_risk[region_id] = np.mean(variances)
            
    def _update_status(self, region_id):
        c = self.config
        # We don't revert from terminal states in this prototype
        if self.status[region_id] in ['mastered', 'blocked_noisy']:
            return
            
        # Check Mastery (using preq_error_ema instead of val_error)
        if (self.preq_error_ema[region_id] < c.get('tau_error', 0.03) and 
            self.uncertainty[region_id] < c.get('tau_uncertainty', 0.02) and 
            abs(self.learning_progress[region_id]) < c.get('tau_progress', 0.002) and 
            self.num_samples[region_id] >= c.get('min_samples', 30)):
            self.status[region_id] = 'mastered'
            
        # Check Blocked / Noisy (using preq_error_ema instead of val_error)
        elif (self.preq_error_ema[region_id] > c.get('tau_high_error', 0.1) and 
              self.learning_progress[region_id] < c.get('tau_progress', 0.002) and 
              self.noise_risk[region_id] > c.get('tau_noise', 0.2) and 
              self.num_samples[region_id] >= c.get('min_samples', 30)):
            self.status[region_id] = 'blocked_noisy'
