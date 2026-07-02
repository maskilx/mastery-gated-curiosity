import numpy as np

class ProceduralEnvironment:
    def __init__(self, difficulty='easy', max_regions=10, fixed_num_active=None, seed=42, forced_families=None):
        self.max_regions = max_regions
        self.difficulty = difficulty
        self.rng = np.random.default_rng(seed)
        
        if fixed_num_active is not None:
            self.num_active = fixed_num_active
        else:
            self.num_active = self.rng.integers(3, max_regions + 1)
            
        self.active_mask = np.zeros(self.max_regions, dtype=np.float32)
        self.active_mask[:self.num_active] = 1.0
        
        self.val_samples_per_region = 100
        self.val_x = {}
        for r in range(self.max_regions):
            self.val_x[r] = self.rng.uniform(-1, 1, size=self.val_samples_per_region)
            
        self.regions = []
        
        if difficulty == 'easy':
            families = ['constant', 'linear', 'quadratic']
            noise_range = (0.01, 0.05)
            allow_noisy = False
        elif difficulty == 'medium':
            families = ['linear', 'quadratic', 'cubic', 'sine', 'piecewise_linear']
            noise_range = (0.03, 0.1)
            allow_noisy = True
        elif difficulty == 'hard':
            families = ['quadratic', 'cubic', 'sine', 'piecewise_linear', 'exp', 'log']
            noise_range = (0.05, 0.2)
            allow_noisy = True
            
        elif difficulty == 'stress_test':
            families = ['deceptive_noisy_to_clean', 'deceptive_simple_to_complex', 
                        'deceptive_narrow_peak', 'noisy_learnable', 'high_freq_sine']
            noise_range = (0.05, 0.2)
            allow_noisy = True
            
        if forced_families is not None:
            families = forced_families
            allow_noisy = 'noisy_unlearnable' in forced_families
            
        learnable_added = False
        
        for r in range(self.num_active):
            fam = self.rng.choice(families)
            
            if allow_noisy and self.rng.random() < 0.2:
                if not learnable_added and r == self.num_active - 1:
                    fam = self.rng.choice(families)
                else:
                    fam = 'noisy_unlearnable'
                    
            if fam != 'noisy_unlearnable':
                learnable_added = True
                
            noise = self.rng.uniform(*noise_range)
            params = self._generate_params(fam, difficulty)
            
            self.regions.append({
                'family': fam,
                'noise': noise if fam not in ['noisy_unlearnable', 'noisy_learnable'] else self.rng.uniform(0.5, 1.5),
                'params': params,
                'is_learnable': fam != 'noisy_unlearnable'
            })
            
        for r in range(self.num_active, self.max_regions):
            self.regions.append({
                'family': 'constant',
                'noise': 0.0,
                'params': {'c': 0},
                'is_learnable': False
            })

    def _generate_params(self, fam, difficulty):
        p = {}
        if fam == 'constant':
            p['c'] = self.rng.uniform(-2, 2)
        elif fam == 'linear':
            p['a'] = self.rng.uniform(-3, 3)
            p['b'] = self.rng.uniform(-2, 2)
        elif fam == 'quadratic':
            p['a'] = self.rng.uniform(-2, 2)
            p['b'] = self.rng.uniform(-2, 2)
            p['c'] = self.rng.uniform(-1, 1)
        elif fam == 'cubic':
            p['a'] = self.rng.uniform(-1, 1)
            p['b'] = self.rng.uniform(-1, 1)
            p['c'] = self.rng.uniform(-1, 1)
            p['d'] = self.rng.uniform(-1, 1)
        elif fam == 'sine':
            if difficulty == 'hard':
                p['freq'] = self.rng.uniform(2*np.pi, 6*np.pi)
            else:
                p['freq'] = self.rng.uniform(np.pi, 3*np.pi)
            p['phase'] = self.rng.uniform(0, 2*np.pi)
            p['amp'] = self.rng.uniform(0.5, 2.0)
            p['offset'] = self.rng.uniform(-1, 1)
        elif fam == 'piecewise_linear':
            p['split'] = self.rng.uniform(-0.8, 0.8)
            p['m1'] = self.rng.uniform(-2, 2)
            p['b1'] = self.rng.uniform(-1, 1)
            p['m2'] = self.rng.uniform(-2, 2)
            if difficulty == 'hard':
                p['b2'] = self.rng.uniform(-1, 1)
            else:
                p['b2'] = p['m1']*p['split'] + p['b1'] - p['m2']*p['split']
        elif fam == 'exp':
            p['a'] = self.rng.uniform(0.1, 1.5) * self.rng.choice([-1, 1])
            p['b'] = self.rng.uniform(-1, 1)
        elif fam == 'log':
            p['a'] = self.rng.uniform(0.1, 1.5) * self.rng.choice([-1, 1])
            p['b'] = self.rng.uniform(1.1, 2.0)
            
        elif fam == 'deceptive_noisy_to_clean':
            p['base_fam'] = self.rng.choice(['linear', 'quadratic'])
            p['base_params'] = self._generate_params(p['base_fam'], difficulty)
            p['noise_scale'] = self.rng.uniform(1.0, 3.0)
            p['noise_decay'] = self.rng.uniform(5.0, 15.0)
            
        elif fam == 'deceptive_simple_to_complex':
            p['split'] = self.rng.uniform(0.3, 0.7) * self.rng.choice([-1, 1])
            p['base_val'] = self.rng.uniform(-1, 1)
            p['freq'] = self.rng.uniform(6*np.pi, 12*np.pi)
            p['amp'] = self.rng.uniform(0.5, 2.0)
            
        elif fam == 'deceptive_narrow_peak':
            p['center'] = self.rng.uniform(-0.8, 0.8)
            p['width'] = self.rng.choice([0.01, 0.03, 0.05, 0.08])
            p['height'] = self.rng.uniform(2.0, 5.0) * self.rng.choice([-1, 1])
            p['base_val'] = self.rng.uniform(-1, 1)
            
        elif fam == 'noisy_learnable':
            p['base_fam'] = self.rng.choice(['sine', 'quadratic'])
            p['base_params'] = self._generate_params(p['base_fam'], difficulty)
            
        elif fam == 'high_freq_sine':
            p['freq'] = self.rng.uniform(8*np.pi, 15*np.pi)
            p['phase'] = self.rng.uniform(0, 2*np.pi)
            p['amp'] = self.rng.uniform(0.5, 2.0)
            p['offset'] = self.rng.uniform(-1, 1)
            
        return p

    def _true_function(self, r, x):
        reg = self.regions[r]
        fam = reg['family']
        p = reg['params']
        
        x_arr = np.asarray(x)
        
        if fam == 'constant':
            res = np.full_like(x_arr, p['c'], dtype=np.float64)
        elif fam == 'linear':
            res = p['a'] * x_arr + p['b']
        elif fam == 'quadratic':
            res = p['a'] * x_arr**2 + p['b'] * x_arr + p['c']
        elif fam == 'cubic':
            res = p['a'] * x_arr**3 + p['b'] * x_arr**2 + p['c'] * x_arr + p['d']
        elif fam == 'sine':
            res = p['amp'] * np.sin(p['freq'] * x_arr + p['phase']) + p['offset']
        elif fam == 'piecewise_linear':
            res = np.where(x_arr < p['split'], p['m1']*x_arr + p['b1'], p['m2']*x_arr + p['b2'])
        elif fam == 'exp':
            res = p['a'] * np.exp(x_arr) + p['b']
        elif fam == 'log':
            res = p['a'] * np.log(x_arr + p['b'])
        elif fam == 'noisy_unlearnable':
            res = np.zeros_like(x_arr, dtype=np.float64)
        elif fam == 'deceptive_noisy_to_clean' or fam == 'noisy_learnable':
            base_reg = {'family': p['base_fam'], 'params': p['base_params']}
            old_reg = self.regions[r]
            self.regions[r] = base_reg
            res = self._true_function(r, x_arr)
            self.regions[r] = old_reg
            if np.isscalar(x): res = res.item()
        elif fam == 'deceptive_simple_to_complex':
            if p['split'] > 0:
                res = np.where(x_arr < p['split'], p['base_val'], p['base_val'] + p['amp'] * np.sin(p['freq'] * (x_arr - p['split'])))
            else:
                res = np.where(x_arr > p['split'], p['base_val'], p['base_val'] + p['amp'] * np.sin(p['freq'] * (x_arr - p['split'])))
        elif fam == 'deceptive_narrow_peak':
            res = p['base_val'] + p['height'] * np.exp(-((x_arr - p['center'])**2) / (2 * p['width']**2))
        elif fam == 'high_freq_sine':
            res = p['amp'] * np.sin(p['freq'] * x_arr + p['phase']) + p['offset']
            
        if np.isscalar(x) and isinstance(res, np.ndarray):
            return res.item()
        return res

    def measure(self, r, x):
        y_true = self._true_function(r, x)
        reg = self.regions[r]
        noise = reg['noise']
        
        x_arr = np.asarray(x)
        
        if reg['family'] == 'deceptive_noisy_to_clean':
            p = reg['params']
            x_dep_noise = noise + p['noise_scale'] * np.exp(-p['noise_decay'] * (x_arr**2))
            if np.isscalar(x):
                return y_true + self.rng.normal(0, x_dep_noise)
            return y_true + self.rng.normal(0, x_dep_noise, size=np.shape(x))
            
        if np.isscalar(x):
            return y_true + self.rng.normal(0, noise)
        return y_true + self.rng.normal(0, noise, size=np.shape(x))
        
    def get_validation_set(self, r):
        x = self.val_x[r]
        if self.regions[r]['family'] == 'noisy_unlearnable':
            y = self.measure(r, x)
        else:
            y = self._true_function(r, x)
        return x, y
        
    @property
    def num_regions(self):
        return self.max_regions
