import numpy as np
from .models import EnsembleMLP
from .metrics import RegionTracker
from .hypothesis import HypothesisDiscriminator

class BaseAgent:
    def __init__(self, num_regions, config, seed=42):
        self.num_regions = num_regions
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        self.model = EnsembleMLP(input_dim=num_regions + 1, lr=config.get('learning_rate', 0.01))
        self.tracker = RegionTracker(num_regions, config)
        
        self.memory = []
        self.last_x = {r: None for r in range(num_regions)}
        self.last_preq_error = {r: None for r in range(num_regions)}
        
    def _one_hot(self, region_id):
        vec = np.zeros(self.num_regions)
        vec[region_id] = 1.0
        return vec
        
    def record(self, region_id, x, y):
        if len(self.memory) >= self.config.get('batch_size', 10):
            vec = self._one_hot(region_id)
            X_test = np.array([np.concatenate([vec, [x]])])
            mean_pred, _ = self.model.predict_with_uncertainty(X_test)
            err = (y - mean_pred[0])**2
            self.last_preq_error[region_id] = err
            
        self.memory.append((region_id, x, y))
        self.tracker.record_sample(region_id, x, y)
        self.last_x[region_id] = x
        
    def train(self, env):
        if len(self.memory) < self.config.get('batch_size', 10):
            return
            
        batch_indices = self.rng.choice(len(self.memory), size=self.config.get('batch_size', 10), replace=True)
        batch = [self.memory[i] for i in batch_indices]
        
        X = []
        Y = []
        for r, x, y in batch:
            vec = self._one_hot(r)
            feat = np.concatenate([vec, [x]])
            X.append(feat)
            Y.append(y)
            
        X = np.array(X)
        Y = np.array(Y)
        
        for _ in range(self.config.get('train_epochs_per_step', 5)):
            self.model.train_batch(X, Y)
            
        for r in range(self.num_regions):
            val_x, val_y = env.get_validation_set(r)
            val_X = np.array([np.concatenate([self._one_hot(r), [vx]]) for vx in val_x])
            
            mean_pred, variance = self.model.predict_with_uncertainty(val_X)
            val_error = np.mean((val_y - mean_pred)**2)
            mean_uncertainty = np.mean(variance)
            
            preq_err = self.last_preq_error[r]
            self.tracker.update_metrics(r, val_error, mean_uncertainty, preq_error=preq_err)

    def select_action(self, active_mask=None):
        raise NotImplementedError
        
    def select_x_for_region(self, region_id):
        if self.last_x[region_id] is not None and self.rng.random() < 0.10:
            return self.last_x[region_id]
        else:
            return self.rng.uniform(-1, 1)

class MasteryGatedAgent(BaseAgent):
    def select_action(self, active_mask=None):
        scores = []
        c = self.config
        for r in range(self.num_regions):
            if active_mask is not None and active_mask[r] == 0.0:
                scores.append(-np.inf)
                continue
                
            status = self.tracker.status[r]
            
            lp = self.tracker.learning_progress[r]
            unc = self.tracker.uncertainty[r]
            noise = self.tracker.noise_risk[r]
            
            score = c.get('w_lp', 1.0) * max(0, lp)
            score += c.get('w_uncertainty', 0.3) * unc
            
            if status == 'unexplored':
                score += c.get('w_unexplored', 0.5)
            
            score -= c.get('w_noise', 1.0) * noise
            
            if status == 'mastered':
                score -= c.get('w_mastered', 2.0)
            elif status == 'blocked_noisy':
                score -= c.get('w_blocked', 3.0)
                
            scores.append(score)
            
        scores = np.array(scores)
        temp = c.get('temperature', 0.3)
        
        scores_scaled = scores / temp
        scores_scaled -= np.max(scores_scaled)
        exp_scores = np.exp(scores_scaled)
        probs = exp_scores / np.sum(exp_scores)
        
        if active_mask is not None:
            active_indices = np.where(active_mask == 1.0)[0]
            if len(active_indices) > 0:
                audit_prob = c.get('audit_probability', 0.02)
                probs = (1 - audit_prob) * probs
                for idx in active_indices:
                    probs[idx] += audit_prob * (1.0 / len(active_indices))
        else:
            audit_prob = c.get('audit_probability', 0.02)
            probs = (1 - audit_prob) * probs + audit_prob * (1.0 / self.num_regions)
        
        return self.rng.choice(self.num_regions, p=probs)

class MasteryGatedHypothesisAgent(MasteryGatedAgent):
    def __init__(self, num_regions, config, seed=42):
        super().__init__(num_regions, config, seed)
        self.discriminator = HypothesisDiscriminator()
        
    def select_x_for_region(self, region_id):
        if self.last_x[region_id] is not None and self.rng.random() < 0.10:
            return self.last_x[region_id]
            
        region_mem = [m for m in self.memory if m[0] == region_id]
        if len(region_mem) < 10:
            return self.rng.uniform(-1, 1)
            
        X_train = np.array([m[1] for m in region_mem])
        Y_train = np.array([m[2] for m in region_mem])
        
        candidates = self.rng.uniform(-1, 1, size=20)
        
        preds = self.discriminator.fit_and_predict(X_train, Y_train, candidates)
        if preds is None or len(preds) < 2:
            return self.rng.choice(candidates)
            
        disagreement = np.var(preds, axis=0)
        
        temp = 0.1
        scaled = disagreement / temp
        scaled -= np.max(scaled)
        probs = np.exp(scaled) / np.sum(np.exp(scaled))
        
        return self.rng.choice(candidates, p=probs)

