from .agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def select_action(self, active_mask=None):
        if active_mask is not None:
            active_indices = np.where(active_mask == 1.0)[0]
            return self.rng.choice(active_indices)
        return self.rng.choice(self.num_regions)
        
class HighestErrorAgent(BaseAgent):
    def select_action(self, active_mask=None):
        errors = [self.tracker.val_error[r] if (active_mask is None or active_mask[r] == 1.0) else -np.inf 
                  for r in range(self.num_regions)]
        return np.argmax(errors)
        
class UncertaintyOnlyAgent(BaseAgent):
    def select_action(self, active_mask=None):
        uncs = [self.tracker.uncertainty[r] if (active_mask is None or active_mask[r] == 1.0) else -np.inf 
                for r in range(self.num_regions)]
        return np.argmax(uncs)
        
class LearningProgressAgent(BaseAgent):
    def select_action(self, active_mask=None):
        lps = [max(0, self.tracker.learning_progress[r]) if (active_mask is None or active_mask[r] == 1.0) else 0.0 
               for r in range(self.num_regions)]
        
        if np.sum(lps) == 0:
            if active_mask is not None:
                active_indices = np.where(active_mask == 1.0)[0]
                return self.rng.choice(active_indices)
            return self.rng.choice(self.num_regions)
            
        probs = np.array(lps) / np.sum(lps)
        return self.rng.choice(self.num_regions, p=probs)

class OracleRandomLearnableAgent(BaseAgent):
    def __init__(self, num_regions, config, seed=42):
        super().__init__(num_regions, config, seed)
        # We also want hypothesis x-selection so it's a fair baseline for Region Selection
        from .agent import MasteryGatedHypothesisAgent
        self.x_agent = MasteryGatedHypothesisAgent(num_regions, config, seed=seed)
        
    def select_x_for_region(self, region_id):
        self.x_agent.memory = self.memory # synchronize memory
        self.x_agent.last_x = self.last_x
        return self.x_agent.select_x_for_region(region_id)
        
    def select_action(self, active_mask=None, env=None):
        if env is None or active_mask is None:
            return self.rng.choice(self.num_regions)
            
        active_indices = np.where(active_mask == 1.0)[0]
        learnable = [r for r in active_indices if env.regions[r]['is_learnable']]
        
        if len(learnable) == 0:
            return self.rng.choice(active_indices)
            
        return self.rng.choice(learnable)

class OracleBestRegionAgent(BaseAgent):
    def __init__(self, num_regions, config, seed=42):
        super().__init__(num_regions, config, seed)
        from .agent import MasteryGatedHypothesisAgent
        self.base_agent = MasteryGatedHypothesisAgent(num_regions, config, seed=seed)
        
    def select_x_for_region(self, region_id):
        self.base_agent.memory = self.memory
        self.base_agent.last_x = self.last_x
        return self.base_agent.select_x_for_region(region_id)
        
    def select_action(self, active_mask=None, env=None):
        self.base_agent.tracker = self.tracker
        
        if env is None or active_mask is None:
            return self.base_agent.select_action(active_mask)
            
        # Oracle filtering mask
        oracle_mask = np.zeros_like(active_mask)
        for r in range(self.num_regions):
            if active_mask[r] == 1.0 and env.regions[r]['is_learnable']:
                oracle_mask[r] = 1.0
                
        # If no learnable active region, fall back to active mask
        if np.sum(oracle_mask) == 0:
            oracle_mask = active_mask
            
        return self.base_agent.select_action(active_mask=oracle_mask)
