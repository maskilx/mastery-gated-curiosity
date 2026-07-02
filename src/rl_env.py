import gymnasium as gym
import numpy as np
from collections import deque
from .environment import Environment
from .world_generator import ProceduralEnvironment
from .agent import MasteryGatedHypothesisAgent

class CuriosityEnv(gym.Env):
    def __init__(self, config, seed=42):
        super().__init__()
        self.config = config
        self.num_regions = config['environment'].get('max_regions', 10)
        self.seed_val = seed
        
        # Action is selecting a region
        self.action_space = gym.spaces.Discrete(self.num_regions)
        
        # Obs: 10 features per region (active_mask + 9 original)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_regions * 10,), dtype=np.float32
        )
        
        self.status_map = {'unexplored': 0, 'learning': 1, 'mastered': 2, 'blocked_noisy': 3}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = seed
            
        if self.config['environment'].get('procedural', False):
            self.env = ProceduralEnvironment(
                difficulty=self.config['environment'].get('difficulty', 'easy'),
                max_regions=self.num_regions,
                fixed_num_active=self.config['environment'].get('fixed_num_active', None),
                seed=self.seed_val,
                forced_families=self.config['environment'].get('forced_families', None)
            )
        else:
            self.env = Environment(
                num_regions=self.config['environment'].get('num_regions', 5),
                max_regions=self.num_regions,
                noise_learnable=self.config['environment']['noise_learnable'],
                noise_unlearnable=self.config['environment']['noise_unlearnable'],
                seed=self.seed_val
            )
        
        self.agent = MasteryGatedHypothesisAgent(
            num_regions=self.num_regions, 
            config=self.config['agent'], 
            seed=self.seed_val
        )
        
        self.step_count = 0
        self.max_steps = self.config['experiment']['total_steps']
        self.inactive_selections = 0
        
        # For smoothed reward
        self.learnable_regions = [r for r in range(self.num_regions) 
                                  if getattr(self.env, 'regions', None) is None and r != 3 
                                  or (hasattr(self.env, 'regions') and self.env.regions[r]['is_learnable'])]
        self.val_error_history = deque(maxlen=10)
        
        self.mastered_regions = set()
        self.blocked_regions = set()
        
        self._update_val_errors()
        for _ in range(10):
            self.val_error_history.append(self._get_mean_learnable_error())
            
        return self._get_obs(), {}
        
    def action_masks(self):
        return self.env.active_mask.astype(bool)

    def _update_val_errors(self):
        for r in range(self.num_regions):
            val_x, val_y = self.env.get_validation_set(r)
            val_X = np.array([np.concatenate([self.agent._one_hot(r), [vx]]) for vx in val_x])
            mean_pred, variance = self.agent.model.predict_with_uncertainty(val_X)
            val_error = np.mean((val_y - mean_pred)**2)
            mean_uncertainty = np.mean(variance)
            self.agent.tracker.update_metrics(r, val_error, mean_uncertainty)
            
    def _get_mean_learnable_error(self):
        errs = [self.agent.tracker.val_error[r] for r in self.learnable_regions]
        return np.mean(errs)
        
    def _get_obs(self):
        obs = []
        for r in range(self.num_regions):
            t = self.agent.tracker
            status_vec = np.zeros(4)
            status_vec[self.status_map[t.status[r]]] = 1.0
            
            active = self.env.active_mask[r]
            
            feat = [
                float(active),
                float(t.num_samples[r]),
                float(t.val_error[r]),
                float(t.uncertainty[r]),
                float(t.learning_progress[r]),
                float(t.noise_risk[r])
            ]
            obs.extend(feat)
            obs.extend(status_vec)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        region = int(action)
        self.step_count += 1
        rl_conf = self.config.get('rl', {})
        
        # Check if inactive
        if self.env.active_mask[region] == 0.0:
            self.inactive_selections += 1
            reward = rl_conf.get('inactive_region_penalty', -0.05)
            terminated = self.step_count >= self.max_steps
            return self._get_obs(), float(reward), terminated, False, {'inactive': True}
            
        mean_err_before = np.mean(self.val_error_history)
        
        # Take step
        x = self.agent.select_x_for_region(region)
        y = self.env.measure(region, x)
        self.agent.record(region, x, y)
        self.agent.train(self.env)
        
        # Update smoothed history
        self.val_error_history.append(self._get_mean_learnable_error())
        mean_err_after = np.mean(self.val_error_history)
        
        # Base reward
        reward = mean_err_before - mean_err_after
        
        # Penalties/Bonuses
        is_noisy = False
        if hasattr(self.env, 'regions'):
            is_noisy = not self.env.regions[region]['is_learnable']
        else:
            is_noisy = (region == 3)
            
        if is_noisy:
            reward -= rl_conf.get('noisy_region_penalty', 0.02)
            
        reward -= rl_conf.get('step_cost', 0.001)
        
        for r in range(self.num_regions):
            if self.env.active_mask[r] == 0.0:
                continue
                
            status = self.agent.tracker.status[r]
            
            is_r_noisy = False
            if hasattr(self.env, 'regions'):
                is_r_noisy = not self.env.regions[r]['is_learnable']
            else:
                is_r_noisy = (r == 3)
                
            if status == 'mastered' and r not in self.mastered_regions:
                self.mastered_regions.add(r)
                if not is_r_noisy:
                    reward += rl_conf.get('newly_mastered_bonus', 0.05)
                else:
                    reward -= rl_conf.get('false_mastery_penalty', 0.2)
                    
            if status == 'blocked_noisy' and r not in self.blocked_regions:
                self.blocked_regions.add(r)
                if not is_r_noisy:
                    reward -= rl_conf.get('false_blocked_penalty', 0.2)
                    
                    
        bonus_val = rl_conf.get('learnable_difficulty_bonus', 0.0)
        if bonus_val > 0 and region != 3:
            status = self.agent.tracker.status[region]
            noise_risk = self.agent.tracker.noise_risk[region]
            uncertainty = self.agent.tracker.uncertainty[region]
            lp = self.agent.tracker.learning_progress[region]
            tau_noise = self.config['agent'].get('gate_thresholds', {}).get('tau_noise', 0.2)
            tau_unc = self.config['agent'].get('gate_thresholds', {}).get('tau_unc', 0.05)
            
            if status in ['learning', 'unexplored']:
                if noise_risk < tau_noise and (uncertainty > tau_unc or lp >= 0):
                    reward += bonus_val
                    
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        info = {
            'inactive': False,
            'inactive_rate': self.inactive_selections / self.step_count
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info

class CuriosityEnvNoValObs(CuriosityEnv):
    def _get_obs(self):
        obs = []
        for r in range(self.num_regions):
            t = self.agent.tracker
            status_vec = np.zeros(4)
            status_vec[self.status_map[t.status[r]]] = 1.0
            
            active = self.env.active_mask[r]
            
            feat = [
                float(active),
                float(t.num_samples[r]),
                float(t.preq_error_ema[r]), # Using observable prequential error instead of val_error
                float(t.uncertainty[r]),
                float(t.learning_progress[r]),
                float(t.noise_risk[r])
            ]
            obs.extend(feat)
            obs.extend(status_vec)
        return np.array(obs, dtype=np.float32)

    def _get_mean_learnable_error(self):
        # Override to use prequential error EMA instead of true validation error
        # NOTE: self.learnable_regions technically relies on the privileged 'is_learnable' 
        # to calculate the reward. Wait!
        # The user requested: "The reward should be based on observable signals only: 
        # improvement in prequential_error_ema". 
        # But if we use `learnable_regions` to filter the mean, that is privileged!
        # Instead, we should take the mean over ALL ACTIVE regions.
        errs = [self.agent.tracker.preq_error_ema[r] for r in range(self.num_regions) if self.env.active_mask[r] == 1.0]
        if len(errs) == 0:
            return 1.0
        return np.mean(errs)
        
    def step(self, action):
        region = int(action)
        self.step_count += 1
        rl_conf = self.config.get('rl', {})
        
        # Check if inactive
        if self.env.active_mask[region] == 0.0:
            self.inactive_selections += 1
            reward = rl_conf.get('inactive_region_penalty', -0.05)
            terminated = self.step_count >= self.max_steps
            return self._get_obs(), float(reward), terminated, False, {'inactive': True}
            
        mean_err_before = np.mean(self.val_error_history)
        
        # Take step
        x = self.agent.select_x_for_region(region)
        y = self.env.measure(region, x)
        self.agent.record(region, x, y)
        self.agent.train(self.env)
        
        # Update smoothed history (which now uses preq_error_ema across active regions)
        self.val_error_history.append(self._get_mean_learnable_error())
        mean_err_after = np.mean(self.val_error_history)
        
        # Base reward
        reward = mean_err_before - mean_err_after
        
        # Penalties/Bonuses
        is_noisy = False
        # In observable setting, we MUST use the tracker's noise risk instead of the privileged label
        if self.agent.tracker.noise_risk[region] > self.config['agent'].get('gate_thresholds', {}).get('tau_noise', 0.2):
            is_noisy = True
            
        if is_noisy:
            reward -= rl_conf.get('noisy_region_penalty', 0.02)
            
        reward -= rl_conf.get('step_cost', 0.001)
        
        for r in range(self.num_regions):
            if self.env.active_mask[r] == 0.0:
                continue
                
            status = self.agent.tracker.status[r]
            
            is_r_noisy = False
            if self.agent.tracker.noise_risk[r] > self.config['agent'].get('gate_thresholds', {}).get('tau_noise', 0.2):
                is_r_noisy = True
                
            if status == 'mastered' and r not in self.mastered_regions:
                self.mastered_regions.add(r)
                if not is_r_noisy:
                    reward += rl_conf.get('newly_mastered_bonus', 0.05)
                else:
                    reward -= rl_conf.get('false_mastery_penalty', 0.2)
                    
            if status == 'blocked_noisy' and r not in self.blocked_regions:
                self.blocked_regions.add(r)
                if not is_r_noisy:
                    reward -= rl_conf.get('false_blocked_penalty', 0.2)
                    
        bonus_val = rl_conf.get('learnable_difficulty_bonus', 0.0)
        if bonus_val > 0:
            status = self.agent.tracker.status[region]
            noise_risk = self.agent.tracker.noise_risk[region]
            uncertainty = self.agent.tracker.uncertainty[region]
            lp = self.agent.tracker.learning_progress[region]
            tau_noise = self.config['agent'].get('gate_thresholds', {}).get('tau_noise', 0.2)
            tau_unc = self.config['agent'].get('gate_thresholds', {}).get('tau_unc', 0.05)
            
            if status in ['learning', 'unexplored']:
                if noise_risk < tau_noise and (uncertainty > tau_unc or lp >= 0):
                    reward += bonus_val
                    
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        info = {
            'inactive': False,
            'inactive_rate': self.inactive_selections / self.step_count
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info

class CuriosityEnvNoValObs_NoStatus(CuriosityEnvNoValObs):
    def __init__(self, config, seed=42):
        super().__init__(config, seed)
        # Obs: 6 features per region instead of 10 (status 4-dim vector removed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_regions * 6,), dtype=np.float32
        )

    def _get_obs(self):
        obs = []
        for r in range(self.num_regions):
            t = self.agent.tracker
            active = self.env.active_mask[r]
            
            feat = [
                float(active),
                float(t.num_samples[r]),
                float(t.preq_error_ema[r]),
                float(t.uncertainty[r]),
                float(t.learning_progress[r]),
                float(t.noise_risk[r])
            ]
            obs.extend(feat)
        return np.array(obs, dtype=np.float32)
