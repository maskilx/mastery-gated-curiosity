import gymnasium as gym
import numpy as np
import torch
from src.audit_drone_world_model import load_model

class BaseDroneWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(2)
        return obs, info

class SurvivalPretrainWrapper(BaseDroneWrapper):
    """
    Rewards the agent just for staying alive.
    +1 for safe step, -10 for crash, -unsafe_penalty for near crash.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        crashed = info.get('crashed', False)
        out_of_bounds = info.get('out_of_bounds', False)
        
        # Penalties
        action_penalty = 0.01 * np.sum(action**2)
        jerk = action - self.prev_action
        jerk_penalty = 0.01 * np.sum(jerk**2)
        
        z = obs[1]
        theta = obs[4]
        unsafe = (z < 0.2) or (abs(theta) > 1.2)
        unsafe_penalty = 0.1 if unsafe else 0.0
        
        if crashed or out_of_bounds:
            reward = -100.0
        else:
            reward = 1.0 - action_penalty - jerk_penalty - unsafe_penalty
            
        self.prev_action = action
        return obs, float(reward), term, trunc, info


class CuriosityPretrainWrapper(BaseDroneWrapper):
    """
    Curiosity-driven exploration reward.
    reward = alpha * clipped_error + beta * novelty - penalties
    """
    def __init__(self, env, world_model_path='results/drone/world_model.pth'):
        super().__init__(env)
        self.world_model = load_model(world_model_path)
        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()
        
        # History for k-NN novelty
        self.history_size = 2000
        self.history = np.zeros((self.history_size, 6))
        self.history_idx = 0
        self.history_count = 0
        
        # Reward coefficients
        self.alpha = 1.0
        self.beta = 0.1
        self.crash_penalty = 100.0
        self.unsafe_penalty = 0.1
        self.action_penalty = 0.01
        self.jerk_penalty = 0.01
        
    def _compute_novelty(self, obs):
        if self.history_count == 0:
            return 1.0
            
        valid_history = self.history[:self.history_count]
        
        # Normalize for distance computation
        norm_obs = (obs - self.world_model.input_mean.numpy()[:6]) / self.world_model.input_std.numpy()[:6]
        norm_hist = (valid_history - self.world_model.input_mean.numpy()[:6]) / self.world_model.input_std.numpy()[:6]
        
        # L2 distance to all
        dists = np.sum((norm_hist - norm_obs)**2, axis=1)
        
        # k-NN distance (k=5)
        k = min(5, len(dists))
        k_dists = np.partition(dists, k-1)[:k]
        mean_knn_dist = np.mean(k_dists)
        
        # Clip/normalize novelty to [0, 1] roughly
        return np.clip(mean_knn_dist, 0.0, 1.0)
        
    def _add_to_history(self, obs):
        self.history[self.history_idx] = obs
        self.history_idx = (self.history_idx + 1) % self.history_size
        self.history_count = min(self.history_count + 1, self.history_size)

    def step(self, action):
        curr_obs = np.copy(self.env.unwrapped.state)
        
        # 1. Prequential prediction (predict before step)
        with torch.no_grad():
            inp = np.concatenate([curr_obs, action])
            inp_t = torch.FloatTensor(inp).unsqueeze(0)
            delta_pred = self.world_model(inp_t).squeeze(0).numpy()
            pred_next_obs = curr_obs + delta_pred
            
        # 2. Environment step
        obs, ext_reward, term, trunc, info = self.env.step(action)
        
        # 3. Compute actual error
        true_delta = obs - curr_obs
        raw_error = np.mean((true_delta - delta_pred)**2)
        
        # Clip error to [0, 1] for reward (assuming typical MSE is much smaller)
        # To make it meaningful, we scale it by a factor (e.g. 1000) so a good error is 0.0 and bad is 1.0
        # If typical error is 0.0001, scaling by 1000 makes it 0.1
        scaled_error = raw_error * 1000.0
        clipped_error = np.clip(scaled_error, 0.0, 1.0)
        
        # 4. Compute Novelty
        novelty = self._compute_novelty(obs)
        self._add_to_history(obs)
        
        # 5. Penalties
        crashed = info.get('crashed', False)
        out_of_bounds = info.get('out_of_bounds', False)
        
        act_pen = self.action_penalty * np.sum(action**2)
        jerk = action - self.prev_action
        jerk_pen = self.jerk_penalty * np.sum(jerk**2)
        
        z = obs[1]
        theta = obs[4]
        unsafe = (z < 0.2) or (abs(theta) > 1.2)
        uns_pen = self.unsafe_penalty if unsafe else 0.0
        
        if crashed or out_of_bounds:
            reward = -self.crash_penalty
        else:
            # Added +1.0 baseline survival to prevent suicide
            reward = 1.0 + self.alpha * clipped_error + self.beta * novelty - uns_pen - act_pen - jerk_pen
            
        # Online world model training
        self.world_model.train()
        self.optimizer.zero_grad()
        inp_t_train = torch.FloatTensor(inp).unsqueeze(0)
        pred_train = self.world_model(inp_t_train)
        target_t = torch.FloatTensor(true_delta).unsqueeze(0)
        loss = self.loss_fn(pred_train, target_t)
        loss.backward()
        self.optimizer.step()
        self.world_model.eval()
        
        self.prev_action = action
        return obs, float(reward), term, trunc, info

class RegimeTracker:
    def __init__(self, min_visits=50, alpha=0.1):
        self.min_visits = min_visits
        self.alpha = alpha  # EMA alpha
        self.regimes = {}
        
    def _discretize_state(self, obs, action):
        z, vz, theta, theta_dot = obs[1], obs[3], obs[4], obs[5]
        
        # State bins (0, 1, 2)
        z_bin = 0 if z < 0.5 else (2 if z > 1.5 else 1)
        vz_bin = 0 if vz < -1.0 else (2 if vz > 1.0 else 1)
        theta_bin = 0 if theta < -0.5 else (2 if theta > 0.5 else 1)
        tdot_bin = 0 if theta_dot < -1.0 else (2 if theta_dot > 1.0 else 1)
        
        # Action mode (0, 1, 2, 3, 4)
        tl, tr = action
        diff = tl - tr
        tot = tl + tr
        if diff < -0.5:
            act_bin = 3 # asymmetric left
        elif diff > 0.5:
            act_bin = 4 # asymmetric right
        elif tot < -0.5:
            act_bin = 0 # balanced low
        elif tot > 0.5:
            act_bin = 2 # balanced high
        else:
            act_bin = 1 # balanced hover
            
        return (z_bin, vz_bin, theta_bin, tdot_bin, act_bin)
        
    def update(self, obs, action, error, unsafe):
        key = self._discretize_state(obs, action)
        
        if key not in self.regimes:
            self.regimes[key] = {
                'visit_count': 0,
                'error_ema': error,
                'lp_ema': 0.0,
                'unsafe_count': 0,
                'status': 'unexplored'
            }
            
        regime = self.regimes[key]
        regime['visit_count'] += 1
        
        if unsafe:
            regime['unsafe_count'] += 1
            
        old_error = regime['error_ema']
        new_error = old_error * (1 - self.alpha) + error * self.alpha
        regime['error_ema'] = new_error
        
        # Learning progress is reduction in error
        lp = old_error - new_error
        regime['lp_ema'] = regime['lp_ema'] * (1 - self.alpha) + lp * self.alpha
        
        # Update status
        if regime['visit_count'] >= self.min_visits:
            unsafe_rate = regime['unsafe_count'] / regime['visit_count']
            if unsafe_rate > 0.5:
                regime['status'] = 'blocked_unsafe'
            elif regime['error_ema'] < 0.05 and abs(regime['lp_ema']) < 0.001:
                regime['status'] = 'mastered'
            elif regime['error_ema'] > 0.1 and regime['lp_ema'] <= 0.0:
                regime['status'] = 'blocked_no_progress'
            else:
                regime['status'] = 'learning'
        else:
            regime['status'] = 'unexplored'
            
        return regime

class SafetyGatedMasteryCuriosityWrapper(CuriosityPretrainWrapper):
    def __init__(self, env, world_model_path='results/drone/world_model.pth'):
        super().__init__(env, world_model_path)
        self.tracker = RegimeTracker(min_visits=50)
        self.gamma = 1.0 # LP bonus
        self.survival_bonus = 0.1
        self.crash_penalty = 10.0
        self.unsafe_penalty = 1.0
        
    def step(self, action):
        curr_obs = np.copy(self.env.unwrapped.state)
        
        # 1. Prequential prediction
        with torch.no_grad():
            inp = np.concatenate([curr_obs, action])
            inp_t = torch.FloatTensor(inp).unsqueeze(0)
            delta_pred = self.world_model(inp_t).squeeze(0).numpy()
            pred_next_obs = curr_obs + delta_pred
            
        # 2. Env step
        obs, ext_reward, term, trunc, info = self.env.step(action)
        
        # 3. Error
        true_delta = obs - curr_obs
        raw_error = np.mean((true_delta - delta_pred)**2)
        clipped_error = np.clip(raw_error * 1000.0, 0.0, 1.0)
        
        # 4. Novelty
        novelty = self._compute_novelty(obs)
        self._add_to_history(obs)
        
        # 5. Penalties and safety
        crashed = info.get('crashed', False)
        out_of_bounds = info.get('out_of_bounds', False)
        
        act_pen = self.action_penalty * np.sum(action**2)
        jerk = action - self.prev_action
        jerk_pen = self.jerk_penalty * np.sum(jerk**2)
        
        z = obs[1]
        theta = obs[4]
        unsafe = (z < 0.2) or (abs(theta) > 1.2)
        
        # Update tracker
        regime = self.tracker.update(obs, action, clipped_error, unsafe or crashed)
        
        if crashed or out_of_bounds:
            reward = -self.crash_penalty
        else:
            uns_pen = self.unsafe_penalty if unsafe else 0.0
            
            # Gating logic
            status = regime['status']
            if status in ['learning', 'unexplored'] and not unsafe:
                gated_curiosity = self.alpha * clipped_error
                lp_bonus = self.gamma * max(0.0, regime['lp_ema'])
            else:
                gated_curiosity = 0.0
                lp_bonus = 0.0
                
            novelty_bonus = self.beta * novelty if not unsafe else 0.0
            
            reward = self.survival_bonus + gated_curiosity + novelty_bonus + lp_bonus - uns_pen - act_pen - jerk_pen
            
        # Online world model training
        self.world_model.train()
        self.optimizer.zero_grad()
        inp_t_train = torch.FloatTensor(inp).unsqueeze(0)
        pred_train = self.world_model(inp_t_train)
        target_t = torch.FloatTensor(true_delta).unsqueeze(0)
        loss = self.loss_fn(pred_train, target_t)
        loss.backward()
        self.optimizer.step()
        self.world_model.eval()
        
        # Log info for callbacks/analysis
        info['regime_status'] = regime['status']
        info['gated_curiosity'] = gated_curiosity if not crashed else 0.0
        info['lp_bonus'] = lp_bonus if not crashed else 0.0
        info['novelty_bonus'] = novelty_bonus if not crashed else 0.0
        
        self.prev_action = action
        return obs, float(reward), term, trunc, info

class SurvivalCuriosityPretrainWrapper(CuriosityPretrainWrapper):
    """
    Survival + Curiosity without Mastery/Blocking.
    """
    def __init__(self, env, world_model_path='results/drone/world_model.pth'):
        super().__init__(env, world_model_path)
        self.survival_bonus = 0.1
        self.crash_penalty = 10.0
        self.unsafe_penalty = 1.0
        
    def step(self, action):
        curr_obs = np.copy(self.env.unwrapped.state)
        with torch.no_grad():
            inp = np.concatenate([curr_obs, action])
            inp_t = torch.FloatTensor(inp).unsqueeze(0)
            delta_pred = self.world_model(inp_t).squeeze(0).numpy()
            
        obs, ext_reward, term, trunc, info = self.env.step(action)
        
        true_delta = obs - curr_obs
        raw_error = np.mean((true_delta - delta_pred)**2)
        clipped_error = np.clip(raw_error * 1000.0, 0.0, 1.0)
        
        novelty = self._compute_novelty(obs)
        self._add_to_history(obs)
        
        crashed = info.get('crashed', False)
        out_of_bounds = info.get('out_of_bounds', False)
        act_pen = self.action_penalty * np.sum(action**2)
        jerk = action - self.prev_action
        jerk_pen = self.jerk_penalty * np.sum(jerk**2)
        z = obs[1]
        theta = obs[4]
        unsafe = (z < 0.2) or (abs(theta) > 1.2)
        
        if crashed or out_of_bounds:
            reward = -self.crash_penalty
        else:
            uns_pen = self.unsafe_penalty if unsafe else 0.0
            curiosity = self.alpha * clipped_error + self.beta * novelty
            reward = self.survival_bonus + curiosity - uns_pen - act_pen - jerk_pen
            
        self.world_model.train()
        self.optimizer.zero_grad()
        inp_t_train = torch.FloatTensor(inp).unsqueeze(0)
        pred_train = self.world_model(inp_t_train)
        target_t = torch.FloatTensor(true_delta).unsqueeze(0)
        loss = self.loss_fn(pred_train, target_t)
        loss.backward()
        self.optimizer.step()
        self.world_model.eval()
        
        self.prev_action = action
        return obs, float(reward), term, trunc, info
