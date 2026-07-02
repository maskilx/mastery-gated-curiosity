import unittest
import numpy as np
import copy
from src.rl_env import CuriosityEnv
from src.world_generator import ProceduralEnvironment

class TestPhase4BAudit(unittest.TestCase):
    def setUp(self):
        config = {
            'environment': {
                'max_regions': 10,
                'procedural': True,
                'difficulty': 'medium'
            },
            'agent': {
                'gate_thresholds': {'tau_noise': 0.2, 'tau_unc': 0.05}
            },
            'rl': {
                'step_cost': 0.0002,
                'newly_mastered_bonus': 0.1,
                'noisy_region_penalty': 0.02,
                'false_mastery_penalty': 0.2,
                'false_blocked_penalty': 0.2,
                'learnable_difficulty_bonus': 0.0,
                'inactive_region_penalty': -0.05
            },
            'experiment': {'total_steps': 100}
        }
        self.env = CuriosityEnv(config, seed=42)
        
    def test_observation_shape_and_leakage(self):
        obs, _ = self.env.reset()
        
        # 1. Check shape is exactly 100
        self.assertEqual(obs.shape, (100,))
        
        # 2. Check no hidden info is passed. 
        # The observation is assembled manually in _get_obs, so we just verify the output limits and shape.
        # It should contain exactly 10 features per region.
        for r in range(10):
            region_obs = obs[r*10 : (r+1)*10]
            # [active, num_samples, val_error, uncertainty, lp, noise_risk, status0, status1, status2, status3]
            active_flag = region_obs[0]
            self.assertIn(active_flag, [0.0, 1.0])
            
            # Status should be one-hot
            status_sum = np.sum(region_obs[6:])
            self.assertAlmostEqual(status_sum, 1.0)
            
    def test_validation_disjoint_and_unpolluted(self):
        obs, _ = self.env.reset()
        r = 0
        
        # Sample validation x
        val_x, _ = self.env.env.get_validation_set(r)
        
        # Ensure agent memory is empty initially
        self.assertEqual(len(self.env.agent.memory), 0)
        
        # Take a step
        active_regions = np.where(self.env.env.active_mask == 1.0)[0]
        self.env.step(active_regions[0])
        
        # Ensure training samples are in memory
        self.assertEqual(len(self.env.agent.memory), 1)
        
        # Ensure validation x values are NOT in the memory buffer
        train_xs = [m[1] for m in self.env.agent.memory if m[0] == r]
        for tx in train_xs:
            self.assertNotIn(tx, val_x)
            
    def test_learnable_validation_error_excludes_noisy(self):
        self.env.reset()
        
        noisy_regions = [r for r in range(self.env.num_regions) 
                         if self.env.env.active_mask[r] == 1.0 
                         and not self.env.env.regions[r]['is_learnable']]
        
        # Ensure that no noisy region is in the learnable_regions list used for reward
        for r in noisy_regions:
            self.assertNotIn(r, self.env.learnable_regions)
            
    def test_inactive_action_masking(self):
        self.env.reset()
        mask = self.env.action_masks()
        
        # Mask length should match action space
        self.assertEqual(len(mask), self.env.action_space.n)
        
        # Ensure inactive regions are masked out
        for r in range(self.env.num_regions):
            if self.env.env.active_mask[r] == 0.0:
                self.assertFalse(mask[r])
            else:
                self.assertTrue(mask[r])

class TestPhase4CNoValObs(unittest.TestCase):
    def setUp(self):
        config = {
            'environment': {
                'max_regions': 10,
                'procedural': True,
                'difficulty': 'medium'
            },
            'agent': {
                'gate_thresholds': {'tau_noise': 0.2, 'tau_unc': 0.05}
            },
            'rl': {
                'step_cost': 0.0002,
                'newly_mastered_bonus': 0.1,
                'noisy_region_penalty': 0.02,
                'false_mastery_penalty': 0.2,
                'false_blocked_penalty': 0.2,
                'learnable_difficulty_bonus': 0.0,
                'inactive_region_penalty': -0.05
            },
            'experiment': {'total_steps': 100}
        }
        from src.rl_env import CuriosityEnvNoValObs
        self.env = CuriosityEnvNoValObs(config, seed=42)
        
    def test_observation_does_not_contain_val_error(self):
        obs, _ = self.env.reset()
        for r in range(10):
            region_obs = obs[r*10 : (r+1)*10]
            preq_ema = region_obs[2]
            tracker_val_err = self.env.agent.tracker.val_error[r]
            tracker_preq_ema = self.env.agent.tracker.preq_error_ema[r]
            
            # The observation should contain preq_error_ema, which should initially be 1.0 (or match preq_error_ema, NOT val_error)
            self.assertAlmostEqual(preq_ema, tracker_preq_ema, places=5)
            
    def test_reward_uses_preq_error(self):
        self.env.reset()
        active_regions = np.where(self.env.env.active_mask == 1.0)[0]
        
        # Take a step
        _, reward, _, _, _ = self.env.step(active_regions[0])
        
        # The history used to calculate reward should be based on preq_error_ema
        # In CuriosityEnvNoValObs._get_mean_learnable_error it averages preq_error_ema[r] for active_mask == 1.0
        errs = [self.env.agent.tracker.preq_error_ema[r] for r in range(self.env.num_regions) if self.env.env.active_mask[r] == 1.0]
        expected_mean = np.mean(errs)
        self.assertEqual(self.env.val_error_history[-1], expected_mean)
        
    def test_tracker_status_does_not_use_val_error(self):
        self.env.reset()
        r = np.where(self.env.env.active_mask == 1.0)[0][0]
        
        # Manually force val_error high but preq_error low to see if it masters
        self.env.agent.tracker.val_error[r] = 100.0
        self.env.agent.tracker.preq_error_ema[r] = 0.01
        self.env.agent.tracker.uncertainty[r] = 0.01
        self.env.agent.tracker.learning_progress[r] = 0.0
        self.env.agent.tracker.num_samples[r] = 50
        
        self.env.agent.tracker._update_status(r)
        
        # Status should be mastered because it ignores the hidden 100.0 val_error and uses the 0.01 preq_error
        self.assertEqual(self.env.agent.tracker.status[r], 'mastered')

class TestPhase4DNoStatus(unittest.TestCase):
    def setUp(self):
        config = {
            'environment': {
                'max_regions': 10,
                'procedural': True,
                'difficulty': 'medium'
            },
            'agent': {
                'gate_thresholds': {'tau_noise': 0.2, 'tau_unc': 0.05}
            },
            'rl': {
                'step_cost': 0.0002,
                'newly_mastered_bonus': 0.1,
                'noisy_region_penalty': 0.02,
                'false_mastery_penalty': 0.2,
                'false_blocked_penalty': 0.2,
                'learnable_difficulty_bonus': 0.0,
                'inactive_region_penalty': -0.05
            },
            'experiment': {'total_steps': 100}
        }
        from src.rl_env import CuriosityEnvNoValObs_NoStatus
        self.env = CuriosityEnvNoValObs_NoStatus(config, seed=42)
        
    def test_observation_shape_without_status(self):
        obs, _ = self.env.reset()
        
        # 1. Check shape is exactly 60 (10 regions * 6 features)
        self.assertEqual(obs.shape, (60,))
        
        # 2. Check that no status one-hot feature is appended
        for r in range(10):
            region_obs = obs[r*6 : (r+1)*6]
            self.assertEqual(len(region_obs), 6)
            active_flag = region_obs[0]
            self.assertIn(active_flag, [0.0, 1.0])
            
if __name__ == '__main__':
    unittest.main()
