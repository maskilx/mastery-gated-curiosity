import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

class RegionSetActionNet(nn.Module):
    def __init__(self, region_feat_dim, global_feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(region_feat_dim + global_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, region_feats, global_feat):
        B, num_regions, _ = region_feats.shape
        global_feat_exp = global_feat.unsqueeze(1).expand(-1, num_regions, -1)
        combined = torch.cat([region_feats, global_feat_exp], dim=-1)
        logits = self.net(combined)
        return logits.squeeze(-1)

class RegionSetPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        self.num_regions = action_space.n
        self.features_per_region = observation_space.shape[0] // self.num_regions
        
        self.region_encoder = nn.Sequential(
            nn.Linear(self.features_per_region, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.action_net = RegionSetActionNet(64, 128)
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, deterministic=False, action_masks=None):
        latent_pi, latent_vf = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if action_masks is not None:
            distribution.apply_masking(action_masks)
            
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False, action_masks=None):
        latent_pi, _ = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if action_masks is not None:
            distribution.apply_masking(action_masks)
            
        return distribution.get_actions(deterministic=deterministic)
        
    def predict_values(self, obs):
        _, latent_vf = self._get_latent(obs)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs, actions, action_masks=None):
        latent_pi, latent_vf = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if action_masks is not None:
            distribution.apply_masking(action_masks)
            
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def _get_latent(self, obs):
        B = obs.shape[0]
        obs_reshaped = obs.view(B, self.num_regions, self.features_per_region)
        
        region_feats = self.region_encoder(obs_reshaped)
        
        mask = obs_reshaped[:, :, 0:1] # (B, num_regions, 1)
        
        sum_feat = torch.sum(region_feats * mask, dim=1)
        count_active = torch.sum(mask, dim=1).clamp(min=1.0)
        mean_pool = sum_feat / count_active
        
        masked_region_feats = region_feats.masked_fill(mask == 0.0, -1e9)
        max_pool, _ = torch.max(masked_region_feats, dim=1)
        
        global_context = torch.cat([mean_pool, max_pool], dim=-1)
        
        return (region_feats, global_context), global_context

    def _get_action_dist_from_latent(self, latent_pi):
        region_feats, global_context = latent_pi
        mean_actions = self.action_net(region_feats, global_context)
        return self.action_dist.proba_distribution(action_logits=mean_actions)
