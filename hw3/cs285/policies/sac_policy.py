from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        if sample:
          # Return the action that the policy prescribes
          sampled_action = self.forward(ptu.from_numpy(observation)).sample()
          return ptu.to_numpy(sampled_action)

        return self.forward(ptu.from_numpy(observation)).mean() 

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        mean = torch.mean(observation)
        std = torch.std(observation) 
        action_distribution = sac_utils.SquashedNormal(mean, std)        
        return action_distribution

    def update(self, obs, critic):
        obs = ptu.from_numpy(obs)
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        action_distribution = self.forward(obs)
        action = action_distribution.sample()
        log_probs = action_distribution.log_probs(action)

        alpha_loss = -self.alpha()*log_probs - self.alpha()*self.target_entropy
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # From Eq 3 in paper: E(Q - log(pi(a|s)))
        q1, q2 = critic(obs, action)
        actor_loss = -torch.mean((torch.minimum(q1, q2) - log_probs))
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        return actor_loss, alpha_loss, self.alpha()
