import torch
from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        critic_loss = 0
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
          critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)  

        # advantage = estimate_advantage(...)
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        #  for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        actor_loss = 0
        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
          actor_loss = self.actor.update(ob_no, ac_na, adv_n=advantage)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        v_s = self.critic.forward(ob_no)
        # 2) query the critic with next_ob_no, to get V(s')
        v_sp = self.critic.forward(next_ob_no)
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        not_terminal_n = torch.as_tensor(1 - terminal_n)
        q = re_n + self.gamma * v_sp * not_terminal_n 
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        adv_n = q - v_s
        adv_n = ptu.to_numpy(adv_n)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
