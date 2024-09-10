import torch
import torch.nn as nn
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple  
# import torch.distributions as D
from torch.distributions import Categorical
import numpy as np
import sys
from debug_side_channel import DebugSideChannel
from torch.distributions import Normal
import torch.nn.functional as F

TRAINING_STEPS = 10000
OBSERVATION_SIZE = 6
ACTION_SIZE = 5
HIDDEN_SIZE = 128
LEARNING_RATE = .001
UPDATE_PERIOD = 100
AGENT_ID_A = 0
AGENT_ID_B = 1
NO_AGENTS = 2
BUFFER_SIZE = 16

Unity_Environment = "C:\\Users\\rmarr\\Documents\\ml-agents-dodgeball-env-ICT"
NO_SIMULATIONS = 1
TIME_SCALE = 2.0

GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
# state
# 10 x 10 grid 
# 0 = nothing 1 = wall 2 = agent position
#debug_side_channel = DebugSideChannel()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_size, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)

        self.log_std = nn.Linear(hidden_size, action_dim)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            return [self.forward_single(s) for s in state]
        return self.forward_single(state)
    
    def forward_single(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def select_action(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            return [self.select_action_single(s) for s in state]
        return self.select_action_single(state)

    def select_action_single(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mu + std*z)

        action = (action + 1) * (ACTION_SIZE/2)
        action = torch.round(action)
        action = torch.clamp(action, 0, ACTION_SIZE-1)
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        if isinstance(state, tuple) or isinstance(state, list):
            return [self.forward_single(s, a) for s, a in zip(state, action)]
        return self.forward_single(state, action)

    def forward_single(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q[0]


class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.size = 0

    def add(self, experience):
        if self.size < BUFFER_SIZE:
            self.buffer.append(experience)
        else:
            self.buffer[self.size % BUFFER_SIZE] = experience
        self.size += 1

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (states, actions, rewards, next_states, dones)

class SACAgent(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size):
        super().__init__()
        self.critic1 = Critic(observation_size, action_size, hidden_size)
        self.critic2 = Critic(observation_size, action_size, hidden_size)
        self.target_critic1 = Critic(observation_size, action_size, hidden_size)
        self.target_critic2 = Critic(observation_size, action_size, hidden_size)
        self.actor = Actor(observation_size, 1, action_size, hidden_size)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LEARNING_RATE)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LEARNING_RATE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)



class Driver():
    def __init__(self):
        self.agent_registry = []
        self.agent_registry.append(SACAgent(observation_size=OBSERVATION_SIZE, action_size=1, hidden_size=HIDDEN_SIZE))
        self.agent_registry.append(SACAgent(observation_size=OBSERVATION_SIZE, action_size=1, hidden_size=HIDDEN_SIZE))
        self.behavior_registry = []
        self.replay_buffer_resistry = []
        self.replay_buffer_resistry.append(ReplayBuffer())
        self.replay_buffer_resistry.append(ReplayBuffer())
        self.engine_channel = EngineConfigurationChannel()
        #self.env = UnityEnvironment(file_name=Unity_Environment, side_channels=[self.engine_channel, debug_side_channel])
        self.env = UnityEnvironment(file_name=Unity_Environment, side_channels=[self.engine_channel])
        self.env.reset()
        self.engine_channel.set_configuration_parameters(time_scale=TIME_SCALE)
        self.behavior_registry.append(list(self.env.behavior_specs.keys())[AGENT_ID_A])
        self.behavior_registry.append(list(self.env.behavior_specs.keys())[AGENT_ID_B])
        self.score_board = {key: 0 for key in range(NO_AGENTS)}

    # TODO need positive reward for success 
    def transition_tuple(self, agent_id):
        behavior_name = self.behavior_registry[agent_id]
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        reward = torch.tensor([0])
        if len(terminal_steps.reward) > 0:
            if terminal_steps.reward[0] == -1.0:
                self.score_board[(1-agent_id)] = self.score_board[(1-agent_id)] + 1
                sys.stdout.write("\r" + f"Agent A - Red: {self.score_board[0]} | Agent B - Blue: {self.score_board[1]}")
                sys.stdout.flush()
                reward = torch.tensor([-1])
        state = decision_steps.obs[0]

        done = len(terminal_steps) > 0
        if done:
            self.env.reset()
        s = torch.tensor(state, dtype=torch.float32)

        agent = self.agent_registry[agent_id]
        # ACTION
        a = agent.actor.select_action(s)[0]
        action = ActionTuple()
        action.add_discrete(a.detach().cpu().numpy()[0].reshape(1, 1))
        self.env.set_actions(behavior_name, action)
        self.env.step()

        if not done:
            decision_steps, terminal_steps = self.env.get_steps(behavior_name)
            next_state = decision_steps.obs[0]
            s_p = torch.tensor(next_state, dtype=torch.float32)#.cpu().numpy()
        else:
            s_p = None

        return (s, a, reward, s_p, done)


    def train(self, agent_id):
        states, actions, rewards, next_states, dones = self.replay_buffer_resistry[agent_id].sample(BUFFER_SIZE)
        # Q Loss
        with torch.no_grad():
            # ACTION
            next_actions = self.agent_registry[agent_id].actor.select_action(next_states)[0]
            target_q1_values = self.agent_registry[agent_id].target_critic1(next_states, next_actions)
            target_q2_values = self.agent_registry[agent_id].target_critic2(next_states, next_actions)
            target_q_values = np.array(list(rewards)) + GAMMA * (1 - np.array(list(dones))) * np.minimum(np.array(target_q1_values), np.array(target_q2_values))

        current_q1_values = self.agent_registry[agent_id].critic1(states, actions)
        current_q1_values = torch.stack([tensor[0] for tensor in current_q1_values])
        current_q2_values = self.agent_registry[agent_id].critic2(states, actions)
        current_q2_values = torch.stack([tensor[0] for tensor in current_q2_values])
        target_q_values = torch.stack([tensor for tensor in target_q_values])

        critic1_loss = nn.MSELoss()(current_q1_values, target_q_values)
        critic2_loss = nn.MSELoss()(current_q2_values, target_q_values)

        self.agent_registry[agent_id].critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.agent_registry[agent_id].critic1_optimizer.step()

        self.agent_registry[agent_id].critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.agent_registry[agent_id].critic2_optimizer.step()

        # Action Loss
        actions, log_probs = self.actor.evaluate(states)
        log_probs_total = log_probs.sum(-1, keepdim=True)
        q1_values = self.critic1(states, actions)
        q2_values = self.critic2(states, actions)
        q_values = torch.min(q1_values, q2_values)        
        actor_loss = (self.alpha * log_probs - q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Transfer params
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def main(self):
        #print(f'behavior_name {behavior_name}')
        #spec = env.behavior_specs[behavior_name]
        #print(f'spec {spec}')
        #self.env.reset()
        # debug_messages = debug_side_channel.get_and_clear_messages()
        # print(debug_messages)
        # for message in debug_messages:
        #     print(f"Received from Unity: {message}")

        # TODO one optimizer or 2?

        #self.env.reset()
        for episode in range(TRAINING_STEPS):
            # Agent actions
            [self.replay_buffer_resistry[agent_id].add(self.transition_tuple(agent_id)) for agent_id in range(NO_AGENTS)]
            # done = any([self.replay_buffer_resistry[agent_id].done() for agent_id in range(NO_AGENTS)])
            # if done:
            #     self.env.reset()
            if (episode + 1) % BUFFER_SIZE == 0:
                [self.train(agent_id) for agent_id in range(NO_AGENTS)]

        self.env.close()
        

if __name__ == "__main__":
    driver = Driver()
    driver.main()