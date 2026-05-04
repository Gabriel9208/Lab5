# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class FCNDQN(nn.Module):
    def __init__(self, num_actions):
        super(FCNDQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########

        # B11030001
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )       
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)

class CNNDQN(nn.Module):
    def __init__(self, num_actions):
        super(CNNDQN, self).__init__()
       
        # B11030001
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # 64 x 7 x 7 = 3136
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )       

    def forward(self, x):
        return self.network(x)


class DQNLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
    def forward(self, rewards, q_values, gamma_multiplier, weights=None):
        loss = (rewards + self.gamma * gamma_multiplier - q_values) ** 2
        loss *= 0.5

        if weights is not None:
            loss = loss * weights

        loss = torch.mean(loss, dim=0, keepdim=False)

        return loss


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# B11030001
class SumTree():
    def __init__(self, capacity):
        self.tree = np.zeros((2 * capacity - 1, ), dtype=np.float32)
        self.capacity = capacity

    def _propagate(self, index, value):
        while index > 0:
            index = (index - 1) // 2
            self.tree[index] += value

    def add(self, idx, value):
        if not (0 <= idx < self.capacity):
            raise ValueError(f"The idx {idx} is out of range [0, {self.capacity - 1}]")
        idx += self.capacity - 1
        dummy = self.tree[idx]
        self.tree[idx] = value

        self._propagate(idx, value - dummy)

    def update(self, idx, priority):
        self.add(idx, priority)

    def sample(self, value):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left >= len(self.tree):
                break
            if self.tree[left] >= value:
                idx = left
            else:
                idx = right
                value -= self.tree[left]
        return idx - (self.capacity - 1), self.tree[idx]

    def get_total_priority(self):
        return self.tree[0]   

    def get_max_priority(self):
        return np.max(self.tree[self.capacity - 1:])


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_steps=3000000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_steps = beta_steps
        self.buffer = []

        # B11030001
        self.priorities = SumTree(capacity)
        self.max_priority = 1.0
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) < self.pos + 1:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities.add(self.pos, self.max_priority)
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ########## 
        return  

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        value = np.random.uniform(0, 
                                  self.priorities.get_total_priority(),
                                  size=batch_size)
        indices = []
        priorities = []
        n = len(self.buffer)
        for v in value:
            idx, priority = self.priorities.sample(v)
            if idx >= n or idx < 0:
                idx = np.random.randint(0, n)
                priority = self.priorities.tree[idx + self.capacity - 1]
            indices.append(idx)
            priorities.append(priority)
        data = [self.buffer[i] for i in indices]
        priorities = np.array(priorities)
        
        ########## END OF YOUR CODE (for Task 3) ########## 
        return indices, data, priorities
        
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-7)**self.alpha
            self.priorities.update(idx, priority)
            if priority > self.max_priority:
                self.max_priority = priority
        ########## END OF YOUR CODE (for Task 3) ########## 
        return

    def update_beta(self, train_count):
        self.beta = min(1.0, 0.4 + (1.0 - 0.4) * train_count / self.beta_steps)
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        # B11030001
        self.task = args.task

        if self.task == 1:
            env_name = "CartPole-v1"
        else:
            env_name = "ALE/Pong-v5"

        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # B11030001
        if not self.task == 1:
            self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = FCNDQN(self.num_actions).to(self.device) if self.task == 1 \
            else CNNDQN(self.num_actions).to(self.device)

        
        self.q_net.apply(init_weights)
        
        self.target_net = FCNDQN(self.num_actions).to(self.device) if self.task == 1 \
            else CNNDQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.dqn_type = args.dqn_type
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.base_gamma = args.discount_factor
        self.gamma = args.discount_factor ** args.msre_step if args.msre else args.discount_factor
        self.loss = DQNLoss(self.gamma)

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0 if self.task == 1 else -21  # 0 for CartPole, -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # B11030001
        self.per = args.per
        self.beta_steps = args.beta_steps
        if self.per:
            self.memory = PrioritizedReplayBuffer(args.memory_size, beta_steps=self.beta_steps)
        else:
            self.memory = deque(maxlen=args.memory_size)

        self.wandb_run_name = args.wandb_run_name

        # B11030001
        self.msre = args.msre
        if self.msre:
            self.msre_steps = args.msre_step
            self.msre_buffer = deque(maxlen=self.msre_steps)
        

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # B11030001
        if self.task != 1:
            state_tensor = torch.from_numpy(np.array(state))\
                            .float().unsqueeze(0).to(self.device) / 255.0
        else:
            state_tensor = torch.from_numpy(np.array(state))\
                            .float().unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def _store_transition(self, transition):
        s, a, r, ns, d = transition
        
        if self.per:
            self.memory.add((s, a, r, ns, d))
        else:
            self.memory.append((s, a, r, ns, d))

    def _multi_step_return(self):
        accumulated_reward = 0
        s_first, a_first, r_first, ns_first, d_first = self.msre_buffer.popleft()
        accumulated_reward += r_first
        next_state_final = ns_first
        done_final = d_first

        for i, (_, _, r, ns, d) in enumerate(self.msre_buffer):
            accumulated_reward += (self.base_gamma ** (i + 1)) *r
            next_state_final = ns
            done_final = d
            
        self._store_transition((s_first, 
                                a_first, 
                                accumulated_reward, 
                                next_state_final, 
                                done_final))


    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            # B11030001
            if self.task != 1:
                state = self.preprocessor.reset(obs)
            else:
                state = obs

            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # B11030001
                if self.task != 1:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs

                # B11030001
                if self.msre: 
                    self.msre_buffer.append((state, action, reward, next_state, done))

                    if len(self.msre_buffer) == self.msre_steps or done:
                        self._multi_step_return()

                        if done:
                            while len(self.msre_buffer) > 0:
                                self._multi_step_return()

                else:
                    self._store_transition((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"{self.wandb_run_name}_model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if self.task == 3 and self.env_count in (600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000):
                model_path = os.path.join(self.save_dir, f"{self.wandb_run_name}_model_{self.env_count}steps.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")
                
            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, f"{self.wandb_run_name}_best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()

        # B11030001
        if self.task != 1:
            state = self.preprocessor.reset(obs)
        else:
            state = obs

        done = False
        total_reward = 0

        while not done:

            #B11030001
            if self.task != 1:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device) / 255.0
            else:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward

            # B11030001
            if self.task != 1:
                state = self.preprocessor.step(next_obs)
            else:
                state = next_obs

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer

        # B11030001
        if self.per:
            indices, mini_batch, priorities = self.memory.sample(self.batch_size)
            priorities = torch.from_numpy(priorities).float().to(self.device) /\
                         self.memory.priorities.get_total_priority()
            weights = torch.pow(len(self.memory) * priorities, -self.memory.beta) 
            weights = weights / weights.max()
        else:
            mini_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        # B11030001
        if self.task != 1:
            states = torch.from_numpy(np.array(states)).float().to(self.device) / 255.0
            next_states = torch.from_numpy(np.array(next_states)).float().to(self.device) / 255.0
        else:
            states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
            next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 

        # B11030001
        with torch.no_grad():
            if self.dqn_type == "DQN":
                target_q_values = self.target_net(next_states).max(dim=1)[0]
                gamma_multiplier = target_q_values * (1 - dones)
            else: # DDQN
                target_actions = self.q_net(next_states).argmax(dim=1)
                gamma_multiplier = self.target_net(next_states)\
                    .gather(1, target_actions.unsqueeze(1))\
                    .squeeze(1) * (1 - dones)            
        
        if self.per: # PER
            errors_td = rewards + self.gamma * gamma_multiplier - q_values
            errors = errors_td.abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, errors)
            loss = self.loss(rewards, q_values, gamma_multiplier, weights)
        else:
            loss = self.loss(rewards, q_values, gamma_multiplier)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.per:
            self.memory.update_beta(self.train_count)
        
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)

    # B11030001
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--dqn-type",type=str, default="DQN") # DQN/DDQN
    parser.add_argument('--per', action='store_true') # Prioritized Experience Replay
    parser.add_argument('--beta-steps', type=int, default=3000000) # Dueling DQN
    parser.add_argument('--msre', action='store_true') # Multi Step Returns
    parser.add_argument('--msre-step', type=int, default=1) # Multi Step Returns
    
    

    args = parser.parse_args()

    # B11030001 add config=args
    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, config=args, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(args.episodes)