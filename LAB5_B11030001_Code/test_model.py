import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

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

        
def evaluate(args, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.task == 1:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    else:
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    num_actions = env.action_space.n

    if args.task == 1:
        model = FCNDQN(num_actions).to(device)
    else:        
        model = CNNDQN(num_actions).to(device)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(1):
        obs, _ = env.reset(seed=seed + ep)
        state = obs
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, help="task 1 or 2")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    args = parser.parse_args()

    for seed in range(20):
        evaluate(args)
