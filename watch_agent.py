import gymnasium as gym
import torch
import time
from agent import Agent

env = gym.make("LunarLander", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint_lunarlander.pth', weights_only=True))

state, info = env.reset()
score = 0

print("Watch it go! (Press Ctrl+C in terminal to stop early)")

for j in range(1000): 
    action = agent.act(state, eps=0.0) 
    
    state, reward, terminated, truncated, info = env.step(action)
    score += reward
    
    time.sleep(0.02)
    
    if terminated or truncated:
        break

print(f"Final Score: {score}")
env.close()