import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent

def dqn_train(env, agent, n_episodes = 10000, max_t = 1000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995):
    """
        n_episodes : maximum number of training episodes
        max_t : maximum number of timesteps per episode
        eps_start : starting value of epsilon, for epsilon-greedy action selection
        eps_end : minimum value of epsilon
        eps_decay : multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    print("Starting Training...")
    
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break 
                
        scores_window.append(score)       
        scores.append(score)              
        
        eps = max(eps_end, eps_decay * eps) 
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_lunarlander.pth')
            break
            
    return scores

if __name__ == "__main__":
    env = gym.make("LunarLander")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    
    scores = dqn_train(env, agent)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('AI Learning Curve - LunarLander')
    plt.show()