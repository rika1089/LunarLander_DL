import gymnasium as gym
import time

env = gym.make("LunarLander-v2", render_mode="human")

state, info = env.reset()
print("---Environment Info---")
print(f"Observation Space : {env.observation_space.shape}")
print(f"Action Space : {env.action_space}")
print("-----------------------\n")

for step in range(500):
    random_action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(random_action)
    time.sleep(0.05)

    if terminated or truncated:
        print(f"Game Over at step {step}! Restarting...")
        env.reset()

env.close()