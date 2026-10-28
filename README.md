# Autonomous Lunar Lander Using Deep Reinforcement Learning

A Deep Q-Network (DQN) agent that learns to land a spacecraft in the [Gymnasium](https://gymnasium.farama.org/) **LunarLander** environment. It is built with Python and PyTorch and learns only from reward feedback. No landing rules are programmed in.

![Project flowchart](LunarLander%20DQN%20Project%20Flowchart.png)

## Table of contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Future work](#future-work)

## How it works

**Environment.** At every step the agent observes 8 numbers: the lander's x/y position, x/y velocity, angle, angular velocity, and whether each leg touches the ground. It then picks one of 4 actions:

| Action | Meaning |
|:------:|---------|
| 0 | Do nothing |
| 1 | Fire left engine |
| 2 | Fire main engine |
| 3 | Fire right engine |

The environment returns a reward (soft, steady landings between the flags earn points, crashes lose points) and the next state.

**Agent.** The agent uses a DQN with these ideas:

- **Q-network:** a neural network that predicts a Q-value (how good an action is) for each of the 4 actions.
- **Epsilon-greedy exploration:** random moves early on, more learned moves later.
- **Replay buffer:** past experiences are stored and sampled randomly in batches, which breaks the link between consecutive frames.
- **Target network:** a second, slowly updated copy of the network that gives steady learning targets (soft update).
- **Double-DQN style target:** the local network chooses the best next action and the target network scores it, which reduces overestimated values.

**Training loop.**

1. Create the environment and the agent (local and target networks, replay buffer).
2. Observe the state and choose an action with epsilon-greedy selection.
3. Step the environment and receive the reward and next state.
4. Store the experience in the replay buffer.
5. Every 4 steps (once the buffer holds more than 64 experiences), sample a batch of 64 and update the local network.
6. Soft-update the target network.
7. Repeat across episodes, reducing epsilon after each one.
8. When the average score of the last 100 episodes reaches **200**, save the weights to `checkpoint_lunarlander.pth` and stop.

**Evaluation.** `watch_agent.py` loads the saved checkpoint and runs the trained agent with no random exploration.

## Project structure

```
LunarLander_DL/
├── train.py                     # Training loop, checkpoint saving, learning-curve plot
├── agent.py                     # Agent: act(), step(), learn(), soft_update()
├── network.py                   # QNetwork (8 -> 128 -> 128 -> 4)
├── memory.py                    # ReplayBuffer
├── watch_agent.py               # Loads the checkpoint and runs the trained agent
├── test_env.py                  # Runs the environment with random actions (setup check)
├── checkpoint_lunarlander.pth   # Saved weights of the trained LunarLander agent
├── checkpoint_cartpole.pth      # Saved CartPole weights (not used by the LunarLander scripts)
├── LunarLander DQN Project Flowchart.png
├── LunarLander_DQN.pptx         # Presentation slides
└── requirements.txt
```

| File | Role |
|------|------|
| `train.py` | Runs up to 10,000 episodes of up to 1000 steps, tracks the 100-episode average, saves the model when solved, and plots the learning curve. |
| `agent.py` | `act()` picks an action, `step()` stores experience and triggers learning every 4 steps, `learn()` updates the network, `soft_update()` blends local weights into the target network. |
| `network.py` | Fully connected network with two hidden layers of 128 units that outputs one Q-value per action. |
| `memory.py` | Stores (state, action, reward, next state, done) and returns random batches. |
| `watch_agent.py` | Renders the trained agent playing one episode (up to 1000 steps) and prints the final score. |
| `test_env.py` | Prints the observation and action spaces and plays random actions for 500 steps. |

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/rika1089/LunarLander_DL.git
cd LunarLander_DL

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` installs `gymnasium[box2d]`, `torch`, `numpy` and `matplotlib`.

## Usage

**1. Check the setup (random agent)**

```bash
python test_env.py
```

**2. Train the agent**

```bash
python train.py
```

Training prints the running average score every episode. When the average of the last 100 episodes reaches 200, it saves `checkpoint_lunarlander.pth`, stops, and shows the learning curve. If the target is not reached within 10,000 episodes, no checkpoint is saved.

**3. Watch the trained agent**

```bash
python watch_agent.py
```

A window opens and the agent plays one episode using only its learned policy. The final score is printed in the terminal. Press `Ctrl+C` to stop early.

## Hyperparameters

| Setting | Value |
|---------|-------|
| Network | 8 inputs, 128, 128, 4 outputs |
| Optimizer / learning rate | Adam / 5e-4 |
| Discount factor (gamma) | 0.99 |
| Replay buffer size | 100,000 |
| Batch size | 64 |
| Learn every | 4 steps |
| Soft update (tau) | 1e-3 |
| Epsilon | starts at 1.0, x0.995 per episode, minimum 0.01 |
| Episode limits | up to 10,000 episodes, 1000 steps each |
| Stopping rule | average of last 100 scores >= 200 |

## Results

The saved checkpoint was tested for 20 episodes using the best action at every step (no exploration), with seeds 0 to 19 on Gymnasium `LunarLander-v3`.

| Metric | Value |
|--------|-------|
| Average score | 214.7 |
| Median score | 228.5 |
| Episodes scoring 200 or more | 15 of 20 |
| Lowest / highest score | 93.1 / 281.8 |

Some episodes score below 200, so results vary from run to run. A reward-versus-episode training curve is produced by `train.py` when you train from scratch; it is not included here.

## Troubleshooting

- **`DeprecatedEnv: LunarLander-v2 is deprecated`**: current Gymnasium has replaced `LunarLander-v2` with `LunarLander-v3`. If `test_env.py` fails with this error, change the environment id in that file to `"LunarLander-v3"`. `train.py` and `watch_agent.py` use the unversioned `"LunarLander"` id, which resolves to the latest version.
- **Box2D install errors**: install the extra with `pip install "gymnasium[box2d]"`. On some systems you may first need `swig` installed.
- **No window appears**: `watch_agent.py` and `test_env.py` use `render_mode="human"`, which needs a display. On a remote or headless machine, use `render_mode="rgb_array"` instead.
- **`torch.load` errors**: `watch_agent.py` loads the checkpoint with `weights_only=True`, which needs a recent PyTorch version. Run `pip install -U torch`.

## Future work

- Make training more stable and tune hyperparameters (learning rate, epsilon decay, tau).
- Evaluate over many more test episodes and report the average.
- Try improvements such as prioritized experience replay or a dueling network.

## Author

[rika1089](https://github.com/rika1089)
