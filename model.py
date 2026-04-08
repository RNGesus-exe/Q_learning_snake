import os
import random
import logging
from logging.handlers import RotatingFileHandler

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from client import send_seed, wait_for_env
from queues import state_queue, action_queue

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("model")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

file_handler = RotatingFileHandler("logs/model.log", maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# -----------------------------
# Constants
# -----------------------------
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

EMPTY    = 0 
SNAKE    = 1 
FRUIT    = 2 
OBSTACLE = 3

GRID_HEIGHT = 10
GRID_WIDTH = 10

LOG_EVERY_N_EPISODES = 10

# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA         = 0.7    # discount factor
EPSILON_START = 1.0    # initial exploration rate
EPSILON_MIN   = 0.01   # minimum exploration rate
MAX_EPISODES  = 10_000 # maximum episodes to train agent 

LEARNING_RATE     = 1e-3
HIDDEN_DIM        = 128     # neurons per hidden layer
BATCH_SIZE        = 64      # transitions sampled per training step
BUFFER_CAPACITY   = 10_000  # max transitions stored in replay buffer
MIN_BUFFER_SIZE   = 1_000   # don't train until buffer reaches this size
TARGET_SYNC_EVERY = 100     # copy online → target every N steps

STATE_DIM  = 4              # length of feature vector from extract_state()
ACTION_DIM = len(ACTIONS)   # 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Replay Buffer
# -----------------------------
replay_buffer: deque = deque(maxlen=BUFFER_CAPACITY)    # (s, a, r', s', done)

def buffer_push(state, action_idx, reward, next_state, done):
    replay_buffer.append((state, action_idx, reward, next_state, done))

def buffer_sample(batch_size: int):
    """
    Returns a batch of tensors ready for the training step.
    We stack numpy arrays first (fast), then convert to tensors once.
    """
    batch = random.sample(replay_buffer, batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states_t      = torch.tensor(np.stack(states),      dtype=torch.float32).to(DEVICE)
    actions_t     = torch.tensor(actions,               dtype=torch.long   ).to(DEVICE)
    rewards_t     = torch.tensor(rewards,               dtype=torch.float32).to(DEVICE)
    next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32).to(DEVICE)
    dones_t       = torch.tensor(dones,                 dtype=torch.float32).to(DEVICE)

    return states_t, actions_t, rewards_t, next_states_t, dones_t

# -----------------------------
# State Representation
# -----------------------------

def extract_state(game_state: dict) -> np.ndarray:
    head   = game_state["snakeBody"][0]
    sx, sy = head["x"], head["y"]
    fx, fy = game_state["fruitPosition"]["x"], game_state["fruitPosition"]["y"]

    state = np.array([
        sx / (GRID_WIDTH  - 1),
        sy / (GRID_HEIGHT - 1),
        fx / (GRID_WIDTH  - 1),
        fy / (GRID_HEIGHT - 1),
    ], dtype=np.float32)

    logger.debug(f"State vector: {state}")
    return state

# -----------------------------
# Network Definition (Q-Table)
# -----------------------------

def build_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(STATE_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, ACTION_DIM),
    ).to(DEVICE)

online_net = build_network()    # trained every step
target_net = build_network()    # frozen copy, synced every TARGET_SYNC_EVERY steps
optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Action Selection (ε-greedy)
# -----------------------------
def choose_action(state: np.ndarray, epsilon: float) -> tuple[str, int]:
    # Exploration if less than epsilon 
    if random.random() < epsilon:
        idx = random.randrange(ACTION_DIM)
        logger.debug(f"Action: {ACTIONS[idx]} (explore | epsilon={epsilon:.3f})")
    # Exploitation if greater than epsilon
    else:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        # Inference
        with torch.no_grad():
            q_vals = online_net(state_t).cpu().numpy()[0]   # shape: (len(ACTIONS),)
        # Softmax
        exp_q = np.exp(q_vals - np.max(q_vals))
        probs = exp_q / exp_q.sum()
        # Sampling
        idx   = np.random.choice(ACTION_DIM, p=probs)
        logger.debug(f"Action: {ACTIONS[idx]} (exploit | epsilon={epsilon:.3f} | Q={q_vals} | probs={probs}| idx={idx})")

    return ACTIONS[idx], idx

# -----------------------------
# Training Step
# -----------------------------

def optimize_model() -> float | None:
    """
    Bellman target:
      y = r                             (if terminal)
      y = r + gamma * max_a Q̂(s', a)    (if not terminal)   ← uses frozen target_net

    Loss:
      L = MSE(y,  Q(s, a))          ← only for the action actually taken
    """
    
    # If replay buffer not populated yet then return
    if len(replay_buffer) < MIN_BUFFER_SIZE:
        return None

    states, actions, rewards, next_states, dones = buffer_sample(BATCH_SIZE)

    # Q(s, a) — online network, only the column for the action taken
    # gather(1, actions) picks the Q-value of the chosen action for each row
    q_current = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # max Q̂(s', a') — target network, no gradients needed
    with torch.no_grad():
        q_next_max = target_net(next_states).max(dim=1).values

    # y = r + γ * max Q̂(s')  — masked to 0 on terminal transitions
    q_target = rewards + GAMMA * q_next_max * (1.0 - dones)

    loss = nn.functional.mse_loss(q_current, q_target)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping — prevents exploding gradients, common in DQN
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()

# -----------------------------
# Target Network Sync
# -----------------------------
def sync_target_network():
    target_net.load_state_dict(online_net.state_dict())
    logger.debug("Target network synced.")

# -----------------------------
# Checkpoint Save / Load
# -----------------------------
def save_checkpoint(episode: int, epsilon: float, path: str = "models/dqn.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "episode":          episode,
        "epsilon":          epsilon,
        "online_state_dict": online_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
    }, path)
    logger.info(f"Checkpoint saved to {path} (episode {episode})")

def load_checkpoint(path: str = "models/dqn.pt") -> tuple[int, float]:
    """Returns (episode, epsilon) so training_loop can resume correctly."""
    checkpoint = torch.load(path, map_location=DEVICE)
    online_net.load_state_dict(checkpoint["online_state_dict"])
    target_net.load_state_dict(checkpoint["target_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    logger.info(f"Checkpoint loaded ← {path} (episode {checkpoint['episode']})")
    return checkpoint["episode"], checkpoint["epsilon"]

# -----------------------------
# WANDB
# -----------------------------
def init_wandb():
    wandb.init(
        project="snake-dqn",
        config={
            "gamma":             GAMMA,
            "epsilon_start":     EPSILON_START,
            "epsilon_min":       EPSILON_MIN,
            "max_episodes":      MAX_EPISODES,
            "learning_rate":     LEARNING_RATE,
            "hidden_dim":        HIDDEN_DIM,
            "batch_size":        BATCH_SIZE,
            "buffer_capacity":   BUFFER_CAPACITY,
            "min_buffer_size":   MIN_BUFFER_SIZE,
            "target_sync_every": TARGET_SYNC_EVERY,
            "state_dim":         STATE_DIM,
            "grid":              f"{GRID_WIDTH}x{GRID_HEIGHT}",
            "device":            str(DEVICE),
        }
    )

    wandb.watch(online_net, log="all", log_freq=100)

    wandb.define_metric("cum_step")
    wandb.define_metric("step/*",    step_metric="cum_step")
    wandb.define_metric("episode/*", step_metric="episode")

# -----------------------------
# Training loop
# -----------------------------
def training_loop():

    # Initialize wandb for monitoring
    init_wandb()

    epsilon  = EPSILON_START
    episode  = 0
    cum_step = 0        # global step counter (for target sync)
    ep_step  = 0        # steps within the current episode
    prev_raw = None     # raw prev state (s)
    state = None        # structured prev state (s)
    action_idx = None   # action (a)

    # Per-episode accumulators
    ep_losses: list[float] = []
    ep_reward_sum: float   = 0.0

    # Wait until env is up
    wait_for_env()
    logger.info(f"Training started on {DEVICE}")

    while True:

        # Step 1: Send seed (Fruit is constant, snake position is randomized)
        if ep_step == 0:
            send_seed(grid=[GRID_HEIGHT, GRID_WIDTH])

        raw_state: dict = state_queue.get() # Blocking Call (s', r)
        cum_step += 1
        ep_step  += 1

        # Step 2: If no (s) in memory then it's the first step of the an episode
        if prev_raw is None:
            logger.info(f"Episode {episode} - Received new (state, reward) | seed: {raw_state.get('seed', 'N/A')}")
            
            # Get new state (s') 
            state = extract_state(raw_state)

            # Generate action (a) based on policy (ε-greedy)
            action, action_idx = choose_action(state, epsilon)
            action_queue.put(action)

            # new state (s') becomes prev state (s)
            prev_raw = raw_state
            state_queue.task_done()
            continue

        # Step 3: Build transition (s, a, r', s', done)
        next_state = extract_state(raw_state)       # (s')
        reward     = raw_state["reward"]            # (r')
        done       = raw_state["gameOver"]          # done
        ep_reward_sum += reward

        logger.debug(f"Reward: {reward:.3f} | done: {done}")

        # Step 4: Store (s, a, r', s', done) in replay buffer
        buffer_push(state, action_idx, reward, next_state, done)

        # Step 5: Train online network
        loss = optimize_model()
        if loss is not None:
            ep_losses.append(loss)
            wandb.log({
                "step/loss":          loss,
                "step/buffer_size":   len(replay_buffer),
                "step/epsilon":       epsilon,
                "cum_step":           cum_step,
            })
        if cum_step % TARGET_SYNC_EVERY == 0:
            sync_target_network()

        # Step 6: Check if terminal state and move to next episode
        if done:
            
            # Linear epsilon decay at the end of each episode
            epsilon = max(EPSILON_MIN, EPSILON_START - (EPSILON_START - EPSILON_MIN) * (episode / MAX_EPISODES))
            
            # Next Episode
            episode += 1

            # Average loss at the end of each episode
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

            logger.info(
                f"Episode {episode} ended | "
                f"ep_steps: {ep_step} | "
                f"total_eps: {cum_step} |"
                f"score: {raw_state['score']} | "
                f"epsilon: {epsilon:.4f} | "
                f"buffer: {len(replay_buffer)} | "
                f"loss: {loss:.4f}" if loss else "loss: N/A"
            )

            wandb.log({
                "episode/score":        raw_state["score"],
                "episode/steps":        ep_step,
                "episode/reward_sum":   ep_reward_sum,
                "episode/avg_loss":     avg_loss,
                "episode/epsilon":      epsilon,
                "episode/buffer_size":  len(replay_buffer),
                "episode":              episode,
            })

            # reset for next episode 
            ep_step    = 0
            ep_losses     = []
            ep_reward_sum = 0.0
            prev_raw   = None
            state      = None
            action_idx = None

            if episode % 1000 == 0:
                save_checkpoint(episode, epsilon, f"models/dqn_{episode}.pt")

            if episode >= MAX_EPISODES:
                save_checkpoint(episode, epsilon)
                logger.info(f"Training complete | episodes: {episode} | epsilon: {epsilon:.4f}")
                wandb.finish()
                break
        
        # Step 6: If non-terminal state then choose next action and send to env
        else:
            action, action_idx = choose_action(next_state, epsilon)   # (a)
            action_queue.put(action)
            # (s') -> (s)
            prev_raw = raw_state
            state    = next_state                          

        state_queue.task_done()