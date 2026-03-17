import os
import random
import logging
from logging.handlers import RotatingFileHandler

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

GRID_HEIGHT = 20
GRID_WIDTH = 20

LOG_EVERY_N_EPISODES = 10

# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA         = 0.9    # discount factor
EPSILON_START = 1.0    # initial exploration rate
EPSILON_MIN   = 0.01   # minimum exploration rate
MAX_EPISODES  = 10_000 # maximum episodes to train agent 

LEARNING_RATE     = 1e-3
HIDDEN_DIM        = 128     # neurons per hidden layer
BATCH_SIZE        = 64      # transitions sampled per training step
BUFFER_CAPACITY   = 10_000  # max transitions stored in replay buffer
MIN_BUFFER_SIZE   = 1_000   # don't train until buffer reaches this size
TARGET_SYNC_EVERY = 100     # copy online → target every N steps

STATE_DIM  = 8              # length of feature vector from extract_state()
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

    dx = fx - sx
    dy = fy - sy

    snake_len     = len(game_state["snakeBody"])
    
    # How far into the episode we are
    step_fraction = game_state.get("step", 0) / (GRID_WIDTH * GRID_HEIGHT)

    state = np.array([
        sx / (GRID_WIDTH  - 1),
        sy / (GRID_HEIGHT - 1),
        fx / (GRID_WIDTH  - 1),
        fy / (GRID_HEIGHT - 1),
        dx / (GRID_WIDTH  - 1),
        dy / (GRID_HEIGHT - 1),
        snake_len / (GRID_WIDTH * GRID_HEIGHT),
        min(step_fraction, 1.0),
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
        logger.debug(f"Action: {ACTIONS[idx]} (explore | ε={epsilon:.3f})")
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
        logger.debug(f"Action: {ACTIONS[idx]} (exploit | ε={epsilon:.3f} | Q={q_vals} | probs={probs}| idx={idx})")

    return ACTIONS[idx], idx

# -----------------------------
# Training Step
# -----------------------------

def optimize_model() -> float | None:
    """
    Bellman target:
      y = r                          (if terminal)
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
    logger.info(f"Checkpoint saved → {path} (episode {episode})")

def load_checkpoint(path: str = "models/dqn.pt") -> tuple[int, float]:
    """Returns (episode, epsilon) so training_loop can resume correctly."""
    checkpoint = torch.load(path, map_location=DEVICE)
    online_net.load_state_dict(checkpoint["online_state_dict"])
    target_net.load_state_dict(checkpoint["target_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    logger.info(f"Checkpoint loaded ← {path} (episode {checkpoint['episode']})")
    return checkpoint["episode"], checkpoint["epsilon"]

# -----------------------------
# Training loop
# -----------------------------
def training_loop():
    epsilon  = EPSILON_START
    episode  = 0
    cum_step = 0        # global step counter (for target sync)
    ep_step  = 0        # steps within the current episode
    prev_raw = None     # raw prev state (s)
    state = None        # structured prev state (s)
    action_idx = None   # action (a)

    # Wait until env is up
    wait_for_env()
    logger.info(f"Training started on {DEVICE}")

    while True:

        # Step 1: Send seed (Fruit is constant, snake position is randomized)
        if ep_step == 0:
            fruit = [4, 4]
            # TODO: @Abdullah has bug where snake and fruit spawn on same time and fruit gets displaced 
            snake = [random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)]
            if snake == fruit:
                snake[0] = (snake[0] + 1) % GRID_WIDTH
            send_seed(grid=[GRID_HEIGHT, GRID_WIDTH], snake=snake, fruit=fruit)

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

        logger.debug(f"Reward: {reward:.3f} | done: {done}")

        # Step 4: Store (s, a, r', s', done) in replay buffer
        buffer_push(state, action_idx, reward, next_state, done)

        # Step 5: Train online network
        loss = optimize_model()
        if loss is not None:
            logger.debug(f"Loss: {loss:.4f}")
        if cum_step % TARGET_SYNC_EVERY == 0:
            sync_target_network()

        # Step 6: Check if terminal state and move to next episode
        if done:
            
            # Linear epsilon decay at the end of each episode
            epsilon = max(EPSILON_MIN, EPSILON_START - (EPSILON_START - EPSILON_MIN) * (episode / MAX_EPISODES))
            
            # Next Episode
            episode += 1

            logger.info(
                f"Episode {episode} ended | "
                f"ep_steps: {ep_step} | "
                f"score: {raw_state['score']} | "
                f"epsilon: {epsilon:.4f} | "
                f"buffer: {len(replay_buffer)} | "
                f"loss: {loss:.4f}" if loss else "loss: N/A"
            )

            # reset for next episode 
            ep_step    = 0
            prev_raw   = None
            state      = None
            action_idx = None

            if episode % 500 == 0:
                save_checkpoint(episode, epsilon, f"models/dqn_{episode}.pt")

            if episode >= MAX_EPISODES:
                save_checkpoint(episode, epsilon)
                logger.info(f"Training complete | episodes: {episode} | ε: {epsilon:.4f}")
                break
        
        # Step 6: If non-terminal state then choose next action and send to env
        else:
            action, action_idx = choose_action(next_state, epsilon)   # (a)
            action_queue.put(action)
            # (s') -> (s)
            prev_raw = raw_state
            state    = next_state                          

        state_queue.task_done()