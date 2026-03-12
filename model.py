import queue
import random
import logging
from logging.handlers import RotatingFileHandler

import numpy as np

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    handlers=[
        RotatingFileHandler("model.log", maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# -----------------------------
# Shared queues
# state_queue  : game state dicts pushed by server.py, consumed by training loop
# action_queue : actions pushed by training loop, consumed by client.py
# -----------------------------
state_queue:  queue.Queue[dict] = queue.Queue()
action_queue: queue.Queue[str]  = queue.Queue()


# -----------------------------
# Constants
# -----------------------------
ACTIONS = ["up", "down", "left", "right"]

EMPTY = 0 
SNAKE = 1 
FRUIT = 2 
OBSTACLE = 3

# -----------------------------
# Hyperparameters
# -----------------------------
ALPHA         = 0.1    # learning rate
GAMMA         = 0.9    # discount factor
EPSILON_START = 1.0    # initial exploration rate
EPSILON_MIN   = 0.01   # minimum exploration rate
EPSILON_DECAY = 0.995  # multiplicative decay per episode
MAX_EPISODES  = 10_000 # maximum episodes to train agent 

# Rewards (kept as reference, actual rewards come from env)
# Reward = Normalize(MAX_EUC_DIST-(euc_dist(x)) + Fruit(+10)  + Collision(-10))

# -----------------------------
# Q-Table
# Keyed by (snake_x, snake_y, fruit_x, fruit_y) → np.array of 4 Q-values
# -----------------------------
q_table: dict[tuple, np.ndarray] = {}

# Returns Q-values for a state, initialising to zeros if unseen
def get_q_value(state: tuple) -> np.ndarray:
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
    return q_table[state]

# TODO: ENV will send snake and fruit coords instead of searching for here
def extract_state(game_state: dict) -> tuple:
    """
    Extracts a compact, hashable state tuple from the raw game state.
    Scans the grid to find snake head (1) and fruit (2) positions.

    State: (snake_x, snake_y, fruit_x, fruit_y)
    """
    grid = game_state["state"]

    snake_x = snake_y = fruit_x = fruit_y = None

    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell == SNAKE:
                snake_x, snake_y = col_idx, row_idx
            elif cell == FRUIT:
                fruit_x, fruit_y = col_idx, row_idx

    if snake_x is None:
        raise ValueError("Snake not found in grid")
    if fruit_x is None:
        raise ValueError("Fruit not found in grid")

    state = (snake_x, snake_y, fruit_x, fruit_y)

    logger.debug(f"State — snake: ({snake_x},{snake_y}) | fruit: ({fruit_x},{fruit_y})")

    return state


# ε - greedy policy
def choose_action(state: tuple, epsilon: float) -> str:
    
    # Exploration if less than epsilon 
    if random.random() < epsilon:
        action = random.choice(ACTIONS)
        logger.debug(f"Action: {action} (explore | ε={epsilon:.3f})")
    # Exploitation if greater than expsilon
    else:
        q_vals = get_q_value(state)
        action = ACTIONS[int(np.argmax(q_vals))]
        logger.debug(f"Action: {action} (exploit | ε={epsilon:.3f} | Q={q_vals})")

    return action


def update_q(state: tuple, action: str, reward: float, next_state: tuple):
    """
    Bellman equation update.
    Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
    """
    a_idx     = ACTIONS.index(action)   # idx(a)
    q_current = get_q_value(state)      # s
    q_next    = get_q_value(next_state) # s'

    # Q'
    q_current[a_idx] = q_current[a_idx] + ALPHA * (reward + GAMMA * np.max(q_next) - q_current[a_idx])

    logger.debug(
        f"Q-update — state: {state} | action: {action} | "
        f"reward: {reward:.2f} | new Q: {q_current}"
    )


# -----------------------------
# Training loop (main thread)
# -----------------------------
def training_loop():
    epsilon  = EPSILON_START
    episode  = 0
    step     = 0
    prev_raw = None # prev state (s)

    logger.info("Training loop started — waiting for first game state...")

    # Step 1: Send seed
    # send_seed()

    while True:

        raw_state: dict = state_queue.get() # Blocking Call (s', r)

        # Step 2: If no (s) in memory then it's the first step of the an episode
        if prev_raw is None:
            logger.info(f"Episode {episode} — Received new (state, reward) | seed: {raw_state.get('seed', 'N/A')}")
            
            # Get new state (s') 
            state = extract_state(raw_state)

            # Get action (a') based on policy (ε-greedy)
            action = choose_action(state, epsilon)     # NOTE: (a') will become (a) for next step
            action_queue.put(action)

            # Save new state (s') as prev state (s)
            prev_raw = raw_state
            state_queue.task_done()
            continue

        # Step 3: prev state (s) is in memory
        state      = extract_state(prev_raw)        # (s)
        next_state = extract_state(raw_state)       # (s')
        reward     = raw_state["reward"]            # (r) NOTE: We get previous action reward

        logger.debug(f"Reward from env: {reward}")

        # Step 4: Update Q-table(s, a, r, s')
        update_q(state, action, reward, next_state)

        # Increment epoch/step
        step += 1

        # Step 5: Check if game over and move to next episode
        if raw_state["gameOver"]:
            
            # Update epsilon at the end of each episode
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            
            # Next Episode
            episode += 1

            logger.info(
                f"Episode {episode} ended — "
                f"steps: {step} | "
                f"score: {raw_state['score']} | "
                f"ε: {epsilon:.4f} | "
                f"Q-table size: {len(q_table)}"
            )

            # reset step and (s, a) for next episode 
            step     = 0
            prev_raw = None     # (s)
            action   = None     # (a)
            
            # Stop training if maximum episodes reached
            if episode >= MAX_EPISODES:
                # TODO: Send stop signal to env
                logger.info(f"Training complete — episodes: {episode} | ε: {epsilon:.4f}")
                break
        
        # Step 6: If game not over then choose next action and send to env
        else:
            action = choose_action(next_state, epsilon)   # (a)
            action_queue.put(action)
            prev_raw = raw_state                          # (s') -> (s)

        state_queue.task_done()