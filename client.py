import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Optional
import requests

from queues import action_queue

logger = logging.getLogger("client")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

file_handler = RotatingFileHandler("logs/client.log", maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


GAME_SERVER_URL = "http://localhost:13000/"

#TODO: /start-recording

def wait_for_env(interval: float = 1.0):
    logger.info("Waiting for env to be ready...")
    logger.info(f"Checking health: {GAME_SERVER_URL}health")

    while True:
        try:
            resp = requests.get(f"{GAME_SERVER_URL}health", timeout=10)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                logger.info("Env is ready.")
                return
        except requests.RequestException:
            pass

        logger.debug(f"Env not ready - retrying in {interval}s")
        time.sleep(interval)

def send_seed(
    grid: list[int],
    snake: Optional[list[int]]  = None,
    direction: Optional[str]    = None,
    fruit: Optional[list[int]]  = None
):
    payload = {"grid": grid}

    if snake is not None:
        payload["snake"] = snake
    if direction is not None:
        payload["direction"] = direction
    if fruit is not None:
        payload["fruit"] = fruit
    
    try:
        # logger.info(f"Sending seed: {GAME_SERVER_URL}seed")
        resp = requests.post(f"{GAME_SERVER_URL}seed", json=payload, timeout=10)
        resp.raise_for_status()
        logger.info(f"Seed sent - payload: {payload} | response: {resp.status_code}")

    except requests.RequestException as e:
        logger.error(f"Failed to send seed: {e}")


def client_loop():
    logger.info("Client loop started - waiting for actions...")

    while True:
        action = action_queue.get()

        try:
            # logger.info(f"Sending action: {GAME_SERVER_URL}perform_action")
            resp = requests.post(f"{GAME_SERVER_URL}perform_action", json={"action": action}, timeout=10)
            resp.raise_for_status()
            logger.debug(f"Action '{action}' sent - response: {resp.status_code}")

        except requests.RequestException as e:
            logger.error(f"Failed to send action '{action}': {e}")

        finally:
            action_queue.task_done()