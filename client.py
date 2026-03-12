import logging
from logging.handlers import RotatingFileHandler
import requests

from model import action_queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    handlers=[
        RotatingFileHandler("client.log", maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

GAME_SERVER_URL = "http://localhost:8000"


#TODO: Add /seed, /start-recording

def client_loop():
    """
    Consumes actions from action_queue and POSTs them to the game server.
    """
    logger.info("Client loop started — waiting for actions...")

    while True:
        action = action_queue.get()

        try:
            resp = requests.post(
                f"{GAME_SERVER_URL}/perform_action",
                json={"action": action},
                timeout=2,
            )
            resp.raise_for_status()
            logger.debug(f"Action '{action}' sent — response: {resp.status_code}")

        except requests.RequestException as e:
            logger.error(f"Failed to send action '{action}': {e}")

        finally:
            action_queue.task_done()