import logging
from logging.handlers import RotatingFileHandler
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from model import state_queue

# Logging
file_handler = RotatingFileHandler(
    "server.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    handlers=[file_handler, logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

app = FastAPI()

# State store
class UpdateStateRequest(BaseModel):
    grid: List[List[int]]
    score: int
    gameOver: bool


# TODO: This will be replaced by a 2D image of the game UI that we will receive
# Request model
class UpdateStateRequest(BaseModel): # Game State
    grid: List[List[int]]
    score: int
    gameOver: bool


# API endpoints
@app.post("/update_state", status_code=201)
def update_state(req: UpdateStateRequest):

    # Convert the 2D List into numpy format
    grid = np.array(req.grid, dtype=np.int32)
    rows, cols = grid.shape

    # Log the received game state
    logger.info(
        f"/update_state — grid: {rows}x{cols} | "
        f"score: {req.score} | "
        f"game_over: {req.gameOver}"
    )
    logger.debug(f"Grid:\n{grid}")

    # Enqueue the data into state_queue
    state_queue.put(grid)

    # Log current size of queue
    logger.info(f"Grid pushed to queue — queue size: {state_queue.qsize()}")

    # Log if game is over
    if req.gameOver:
        logger.info(f"Game over — final score: {req.score}")

    # Return response code to server
    return {"status": "ok"}


@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}