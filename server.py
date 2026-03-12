import logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

from queues import state_queue

# Logging
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

file_handler = RotatingFileHandler("logs/server.log", maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

app = FastAPI()

# State store
class UpdateStateRequest(BaseModel):
    grid: List[List[int]]
    score: int
    gameOver: bool

class Vector2(BaseModel):
    x:            float
    y:            float
    magnitude:    float
    sqrMagnitude: float

# TODO: This will be replaced by a 2D image of the game UI that we will receive
# Request model
class UpdateStateRequest(BaseModel):
    grid:          List[List[int]]
    snakeBody:     List[Vector2]
    fruitPosition: Vector2        
    score:         int
    gameOver:      bool
    reward:        float
    seed:          str


# API endpoints
@app.post("/update_state", status_code=201)
def update_state(req: UpdateStateRequest):
    head = req.snakeBody[0]

    # Log the received game state
    logger.info(
        f"/update_state — grid: {len(req.grid)}x{len(req.grid)} | "
        f"snake: ({head.x},{head.y}) | "
        f"fruit: ({req.fruitPosition.x}, {req.fruitPosition.y}) | "
        f"score: {req.score} | "
        f"reward: {req.reward} | "
        f"game_over: {req.gameOver}"
    )

    # Enqueue the data into state_queue
    state_queue.put(req.model_dump())

    # Log current size of queue
    logger.info(f"State pushed to queue — queue size: {state_queue.qsize()}")

    # Log if game is over
    if req.gameOver:
        logger.info(f"Game over — final score: {req.score}")

    # Return response code to server
    return {"status": "ok"}


@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}