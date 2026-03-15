import os
import shutil
import threading
import logging
import uvicorn

from model import training_loop
from client import client_loop

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    # Start client thread
    client_thread = threading.Thread(
        target=client_loop,
        daemon=True,
        name="ClientThread",
    )
    client_thread.start()
    logger.info("Client thread started")
    
    # Start server thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": "server:app", "host": "0.0.0.0", "port": 14000, "log_level": "info"},
        daemon=True,
        name="ServerThread",
    )
    server_thread.start()
    logger.info("Server thread started")

    # Training loop runs on the main process [BLOCKING CALL]
    training_loop()