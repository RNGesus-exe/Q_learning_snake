import queue

state_queue:  queue.Queue[dict] = queue.Queue()
action_queue: queue.Queue[str]  = queue.Queue()