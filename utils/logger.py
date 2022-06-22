import logging
import os
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

logger = logging.getLogger(os.getenv("service_name", ""))
logger.setLevel(os.getenv("log-level", "INFO"))

formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream = stream_handler.stream

# file_logger = logging.FileHandler("./logs/system_logs.log", "w")

logger.addHandler(stream_handler)
# logger.addHandler(file_logger)
