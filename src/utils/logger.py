import logging

def setup_logger():
    logger = logging.getLogger("agent_logs")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler("logs/app.log", mode="w")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
