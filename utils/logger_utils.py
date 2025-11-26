import logging

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_logger(log_dir, model_name):
    # Evitar agregar múltiples handlers si ya existen
    if not logger.handlers:
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, model_name))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # También agregar un StreamHandler para la consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def log_string(str):
    logger.info(str)
    print(str)
