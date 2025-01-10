import logging
import warnings

def logging_setup(log_file: str = 'utils.log', logger_name: str = 'Utils') -> None:
    """
    Setup logging with a file handler.

    Args:
        log_file (str): The path to the log file. Defaults to 'utils.log'.
        logger_name (str): The name of the logger. Defaults to 'Utils'.

    Returns:
        None
    """

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create file handler for file logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) 

    # add formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear() 
        print('Cleared existing logger handlers.')

    # add file handler to logger
    logger.addHandler(file_handler)

    return logger

# create default logger
logger = logging_setup()

# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)
