import logging

def init_loggers(info_file='info.log', debug_file='debug.log'):
    # Create the logger for information logs
    info_logger = logging.getLogger('info_logger')
    info_logger.setLevel(logging.INFO)

    # Create the logger for debug logs
    debug_logger = logging.getLogger('debug_logger')
    debug_logger.setLevel(logging.DEBUG)

    # Create file handlers for the logs
    info_handler = logging.FileHandler(info_file)
    debug_handler = logging.FileHandler(debug_file)

    # Create formatters for the log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set the formatter for the handlers
    info_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)

    # Add the handlers to the loggers
    info_logger.addHandler(info_handler)
    debug_logger.addHandler(debug_handler)
    return info_logger, debug_logger


if __name__ == '__main__':
    info_logger, debug_logger = init_loggers()
    for i in range(10):
        info_logger.info('This is an informational message')
        debug_logger.debug('This is a debug message')