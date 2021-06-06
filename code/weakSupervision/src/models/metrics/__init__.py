from .seg_meter import SegMeter
from .med_meter import MedMeter

import logging
from logging import StreamHandler

# Create the Logger
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#Create the Handler for logging data to console.
#console_handler = StreamHandler()

# Create a Formatter for formatting the log messages
#logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
#console_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
#logger.addHandler(console_handler)