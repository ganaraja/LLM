#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

# Configuration Loading Using svlearn
# ConfigurationMixin is a utility class provided by svlearn for loading settings from a YAML config file
# This is useful to externalize parameters like collection name, index type, model paths, chunk size, etc.
from svlearn.config.configuration import ConfigurationMixin

# Load environment variables from a .env file into the system environment
from dotenv import load_dotenv

load_dotenv()

# Instantiate the mixin and load configuration
# If no path is provided in the .env file, it looks for a default config.yaml in the current or pre-defined directory
config = ConfigurationMixin().load_config()
