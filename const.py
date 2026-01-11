"""Constants for the energy planner integration."""

from __future__ import annotations

import logging
from datetime import timedelta

DOMAIN = "energy_planner"
SERVICE_UPDATE_PLAN = "update_plan"
SERVICE_RUN_OPTIMIZER = "run_optimizer"

CONF_PLAN_LIMIT = "plan_limit"
CONF_MARKDOWN_LIMIT = "markdown_limit"
CONF_MARKDOWN_MAX_LENGTH = "markdown_max_length"

DEFAULT_NAME = "Energy Plan"
DEFAULT_PLAN_LIMIT = 288
DEFAULT_MARKDOWN_LIMIT = 96
DEFAULT_MARKDOWN_MAX_LENGTH = 0
DEFAULT_SCAN_INTERVAL = timedelta(minutes=15)

DATA_COORDINATORS = "coordinators"
DATA_SERVICE_REGISTERED = "service_registered"
DATA_OPTIMIZER_SERVICE_REGISTERED = "optimizer_service_registered"

LOGGER = logging.getLogger(__package__ or DOMAIN)
