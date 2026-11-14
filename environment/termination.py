from enum import Enum


class TerminationReason(str, Enum):
    SUCCESS = "success"
    MANUFACTURING_FAILURE = "manufacturing_failure"
    HORIZON_TIMEOUT = "horizon_timeout"
    COMMUNICATION_FAILURE = "communication_failure"