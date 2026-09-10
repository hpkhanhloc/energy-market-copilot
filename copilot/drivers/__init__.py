"""Driver checks: each turns one hypothesis into a number and an honest verdict."""

from copilot.drivers.base import DriverResult, Verdict
from copilot.drivers.checks import ALL_CHECKS, run_all

__all__ = ["ALL_CHECKS", "DriverResult", "Verdict", "run_all"]
