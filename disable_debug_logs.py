#!/usr/bin/env python3
"""Disable debug logs for cleaner output"""
from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add new handler with INFO level only
logger.add(sys.stderr, level="INFO")

print("Debug logs disabled. Only INFO and above will be shown.")