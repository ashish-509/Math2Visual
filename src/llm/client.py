# Simple LLM client for Math2Visual
# Connects to a language model server and gets generated text for a prompt.
# Set M2V_LLM_ENDPOINT in your environment, or just use the default below.

import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """
    Custom exception raised when LLM client operations fail.
    
    This exception is raised for network errors, invalid responses,
    or server-side failures during code generation requests.
    """
    pass


class LLMClient:
    def __init__(self, endpoint=None):
        # Use env var or fallback to localhost
        self.endpoint = endpoint or os.environ.get("M2V_LLM_ENDPOINT", "http://localhost:8000/generate")
        logger.info(f"LLMClient using endpoint: {self.endpoint}")

    def generate(self, prompt, timeout=60):
        # Send prompt to server, get response
        try:
            resp = requests.post(self.endpoint, json={"prompt": prompt}, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Try a few possible keys for output
            return data.get("output") or data.get("text") or data.get("response") or ""
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise LLMClientError(str(e))
