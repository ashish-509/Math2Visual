# src/llm/client.py
import os
import requests
import logging
logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    pass

class LLMClient:
    """
    Minimal client abstraction. It tries environment variables to connect:
      - M2V_LLM_ENDPOINT (http://host:port/generate)
      - M2V_LLM_PROVIDER ('local'|'hf'|'llama-server')
    The server is expected to accept a JSON {"prompt": "..."} and return {"output": "..."}.
    """
    def __init__(self, endpoint: str = None):
        self.endpoint = endpoint or os.environ.get("M2V_LLM_ENDPOINT", "http://localhost:8000/generate")
        logger.info("LLMClient configured to %s", self.endpoint)

    def generate(self, prompt: str, timeout: int = 60) -> str:
        try:
            resp = requests.post(self.endpoint, json={"prompt": prompt}, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            out = data.get("output") or data.get("text") or data.get("response") or ""
            return out
        except Exception as e:
            logger.exception("LLM request failed")
            raise LLMClientError(str(e))
