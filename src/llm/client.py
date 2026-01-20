# This code handles talking to different AI models (Mistral, CodeLlama, Phi-2)
# If one model doesn't work, it automatically tries another one

import os
import requests
import logging
import time

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """
    This error is raised when all LLM operations fail.
    It could be network problems, servers down, or bad responses.
    """
    pass

class LLMClient:
    # Default web addresses for each AI model
    # These can be changed using environment variables
    DEFAULT_ENDPOINTS = {
        'mistral': 'http://localhost:8001/generate',
        'codellama': 'http://localhost:8002/generate',
        'phi2': 'http://localhost:8003/generate'
    }

    def __init__(self, preferred_model='mistral', model_endpoints=None):
        """
        Set up the client to talk to multiple AI models.
        model_endpoints: custom web addresses for models (optional)
        """
        # Get the web addresses for each model
        self.model_endpoints = {}
        endpoints_config = model_endpoints or {}

        for model_name in ['mistral', 'codellama', 'phi2']:
            # Check if there's a custom address in environment variables
            env_var_name = f"M2V_{model_name.upper()}_ENDPOINT"
            address = os.environ.get(env_var_name) or endpoints_config.get(model_name) or self.DEFAULT_ENDPOINTS.get(model_name)
            self.model_endpoints[model_name] = address

        # Remember which model we prefer to use first
        self.preferred_model = preferred_model
        if preferred_model not in self.model_endpoints:
            logger.warning(f"Preferred model '{preferred_model}' not found, using 'mistral' instead")
            self.preferred_model = 'mistral'

        # Keep track of which models are working
        self.model_is_healthy = {name: True for name in self.model_endpoints.keys()}

        logger.info(f"LLM client ready. Preferred model: {self.preferred_model}")
        for name, address in self.model_endpoints.items():
            logger.info(f"  {name}: {address}")
    
    def check_health(self, model_name, timeout_seconds=5):
        """
        Returns True if the model is healthy, False if it's not.
        """
        address = self.model_endpoints.get(model_name)
        if not address:
            return False

        try:
            # Try to check if the server has a health page
            health_address = address.replace('/generate', '/health')
            response = requests.get(health_address, timeout=timeout_seconds)
            if response.status_code == 200:
                return True

            # If no health page, try a small test request
            response = requests.post(address, json={"prompt": "test"}, timeout=timeout_seconds)
            return response.status_code in [200, 201]
        except:
            return False
    
    def get_fallback_order(self):
        """
        Get the list of models to try, in order.
        Preferred model first, then other healthy models.
        """
        # Start with the preferred model
        order = [self.preferred_model]

        # Add other models that are working
        other_models = [m for m in self.model_endpoints.keys() if m != self.preferred_model]

        # Sort so healthy models come first
        other_models.sort(key=lambda m: (not self.model_is_healthy[m], m))

        order.extend(other_models)
        return order
    
    def generate(self, prompt, timeout_seconds=60, max_tries=3):
        """
        Generate code using the available models.
        Tries the preferred model first, then falls back to others if it fails.
        """
        fallback_order = self.get_fallback_order()
        last_error_message = None

        for model_name in fallback_order:
            address = self.model_endpoints[model_name]

            # Skip models we know are not working
            if not self.model_is_healthy[model_name]:
                logger.debug(f"Skipping {model_name} (marked as not healthy)")
                continue

            logger.info(f"Trying model: {model_name}")

            try:
                response = requests.post(
                    address,
                    json={"prompt": prompt, "model": model_name},
                    timeout=timeout_seconds
                )
                response.raise_for_status()
                data = response.json()

                # Get the generated text from the response
                result = data.get("output") or data.get("text") or data.get("response") or data.get("generated_text") or ""

                if result:
                    # Success! Mark this model as healthy
                    self.model_is_healthy[model_name] = True
                    logger.info(f"Successfully generated code using {model_name}")
                    return result
                else:
                    logger.warning(f"Model {model_name} returned empty response")

            except requests.exceptions.Timeout:
                logger.warning(f"Model {model_name} timed out after {timeout_seconds} seconds")
                self.model_is_healthy[model_name] = False
                last_error_message = f"Timeout on {model_name}"

            except requests.exceptions.ConnectionError:
                logger.warning(f"Could not connect to {model_name} at {address}")
                self.model_is_healthy[model_name] = False
                last_error_message = f"Connection error to {model_name}"

            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                self.model_is_healthy[model_name] = False
                last_error_message = str(e)

        # If we get here, all models failed
        error_message = f"All models failed. Last error: {last_error_message}"
        logger.error(error_message)
        raise LLMClientError(error_message)
    
    def get_model_status(self):
        """
        Get the health status of all models.
        Useful for debugging and seeing which models are working.
        """
        status = {}
        for model_name in self.model_endpoints.keys():
            is_working = self.check_health(model_name)
            self.model_is_healthy[model_name] = is_working
            status[model_name] = {
                'address': self.model_endpoints[model_name],
                'healthy': is_working,
                'preferred': model_name == self.preferred_model
            }
        return status
