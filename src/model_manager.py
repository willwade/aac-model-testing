"""
Model Manager

This module handles the management and interaction with language models
through the native Ollama Python client with structured output support.
"""

import subprocess
import logging
from typing import Dict, Any, List, Type, TypeVar
import time
import os
import platform
from pydantic import BaseModel
import ollama
from ollama import ListResponse, list as ollama_list


T = TypeVar("T", bound=BaseModel)


class ModelManager:
    """
    Manages language model interactions through the native Ollama client.

    This class provides a unified interface for interacting with Ollama models
    with structured output support using Pydantic schemas.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the model manager.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.logger = logging.getLogger("AAC_Testing.ModelManager")
        self._available_models = None
        self._ollama_path = None
        self._client = None

        # Initialize Ollama client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Ollama client."""
        try:
            # Create Ollama client
            self._client = ollama.Client()

            # Test the connection
            self._client.list()
            self.logger.info("Ollama client initialized successfully")

        except Exception as e:
            raise Exception(f"Failed to initialize Ollama client: {str(e)}")

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models.

        Returns:
            Dictionary of available models with their details
        """
        if self._available_models is None:
            self._available_models = self._fetch_available_models()

        return self._available_models

    def _fetch_available_models(self) -> Dict[str, Any]:
        """Fetch locally installed Ollama models suitable for AAC testing."""
        try:
            # Check if client is initialized
            if not self._client:
                raise Exception("Ollama client not initialized")

            # Get locally installed models using native Ollama client
            self.logger.debug("Fetching locally installed models from Ollama...")
            response: ListResponse = ollama_list()

            models = {}
            self.logger.debug(f"Found {len(response.models)} total models")

            for model in response.models:
                model_name = model.model
                self.logger.debug(f"Found model: {model_name}")

                # Only include models suitable for AAC
                if self._is_suitable_for_aac(model_name):
                    # Calculate size in MB
                    size_mb = (
                        f"{(model.size / 1024 / 1024):.1f}MB"
                        if model.size
                        else "unknown"
                    )

                    models[model_name] = {
                        "provider": "Ollama",
                        "name": model_name,
                        "aliases": [],
                        "available": True,
                        "locally_installed": True,
                        "size": size_mb,
                        "recommended_for_aac": True,
                        "details": {
                            "format": model.details.format if model.details else None,
                            "family": model.details.family if model.details else None,
                            "parameter_size": (
                                model.details.parameter_size if model.details else None
                            ),
                            "quantization_level": (
                                model.details.quantization_level
                                if model.details
                                else None
                            ),
                        },
                    }
                    self.logger.debug(
                        f"Model {model_name} is suitable for AAC (size: {size_mb})"
                    )
                else:
                    self.logger.debug(f"Model {model_name} is not suitable for AAC")

            self.logger.info(
                f"Found {len(models)} locally installed AAC-suitable models: {list(models.keys())}"
            )

            if not models:
                self.logger.warning(
                    "No suitable AAC models found locally. Consider installing: gemma3:1b-it-qat, tinyllama:1.1b"
                )

            return models

        except Exception as e:
            self.logger.error(f"Failed to fetch local models: {str(e)}")
            return {}

    def _is_suitable_for_aac(self, model_name: str) -> bool:
        """
        Determine if a model is suitable for AAC use based on its name/characteristics.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            True if the model is suitable for AAC applications
        """
        model_lower = model_name.lower()

        # Include models with small parameter counts (typically good for AAC)
        small_model_indicators = [
            "1b",  # 1 billion parameters
            "3b",  # 3 billion parameters
            "7b",  # 7 billion parameters (upper limit)
            "mini",  # Often indicates smaller models
            "small",  # Explicitly small models
            "tiny",  # Very small models
            "micro",  # Micro models
            "nano",  # Nano models
            "qat",  # Quantized models (more efficient)
            "q4",  # 4-bit quantized
            "q8",  # 8-bit quantized
        ]

        # Exclude very large models that aren't suitable for low-powered devices
        large_model_indicators = [
            "13b",  # 13+ billion parameters
            "30b",  # 30+ billion parameters
            "70b",  # 70+ billion parameters
            "405b",  # Very large models
            "large",  # Explicitly large models
            "xl",  # Extra large models
        ]

        # Check for large model indicators first (exclusion)
        for indicator in large_model_indicators:
            if indicator in model_lower:
                return False

        # Check for small model indicators (inclusion)
        for indicator in small_model_indicators:
            if indicator in model_lower:
                return True

        # Default inclusion for common small model families
        suitable_families = [
            "gemma",  # Google's Gemma models (generally efficient)
            "phi",  # Microsoft's Phi models (designed for efficiency)
            "llama3.2",  # Llama 3.2 has small variants
            "qwen",  # Qwen has efficient small models
            "mistral",  # Mistral has efficient models
        ]

        for family in suitable_families:
            if family in model_lower:
                return True

        # If we can't determine, err on the side of caution
        return False

    def _get_recommended_aac_models(self) -> Dict[str, Any]:
        """
        Get a list of recommended models for AAC applications.

        Returns:
            Dictionary of recommended models with their metadata
        """
        return {
            "gemma3:1b-it-qat": {
                "provider": "Ollama",
                "name": "gemma3:1b-it-qat",
                "aliases": [],
                "description": "Google Gemma 3 1B parameters, instruction-tuned, quantized",
                "parameters": "1B",
                "quantization": "QAT",
                "memory_requirement_mb": 800,
                "recommended_use": "Primary recommendation - excellent balance of quality and efficiency",
                "aac_suitability": "excellent",
            },
            "tinyllama:1.1b": {
                "provider": "Ollama",
                "name": "tinyllama:1.1b",
                "aliases": [],
                "description": "TinyLlama 1.1B - ultra-lightweight and fast",
                "parameters": "1.1B",
                "quantization": "default",
                "memory_requirement_mb": 700,
                "recommended_use": "Ultra-lightweight option for resource-constrained devices",
                "aac_suitability": "very_good",
            },
            "phi3:mini": {
                "provider": "Ollama",
                "name": "phi3:mini",
                "aliases": [],
                "description": "Microsoft Phi-3 Mini - optimized for efficiency",
                "parameters": "3.8B",
                "quantization": "default",
                "memory_requirement_mb": 2300,
                "recommended_use": "Good alternative with slightly more capability",
                "aac_suitability": "very_good",
            },
            "llama3.2:1b": {
                "provider": "Ollama",
                "name": "llama3.2:1b",
                "aliases": [],
                "description": "Meta Llama 3.2 1B - compact and efficient",
                "parameters": "1B",
                "quantization": "default",
                "memory_requirement_mb": 1200,
                "recommended_use": "Alternative 1B model for comparison",
                "aac_suitability": "very_good",
            },
            "qwen2.5:1.5b": {
                "provider": "Ollama",
                "name": "qwen2.5:1.5b",
                "aliases": [],
                "description": "Alibaba Qwen 2.5 1.5B - efficient multilingual model",
                "parameters": "1.5B",
                "quantization": "default",
                "memory_requirement_mb": 1000,
                "recommended_use": "Good for multilingual AAC applications",
                "aac_suitability": "good",
            },
        }

    def get_model(self, model_name: str) -> "ModelWrapper":
        """
        Get a model wrapper for the specified model.

        Args:
            model_name: Name of the model to get

        Returns:
            ModelWrapper instance for the specified model
        """
        available_models = self.get_available_models()

        # Check if model is available
        if model_name not in available_models:
            # Check aliases
            found = False
            for name, info in available_models.items():
                if model_name in info.get("aliases", []):
                    model_name = name
                    found = True
                    break

            if not found:
                raise Exception(
                    f"Model '{model_name}' not available. Available models: {list(available_models.keys())}"
                )

        return ModelWrapper(model_name, self.verbose)

    def get_installation_instructions(self, model_name: str) -> str:
        """
        Get installation instructions for a model.

        Args:
            model_name: Name of the model

        Returns:
            Installation instructions as a string
        """
        available_models = self.get_available_models()

        if model_name in available_models:
            model_info = available_models[model_name]

            if model_info["available"]:
                return f"Model '{model_name}' is already installed and ready to use."
            else:
                return f"To install '{model_name}', run: ollama pull {model_name}"
        else:
            return f"Model '{model_name}' is not in the recommended list. Please verify the model name."

    def list_recommended_models(self) -> Dict[str, Any]:
        """
        Get a formatted list of recommended models with their details.

        Returns:
            Dictionary with model information formatted for display
        """
        available_models = self.get_available_models()

        result = {
            "installed": {},
            "available_for_install": {},
            "summary": {
                "total_recommended": len(available_models),
                "installed_count": 0,
                "available_count": 0,
            },
        }

        for model_name, model_info in available_models.items():
            if model_info["available"]:
                result["installed"][model_name] = model_info
                result["summary"]["installed_count"] += 1
            else:
                result["available_for_install"][model_name] = model_info
                result["summary"]["available_count"] += 1

        return result

    def get_locally_installed_models(self) -> List[str]:
        """
        Get a list of locally installed model names suitable for AAC.

        Returns:
            List of model names that are locally installed and suitable for AAC
        """
        available_models = self.get_available_models()
        return [
            name
            for name, info in available_models.items()
            if info.get("available", False) and info.get("locally_installed", False)
        ]


class ModelWrapper:
    """
    Wrapper class for individual model interactions.

    This class provides a consistent interface for interacting with a specific
    language model through the native Ollama Python client with structured output.
    """

    def __init__(self, model_name: str, verbose: bool = False):
        """
        Initialize the model wrapper.

        Args:
            model_name: Name of the model
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.verbose = verbose
        self.logger = logging.getLogger(f"AAC_Testing.Model.{model_name}")

        # Create Ollama client
        try:
            self._client = ollama.Client()
            self.logger.info(f"Successfully initialized client for model: {model_name}")
        except Exception as e:
            raise Exception(
                f"Failed to initialize client for model {model_name}: {str(e)}"
            )

        # Test the model to ensure it's working
        self._test_model()

    def _test_model(self):
        """Test that the model is working properly."""
        try:
            test_response = self.generate_response("Hello, can you respond?")
            if not test_response or len(test_response.strip()) == 0:
                raise Exception("Model returned empty response")

            self.logger.info(f"Model {self.model_name} is working properly")

        except Exception as e:
            raise Exception(f"Model {self.model_name} failed test: {str(e)}")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt

        Returns:
            The model's response as a string
        """
        try:
            if self.verbose:
                self.logger.debug(f"Generating response for model: {self.model_name}")
                self.logger.debug(f"Prompt: {prompt[:100]}...")

            start_time = time.time()

            # Use the Ollama client to generate response
            response = self._client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},  # Make responses deterministic
            )

            execution_time = time.time() - start_time

            # Extract response text
            response_text = response["message"]["content"].strip()

            if self.verbose:
                self.logger.debug(f"Response time: {execution_time:.2f}s")
                self.logger.debug(f"Response: {response_text[:200]}...")

            return response_text

        except Exception as e:
            self.logger.error(f"Failed to generate response: {str(e)}")
            raise Exception(f"Model generation failed: {str(e)}")

    def generate_structured_response(self, prompt: str, response_schema: Type[T]) -> T:
        """
        Generate a structured response from the model using Pydantic schema.

        Args:
            prompt: The input prompt
            response_schema: Pydantic model class defining the expected response structure

        Returns:
            Validated response object of the specified schema type
        """
        try:
            if self.verbose:
                self.logger.debug(
                    f"Generating structured response for model: {self.model_name}"
                )
                self.logger.debug(f"Schema: {response_schema.__name__}")
                self.logger.debug(f"Prompt: {prompt[:100]}...")

            start_time = time.time()

            # Use the Ollama client with structured output
            response = self._client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format=response_schema.model_json_schema(),
                options={"temperature": 0},  # Make responses deterministic
            )

            execution_time = time.time() - start_time

            # Parse and validate the response
            response_content = response["message"]["content"]
            validated_response = response_schema.model_validate_json(response_content)

            if self.verbose:
                self.logger.debug(f"Response time: {execution_time:.2f}s")
                self.logger.debug(f"Validated response: {validated_response}")

            return validated_response

        except Exception as e:
            self.logger.error(f"Failed to generate structured response: {str(e)}")
            raise Exception(f"Structured generation failed: {str(e)}")

    def generate_response_with_metadata(
        self, prompt: str, timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Generate a response with additional metadata.

        Args:
            prompt: The input prompt
            timeout: Timeout in seconds

        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()

        try:
            response = self.generate_response(prompt, timeout)
            execution_time = time.time() - start_time

            return {
                "response": response,
                "execution_time": execution_time,
                "model_name": self.model_name,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "success": True,
                "error": None,
            }

        except Exception as e:
            execution_time = time.time() - start_time

            return {
                "response": "",
                "execution_time": execution_time,
                "model_name": self.model_name,
                "prompt_length": len(prompt),
                "response_length": 0,
                "success": False,
                "error": str(e),
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "type": "LLM via Ollama" if "ollama" in self.model_name.lower() else "LLM",
            "available": True,
        }

    def __str__(self) -> str:
        """String representation of the model wrapper."""
        return f"ModelWrapper({self.model_name})"

    def __repr__(self) -> str:
        """Detailed string representation of the model wrapper."""
        return f"ModelWrapper(model_name='{self.model_name}', verbose={self.verbose})"

    def cleanup(self):
        """Clean up model resources."""
        try:
            # Clear the model reference
            self.model = None
            self.logger.debug(f"Cleaned up model: {self.model_name}")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
