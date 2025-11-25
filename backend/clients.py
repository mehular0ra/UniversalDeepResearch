# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Client management module for LLM and search API interactions.

This module provides client creation and management for:
- Large Language Models (OpenAI, NVIDIA, local vLLM)
- Web search (Tavily API)
- Configuration-based client setup
"""

from typing import Any, Dict, List, Literal, TypedDict

from openai import OpenAI
from tavily import TavilyClient

from google.genai import Client as GeminiClient
from google.genai import types

from config import get_config

# Get configuration
config = get_config()

# Configuration system
ApiType = Literal["nvdev", "openai", "tavily", "gemini"]


class ModelConfig(TypedDict):
    base_url: str
    api_type: ApiType
    completion_config: Dict[str, Any]


# Available model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gemini-2.5": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_type": "gemini",
        "completion_config": {
            "model": "models/gemini-2.0-flash",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 2048,
            "stream": True,
        },
    },
    "gpt-4.1-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_type": "openai",
        "completion_config": {
            "model": "gpt-4.1-mini",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 2048,
            "stream": True,
        },
    },
    "llama-3.1-8b": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_type": "nvdev",
        "completion_config": {
            "model": "nvdev/meta/llama-3.1-8b-instruct",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 2048,
            "stream": True,
        },
    },
    "llama-3.1-nemotron-8b": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_type": "nvdev",
        "completion_config": {
            "model": "nvdev/nvidia/llama-3.1-nemotron-nano-8b-v1",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 2048,
            "stream": True,
        },
    },
    "llama-3.1-nemotron-253b": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_type": "nvdev",
        "completion_config": {
            "model": "nvdev/nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 2048,
            "stream": True,
        },
    },
}

# Default model to use (from configuration)
DEFAULT_MODEL = config.model.default_model


def get_api_key(api_type: ApiType) -> str:
    """
    Get the API key for the specified API type.

    This function reads API keys from configuration-specified files.
    The file paths can be customized via environment variables.

    Args:
        api_type: The type of API to get the key for ("nvdev", "openai", "tavily")

    Returns:
        str: The API key from the configured file

    Raises:
        FileNotFoundError: If the API key file doesn't exist
        ValueError: If the API type is unknown

    Example:
        >>> get_api_key("tavily")
        "your-tavily-api-key"
    """
    api_key_files = {
        "nvdev": config.model.api_key_file,
        "openai": "openai_api.txt",
        "tavily": config.search.tavily_api_key_file,
        "gemini": "gemini_api.txt",
    }

    key_file = api_key_files.get(api_type)
    if not key_file:
        raise ValueError(f"Unknown API type: {api_type}")

    try:
        with open(key_file, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key file not found for {api_type}. "
            f"Please create {key_file} with your API key. "
            f"See README.md for configuration instructions."
        )


def create_lm_client(model_config: ModelConfig | None = None) -> OpenAI:
    """
    Create an OpenAI and Gemini client instance with the specified configuration.

    This function creates a client for the configured LLM provider.
    The client can be customized with specific model configurations
    or will use the default model from configuration.

    Args:
        model_config: Optional model configuration to override defaults.
                     If None, uses the default model from configuration.

    Returns:
        OpenAI: Configured OpenAI client instance

    Example:
        >>> client = create_lm_client()
        >>> response = client.chat.completions.create(...)
    """
    model_config = model_config or MODEL_CONFIGS[DEFAULT_MODEL]
    api_key = get_api_key(model_config["api_type"])

    if model_config["api_type"] == "gemini":
        return GeminiClient(api_key=api_key)
    elif model_config["api_type"] == "openai":
        return OpenAI(base_url=model_config["base_url"], api_key=api_key)


def create_tavily_client() -> TavilyClient:
    """
    Create a Tavily client instance for web search functionality.

    This function creates a client for the Tavily search API using
    the API key from the configured file path.

    Returns:
        TavilyClient: Configured Tavily client instance

    Raises:
        FileNotFoundError: If the Tavily API key file is not found

    Example:
        >>> client = create_tavily_client()
        >>> results = client.search("quantum computing")
    """
    api_key = get_api_key("tavily")
    return TavilyClient(api_key=api_key)


def get_completion(
    client: OpenAI | GeminiClient,
    messages: List[Dict[str, Any]],
    model_config: ModelConfig | None = None,
) -> str:
    """
    Get completion from the OpenAI client using the specified model configuration.

    This function handles both streaming and non-streaming completions,
    with special handling for certain model configurations that require
    specific message formatting.

    Args:
        client: OpenAI or Gemini client instance
        messages: List of messages for the completion
        model_config: Optional model configuration to override defaults.
                     If None, uses the default model configuration.

    Returns:
        str: The completion text

    Example:
        >>> client = create_lm_client()
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = get_completion(client, messages)
        >>> print(response)
        "Hello! How can I help you today?"
    """
    model_config = model_config or MODEL_CONFIGS[DEFAULT_MODEL]

    # Handle special model configurations
    if "retarded" in model_config and model_config["retarded"]:
        if messages[0]["role"] == "system":
            first_message = messages[0]
            messages = [msg for msg in messages if msg["role"] != "system"]
            messages[0]["content"] = (
                first_message["content"] + "\n\n" + messages[0]["content"]
            )
            messages.insert(0, {"role": "system", "content": "detailed thinking off"})

    if model_config["api_type"] == "gemini":
        return get_completion_gemini(client, messages, model_config)
    elif model_config["api_type"] == "openai":
        return get_completion_openai(client, messages, model_config)


def get_completion_openai(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model_config: ModelConfig,
) -> str:
    completion = client.chat.completions.create(
        messages=messages, **model_config["completion_config"]
    )

    # Handle streaming vs non-streaming responses
    if model_config["completion_config"]["stream"]:
        ret = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                ret += chunk.choices[0].delta.content
    else:
        ret = completion.choices[0].message.content

    return ret


def gemini_completion(
    client: GeminiClient, messages: List[Dict[str, Any]], model_config: ModelConfig
) -> str:
    history = to_gemini_history(messages)
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=model_config["completion_config"],
        history=history,
    )
    cfg = _gemini_config_from_completion_config(model_config["completion_config"])
    response = chat.send_message("", config=cfg)
    return response.text


def to_gemini_history(messages):
    out = []
    for m in messages:
        # Expecting your internal format: {"role": "...", "content": "..."}
        out.append(
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [{"text": m["content"]}],
            }
        )
    return out


def _gemini_config_from_completion_config(cfg: dict) -> types.GenerateContentConfig:
    """Convert OpenAI-style completion_config into a valid Gemini GenerateContentConfig."""

    # extract only fields Gemini accepts
    temperature = cfg.get("temperature")
    top_p = cfg.get("top_p")
    top_k = cfg.get("top_k")
    candidate_count = cfg.get("candidate_count") or cfg.get("n")
    max_output_tokens = cfg.get("max_output_tokens") or cfg.get("max_tokens")

    typed = {}

    if temperature is not None:
        typed["temperature"] = float(temperature)
    if top_p is not None:
        typed["top_p"] = float(top_p)
    if top_k is not None:
        typed["top_k"] = int(top_k)
    if candidate_count is not None:
        typed["candidate_count"] = int(candidate_count)
    if max_output_tokens is not None:
        typed["max_output_tokens"] = int(max_output_tokens)

    return types.GenerateContentConfig(**typed)


def is_output_positive(output: str) -> bool:
    """
    Check if the output contains positive indicators.

    This function checks if the given output string contains
    positive words like "yes" or "true" (case-insensitive).

    Args:
        output: The string to check for positive indicators

    Returns:
        bool: True if positive indicators are found, False otherwise

    Example:
        >>> is_output_positive("Yes, that's correct")
        True
        >>> is_output_positive("No, that's not right")
        False
    """
    positive_words = ["yes", "true"]
    return any(word in output.lower() for word in positive_words)
