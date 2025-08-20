import os
import logging
from typing import List, Optional, Union
from openai.types.chat import ChatCompletion
from tqdm import tqdm

import adalflow as adal
from adalflow.core.types import (
    Document,
    ModelType,
    RetrieverOutput,
    RetrieverOutputType,
)
from adalflow.core.component import DataComponent

from deepwiki_cli.clients.dashscope_client import DashScopeClient
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from deepwiki_cli.configs import configs
from deepwiki_cli.core.types import DualVectorDocument

logger = get_tqdm_compatible_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
# System prompt designed specifically for the code understanding task
CODE_UNDERSTANDING_SYSTEM_PROMPT = """
You are an expert programmer and a master of code analysis.
Your task is to provide a concise, high-level summary of the given code snippet.
Focus on the following aspects:
1.  **Purpose**: What is the main goal or functionality of the code?
2.  **Inputs**: What are the key inputs, arguments, or parameters?
3.  **Outputs**: What does the code return or produce?
4.  **Key Logic**: Briefly describe the core logic or algorithm.

Keep the summary in plain language and easy to understand for someone with technical background but not necessarily familiar with this specific code.
Do not get lost in implementation details. Provide a "bird's-eye view" of the code.
The summary should be in English and as concise as possible.
"""

CODE_UNDERSTANDING_SYSTEM_PROMPT = """
You are an expert programmer and a master of code analysis.
Your task is to provide a concise, high-level summary of the given code snippet.

Keep the summary in plain language and easy to understand for someone with technical background but not necessarily familiar with this specific code.
Do not get lost in implementation details. Provide a "bird's-eye view" of the code.
The summary should be in English and as concise as possible.
"""


class CodeUnderstandingGenerator:
    """
    Uses the Dashscope model to generate natural language summaries for code.
    """

    def __init__(self, **kwargs):
        """
        Initializes the code understanding generator.

        """
        code_understanding_generator_config = configs()["rag"]["code_understanding"]
        assert (
            "model" in code_understanding_generator_config
        ), f"rag/dual_vector_pipeline.py:model not found in code_understanding_generator_config"
        self.model = code_understanding_generator_config["model"]
        assert (
            "model_client" in code_understanding_generator_config
        ), f"rag/dual_vector_pipeline.py:model_client not found in code_understanding_generator_config"
        model_client = code_understanding_generator_config["model_client"]
        # Initialize client

        # Get API configuration from environment
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        if model_client == DashScopeClient:
            self.client = model_client(
                api_key=api_key,
                # workspace_id=workspace_id
            )
        else:
            raise ValueError(
                f"rag/dual_vector_pipeline.py:Unsupported client class: {model_client.__class__.name}"
            )

        # Extract configuration
        if "model_kwargs" in code_understanding_generator_config:
            self.model_kwargs = code_understanding_generator_config["model_kwargs"]
        else:
            self.model_kwargs = {}

        # Get API configuration from environment
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "CodeUnderstandingGenerator: DASHSCOPE_API_KEY environment variable not set"
            )

    def generate_code_understanding(
        self, code: Union[str, List[str]], file_path: Optional[str] = None
    ) -> Union[str, None]:
        """
        Generates a summary for the given code snippet.

        Args:
            code: The code string to be summarized.
            file_path: The file path where the code is located (optional).

        Returns:
            The generated code summary string.
        """
        try:
            prompt = f"File Path: `{file_path}`\n\n```\n{code}\n```"

            result = self.client.call(
                api_kwargs={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": CODE_UNDERSTANDING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    **self.model_kwargs,
                },
                model_type=ModelType.LLM,
            )

            # Extract content from GeneratorOutput data field
            assert isinstance(result, ChatCompletion), f"result is not a ChatCompletion: {type(result)}"
            summary = result.choices[0].message.content

            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate code understanding for {file_path}: {e}")
            # Return an empty or default summary on error
            return None