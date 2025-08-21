import os
import asyncio
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
        assert (
            "model" in kwargs
        ), f"rag/dual_vector_pipeline.py:model not found in code_understanding_generator_config"
        self.model = kwargs["model"]
        assert (
            "model_client" in kwargs
        ), f"rag/dual_vector_pipeline.py:model_client not found in code_understanding_generator_config"
        model_client = kwargs["model_client"]
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
        assert "model_kwargs" in kwargs, f"model_kwargs not found in the hydra config of code_understanding in rag.yaml."
        self.model_kwargs = kwargs["model_kwargs"]

        assert "batch_size" in kwargs, f"batch_size not found in the hydra config of code_understanding in rag.yaml."
        self.batch_size = kwargs["batch_size"]

        # Get API configuration from environment
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "CodeUnderstandingGenerator: DASHSCOPE_API_KEY environment variable not set"
            )

    def call(
        self, code: Union[str, List[str]], file_path: Union[str, List[str]]
    ) -> Union[List[str], None]:
        """
        Generates a summary for the given code snippet.

        Args:
            code: The code string to be summarized, or a list of code strings to be summarized.
            file_path: The file path where the code is located (optional).

        Returns:
            A list of generated code summary strings.
        """
        if isinstance(code, str):
            code = [code]
        if isinstance(file_path, str):
            file_path = [file_path]
        assert len(code) == len(file_path), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"
        summaries = []
        for i, code_snippet in enumerate(code):  
            prompt = f"File Path: `{file_path[i]}`\n\n```\n{code_snippet}\n```"
            try:
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

                summaries.append(summary.strip())
            except Exception as e:
                logger.error(f"Failed to generate code understanding for {file_path[i]}: {e}")
                # Return an empty or default summary on error
                return None
            
        return summaries

    async def acall(self, code: Union[str, List[str]], file_path: Union[str, List[str]]) -> Union[List[str], None]:
        """
        Generates a summary for the given code snippet.

        Args:
            code: The code string to be summarized, or a list of code strings to be summarized.
            file_path: The file path where the code is located (optional).

        Returns:
            A list of generated code summary strings.
        """
        if isinstance(code, str):
            code = [code]
        if isinstance(file_path, str):
            file_path = [file_path]
        assert len(code) == len(file_path), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"
        
        async def _acall(code_snippet, file_path):
            prompt = f"File Path: `{file_path}`\n\n```\n{code_snippet}\n```"
            try:
                result = await self.client.acall(
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
                # Re-raise the exception to let caller handle it
                raise
        
        tasks = [_acall(c, f) for c, f in zip(code, file_path)]
        try:
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            logger.error(f"One or more code understanding tasks failed: {e}")
            return None
    
    def call_in_sequence(self, code: List[str], file_path: List[str]) -> List[str]:
        assert len(code) == len(file_path), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"
        understanding_texts = []
        for _, code_snippet in enumerate(
            tqdm(
                code,
                desc="Generating code understanding",
                disable=False,
                total=len(code),
            )
        ):
            understanding_text = self.call(code_snippet, file_path[i])
            understanding_texts.extend(understanding_text)
    
    async def batch_call(self, code: List[str], file_path: List[str], batch_size: Optional[int] = None) -> List[str]:
        """
        Process code understanding in batches to avoid overwhelming the API.
        
        Args:
            code: List of code strings to process
            file_path: List of corresponding file paths
            batch_size: Number of concurrent requests per batch (uses config if None)
            
        Returns:
            List of understanding texts
        """
        assert len(code) == len(file_path), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"
        
        # Use configured batch size if not provided
        if batch_size is None:
            batch_size = self.batch_size
            
        logger.info(f"Processing {len(code)} documents with batch size: {batch_size}")
        
        # Divide tasks into batches
        all_results = []
        total_items = len(code)
        
        with tqdm(
            total=total_items,
            desc="Generating code understanding (batched)",
            disable=False
        ) as pbar:
            
            for i in range(0, total_items, batch_size):
                # Create batch
                batch_end = min(i + batch_size, total_items)
                batch_code = code[i:batch_end]
                batch_paths = file_path[i:batch_end]
                
                logger.debug(f"Processing batch {i//batch_size + 1}: items {i+1}-{batch_end} of {total_items}")
                
                # Create tasks for current batch
                batch_tasks = [
                    self.acall(code_snippet, file_path_item) 
                    for code_snippet, file_path_item in zip(batch_code, batch_paths)
                ]
                
                # Process batch with asyncio.gather for parallel execution within batch
                try:
                    batch_results = await asyncio.gather(*batch_tasks)
                    
                    # Flatten and collect results from this batch
                    for result in batch_results:
                        if result:  # Check if result is not None
                            all_results.extend(result)
                        pbar.update(1)
                        
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Update progress bar even on error
                    pbar.update(len(batch_tasks))
                    # Re-raise exception to handle at higher level
                    raise
                
                # Optional: Add small delay between batches to be API-friendly
                if batch_end < total_items:  # Not the last batch
                    await asyncio.sleep(0.1)  # 100ms delay between batches
        
        logger.info(f"Completed batch processing: {len(all_results)} understanding texts generated")
        return all_results