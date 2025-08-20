"""GPU-accelerated HuggingFace ModelClient."""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from typing import Sequence
import pickle

import adalflow as adal
from adalflow.core.types import Document
from sentence_transformers import SentenceTransformer
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput, Embedding
from adalflow.core.component import DataComponent
from adalflow.core.embedder import (
    BatchEmbedderOutputType,
    BatchEmbedderInputType,
)
from adalflow.core.types import (
    EmbedderOutputType,
    EmbedderInputType,
    BatchEmbedderInputType,
    BatchEmbedderOutputType,
)
import adalflow.core.functional as F

# Configure logging
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

log = get_tqdm_compatible_logger(__name__)


class HuggingfaceClient(ModelClient):
    """A GPU-accelerated component wrapper for HuggingFace sentence-transformers models.

    This client uses PyTorch GPU acceleration and avoids Faiss GPU dependencies.

    Example:
        ```python
        from api.huggingface_gpu_client import HuggingfaceClient

        client = HuggingfaceClient()
        embedder = HuggingfaceEmbedder(
            model_client=client,
            model_kwargs={"model": "intfloat/multilingual-e5-large-instruct"}
        )
        ```
    """

    def __init__(self, device: str = "cuda", **kwargs):
        """Initialize the HuggingFace GPU client.

        Args:
            device: Device to run the model on ("cuda" or "cpu")
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_name = None

        if self.device == "cuda":
            log.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            log.info("Using CPU (CUDA not available)")

    def _load_model(self, model_name: str):
        """Load the sentence-transformers model.

        Args:
            model_name: Name of the model to load
        """
        if self.model is None or self.model_name != model_name:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_name = model_name

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs and model_kwargs to api_kwargs.

        This method is required by the adalflow ModelClient base class.

        Args:
            input: The input data (text or list of texts)
            model_kwargs: Model-specific keyword arguments
            model_type: Type of model call (EMBEDDER, LLM, etc.)

        Returns:
            Combined api_kwargs dictionary
        """
        api_kwargs = model_kwargs.copy()

        # Handle different input types
        if isinstance(input, str):
            api_kwargs["input"] = [input]
        elif isinstance(input, list):
            api_kwargs["input"] = input
        elif input is not None:
            # Try to convert to list
            try:
                api_kwargs["input"] = list(input)
            except:
                api_kwargs["input"] = [str(input)]
        else:
            api_kwargs["input"] = []

        return api_kwargs

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        """Parse the embedding response from the model.

        This method is required by the adalflow ModelClient base class.

        Args:
            response: The raw response from the model (numpy array or torch tensor)

        Returns:
            EmbedderOutput with parsed embeddings
        """
        try:
            if isinstance(response, EmbedderOutput):
                response = response.data

            if isinstance(response, torch.Tensor):
                embeddings = response.cpu().numpy()
            elif isinstance(response, np.ndarray):
                embeddings = response
            elif hasattr(response, "tolist"):
                embeddings = np.array(response)
            else:
                embeddings = np.array(response)

            # Handle different array shapes
            if embeddings.ndim == 0:
                # Scalar array (0-d), convert to 2-d
                embeddings = np.array([[embeddings]])
            elif embeddings.ndim == 1:
                # 1-d array, reshape to 2-d
                embeddings = embeddings.reshape(1, -1)

            # Ensure we have a 2-d array
            if embeddings.ndim != 2:
                log.error(f"Unexpected embedding shape: {embeddings.shape}")
                return EmbedderOutput(
                    data=[],
                    error=f"Unexpected embedding shape: {embeddings.shape}",
                    raw_response=embeddings,
                )

            # Convert to list format expected by adalflow
            embedding_list = []
            for i, embedding in enumerate(embeddings):
                embedding_list.append(Embedding(embedding=embedding.tolist(), index=i))

            # Create EmbedderOutput
            result = EmbedderOutput(
                data=embedding_list, error=None, raw_response=embeddings
            )

            return result

        except Exception as e:
            log.error(f"Error parsing embedding response: {e}")
            raise

    def call(
        self,
        api_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
        **kwargs,
    ):
        """Make a call to the HuggingFace model.

        Args:
            api_kwargs: Arguments for the model call
            model_type: Type of model call (should be EMBEDDER)

        Returns:
            EmbedderOutput with embeddings
        """
        if model_type != ModelType.EMBEDDER:
            raise ValueError(
                f"clients/huggingface_embedder_client.py:😭 Model type {model_type} is not supported. Only EMBEDDER is supported."
            )

        # Extract model name and texts from api_kwargs
        model_name = api_kwargs.get("model", "intfloat/multilingual-e5-large-instruct")
        texts = api_kwargs.get("input", [])

        if not texts:
            log.warning("😭No input texts provided")
            return EmbedderOutput(
                data=[], error="No input texts provided", raw_response=None
            )

        # Filter out empty or None texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                log.warning(
                    f"Skipping empty or invalid text at index {i}: type={type(text)}, length={len(text) if hasattr(text, '__len__') else 'N/A'}, repr={repr(text)[:100]}"
                )
        if not valid_texts:
            log.error("No valid texts found after filtering")
            return EmbedderOutput(
                data=[], error="No valid texts found after filtering", raw_response=None
            )

        if len(valid_texts) != len(texts):
            filtered_count = len(texts) - len(valid_texts)
            log.warning(
                f"Filtered out {filtered_count} empty/invalid texts out of {len(texts)} total texts"
            )

        try:
            # Load the model
            self._load_model(model_name)

            # Generate embeddings with GPU acceleration
            with torch.no_grad():  # Disable gradient computation for inference
                embeddings = self.model.encode(
                    valid_texts, convert_to_numpy=True, show_progress_bar=False
                )
            # Parse the response
            result = self.parse_embedding_response(embeddings)
            # If we filtered texts, we need to create embeddings for the original indices
            if len(valid_texts) != len(texts):
                log.info(f"Creating embeddings for {len(texts)} original positions")

                # Get the correct embedding dimension from the first valid embedding
                embedding_dim = 1024  # Default fallback
                if (
                    result.data
                    and len(result.data) > 0
                    and hasattr(result.data[0], "embedding")
                ):
                    embedding_dim = len(result.data[0].embedding)
                    log.info(f"Using embedding dimension: {embedding_dim}")

                final_data = []
                final_raw_response = []
                valid_idx = 0
                for i in range(len(texts)):
                    if i in valid_indices:
                        # Use the embedding from valid texts
                        final_data.append(result.data[valid_idx])
                        final_raw_response.append(result.raw_response[valid_idx])
                        valid_idx += 1
                    else:
                        # Create empty embedding for filtered texts with correct dimension
                        log.warning(
                            f"Creating empty embedding for filtered text at index {i}"
                        )
                        final_data.append(
                            Embedding(
                                embedding=[0.0]
                                * embedding_dim,  # Use correct embedding dimension
                                index=i,
                            )
                        )
                        final_raw_response.append(np.zeros(embedding_dim))
                # np.save('./embedding_cache/embeddings.npy', embeddings)
                result = EmbedderOutput(
                    data=final_data,
                    error=None,
                )
                return result

            return result

        except Exception as e:
            log.error(f"🤡 Error generating embeddings: {e}")
            raise

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """Async version of call (not implemented, falls back to sync)."""
        return self.call(api_kwargs, model_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the component to a dictionary."""
        exclude = ["model"]  # Don't serialize the model
        output = super().to_dict(exclude=exclude)
        return output

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an instance from a dictionary."""
        obj = super().from_dict(data)
        obj.model = None  # Model will be loaded when needed
        return obj


ToEmbeddingsInputType = Sequence[Document]
ToEmbeddingsOutputType = Sequence[Document]


class HuggingfaceEmbedder(adal.Embedder):
    r"""
    A user-facing component that orchestrates an embedder model via the model client and output processors.

    Args:
        model_client (ModelClient): The model client to use for the embedder.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}.
        output_processors (Optional[Component], optional): The output processors after model call. Defaults to None.
            If you want to add further processing, it should operate on the ``EmbedderOutput`` data type.

    input: a single str or a list of str. When a list is used, the list is processed as a batch of inputs in the model client.

    Note:
        - The ``output_processors`` will be applied only on the data field of ``EmbedderOutput``, which is a list of ``Embedding``.
        - Use ``BatchEmbedder`` for automatically batching input of large size, larger than 100.
    """

    model_type: ModelType = ModelType.EMBEDDER

    def __init__(
        self,
        *,
        model_kwargs: Dict[str, Any] = {},
        output_processors: Optional[DataComponent] = None,
    ) -> None:

        super().__init__(model_client=HuggingfaceClient(), model_kwargs=model_kwargs, output_processors=output_processors)
        if not isinstance(model_kwargs, Dict):
            raise TypeError(
                f"{type(self).__name__} requires a dictionary for model_kwargs, not a string"
            )

    def call(
        self,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        log.debug(f"Calling {self.__class__.__name__} with input: {input}")
        api_kwargs = self._pre_call(input=input, model_kwargs=model_kwargs)
        try:
            output = self.model_client.call(
                api_kwargs=api_kwargs, model_type=self.model_type
            )
        except Exception as e:
            log.error(f"🤡 Error calling the model: {e}")
            raise
        return output

class HuggingfaceBatchEmbedder(adal.BatchEmbedder):
    __doc__ = r"""Adds batching to the embedder component.

    Args:
        embedder (HuggingfaceEmbedder): The embedder to use for batching.
        batch_size (int, optional): The batch size to use for batching. Defaults to 100.
    """

    def __init__(self, embedder: HuggingfaceEmbedder, batch_size: int = 500) -> None:
        super().__init__(embedder=embedder, batch_size=batch_size)

    def call(
        self, input: BatchEmbedderInputType, model_kwargs: Optional[Dict] = {}
    ) -> BatchEmbedderOutputType:
        r"""Call the embedder with batching.

        Args:
            input (BatchEmbedderInputType): The input to the embedder. Use this when you have a large input that needs to be batched. Also ensure
            the output can fit into memory.
            model_kwargs (Optional[Dict], optional): The model kwargs to pass to the embedder. Defaults to {}.

        Returns:
            BatchEmbedderOutputType: The output from the embedder.
        """
        if isinstance(input, str):
            input = [input]
        n = len(input)
        embeddings: List[EmbedderOutput] = []
        for i in tqdm(
            range(0, n, self.batch_size),
            desc="Batch embedding documents",
        ):
            batch_input = input[i : min(i + self.batch_size, n)]
            batch_output = self.embedder.call(
                input=batch_input, model_kwargs=model_kwargs
            )
            embeddings.append(batch_output)

        return embeddings