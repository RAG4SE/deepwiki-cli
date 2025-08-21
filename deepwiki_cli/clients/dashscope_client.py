"""Dashscope (Alibaba Cloud) ModelClient integration."""

import os
from typing import (
    Dict,
    Optional,
    Any,
    Callable,
    Generator,
    Union,
    Literal,
    List,
    Sequence,
)

import backoff
from copy import deepcopy
from tqdm import tqdm
import adalflow as adal

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI, Stream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    CompletionUsage,
    GeneratorOutput,
    Document,
    Embedding,
    EmbedderOutputType,
    EmbedderInputType,
)
from adalflow.core.component import DataComponent
from adalflow.core.embedder import (
    BatchEmbedderOutputType,
    BatchEmbedderInputType,
)
import adalflow.core.functional as F
from adalflow.components.model_client.utils import parse_embedding_response

from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

log = get_tqdm_compatible_logger(__name__)


def get_first_message_content(completion: ChatCompletion) -> str:
    """When we only need the content of the first message."""
    log.info(f"🔍 get_first_message_content called with: {type(completion)}")
    log.debug(f"raw completion: {completion}")

    if hasattr(completion, "choices") and len(completion.choices) > 0:
        choice = completion.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            content = choice.message.content
            log.info(
                f"✅ Successfully extracted content: {type(content)}, length: {len(content) if content else 0}"
            )
            return content
        else:
            log.error("❌ Choice doesn't have message.content")
            return str(completion)
    else:
        raise ValueError("❌ Completion doesn't have choices")


def parse_stream_response(completion: ChatCompletionChunk) -> str:
    """Parse the response of the stream API."""
    return completion.choices[0].delta.content


# TODO: test handle_streaming_response
async def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """Handle the streaming response asynchronously."""
    async for completion in generator:
        log.debug(f"Raw chunk completion: {completion}")
        parsed_content = parse_stream_response(completion)
        yield parsed_content


class DashScopeClient(ModelClient):
    """A component wrapper for the Dashscope (Alibaba Cloud) API client.

    Dashscope provides access to Alibaba Cloud's Qwen and other models through an OpenAI-compatible API.

    Args:
        api_key (Optional[str], optional): Dashscope API key. Defaults to None.
        workspace_id (Optional[str], optional): Dashscope workspace ID. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://dashscope.aliyuncs.com/compatible-mode/v1".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "DASHSCOPE_API_KEY".
        env_workspace_id_name (str): Environment variable name for the workspace ID. Defaults to "DASHSCOPE_WORKSPACE_ID".

    References:
        - Dashscope API Documentation: https://help.aliyun.com/zh/dashscope/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "DASHSCOPE_BASE_URL",
        env_api_key_name: str = "DASHSCOPE_API_KEY",
        env_workspace_id_name: str = "DASHSCOPE_WORKSPACE_ID",
        **kwargs,
    ):
        super().__init__()
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._env_api_key_name = env_api_key_name
        self._env_workspace_id_name = env_workspace_id_name
        self._env_base_url_name = env_base_url_name
        self.base_url = base_url or os.getenv(
            self._env_base_url_name, "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.sync_client = self.init_sync_client()
        self.async_client = self.init_async_client()

        # Force use of get_first_message_content to ensure string output
        if chat_completion_parser is None:
            self.chat_completion_parser = get_first_message_content
        else:
            self.chat_completion_parser = chat_completion_parser
        self._input_type = input_type
        self._api_kwargs = {}

    def _prepare_client_config(self):
        """
        Private helper method to prepare client configuration.

        Returns:
            tuple: (api_key, workspace_id, base_url) for client initialization

        Raises:
            ValueError: If API key is not provided
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        workspace_id = self._workspace_id or os.getenv(self._env_workspace_id_name)

        if not api_key:
            raise ValueError(
                f"clients/dashscope_client.py:Environment variable {self._env_api_key_name} must be set."
            )

        if not workspace_id:
            log.warning(
                f"Environment variable {self._env_workspace_id_name} not set. Some features may not work properly."
            )

        # For Dashscope, we need to include the workspace ID in the base URL if provided
        base_url = self.base_url
        if workspace_id:
            # Add workspace ID to headers or URL as required by Dashscope
            base_url = f"{self.base_url.rstrip('/')}"

        return api_key, workspace_id, base_url

    def init_sync_client(self):
        api_key, workspace_id, base_url = self._prepare_client_config()

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Store workspace_id for later use in requests
        if workspace_id:
            client._workspace_id = workspace_id

        return client

    def init_async_client(self):
        api_key, workspace_id, base_url = self._prepare_client_config()

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Store workspace_id for later use in requests
        if workspace_id:
            client._workspace_id = workspace_id

        return client

    # TODO: parse streaming response
    """
    This function is required in adalflow.core.generator.Generator._post_call
    """

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Stream[ChatCompletionChunk]],
    ) -> "GeneratorOutput":
        """Parse the completion response to a GeneratorOutput."""
        # If the completion is already a GeneratorOutput, return it directly (prevent recursion)
        if not isinstance(completion, ChatCompletion):
            raise ValueError(
                f"Invalid completion type: expected ChatCompletion, got {type(completion)}"
            )

        reply = self.chat_completion_parser(completion)

        '''
        Here is the _post_call function in adalflow.core.generator.Generator
        
        def _post_call(self, completion: Any) -> GeneratorOutput:
            r"""Get string completion and process it with the output_processors."""
            # parse chat completion will only fill the raw_response
            output: GeneratorOutput = self.model_client.parse_chat_completion(completion)
            # Now adding the data field to the output
            data = output.raw_response
            if self.output_processors:
                if data:
                    try:
                        data = self.output_processors(data)
                        output.data = data
                    except Exception as e:
                        log.error(f"Error processing the output processors: {e}")
                        output.error = str(e)

            else:
                output.data = data

            return output
        
        If transforms output.raw_response to output.data, which will be used later for retrieving
        the completion content. So we need to assign `reply` to GeneratorOutput's raw_response
        '''

        return GeneratorOutput(raw_response=reply)

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        """Track the completion usage."""
        if isinstance(completion, ChatCompletion):
            return CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
        else:
            # For streaming, we can't track usage accurately
            return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        """Parse the embedding response to a EmbedderOutput."""
        # Add detailed debugging
        try:
            result = parse_embedding_response(response)
            if not result.data:
                log.error(f"🔍 No embedding data found in result")
                raise
            return result
        except Exception as e:
            log.error(f"🔍 Error parsing DashScope embedding response: {e}")
            log.error(f"🔍 Raw response details: {repr(response)}")
            raise

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to API kwargs."""
        final_model_kwargs = model_kwargs.copy()

        if model_type == ModelType.LLM:
            messages = []
            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, list):
                messages = input
            else:
                raise ValueError(
                    f"clients/dashscope_client.py:Unsupported input type: {type(input)}"
                )

            api_kwargs = {"messages": messages, **final_model_kwargs}

            # Add workspace ID to headers if available
            workspace_id = getattr(self.sync_client, "_workspace_id", None) or getattr(
                self.async_client, "_workspace_id", None
            )
            if workspace_id:
                # Dashscope may require workspace ID in headers
                if "extra_headers" not in api_kwargs:
                    api_kwargs["extra_headers"] = {}
                api_kwargs["extra_headers"]["X-DashScope-WorkSpace"] = workspace_id
            return api_kwargs

        elif model_type == ModelType.EMBEDDER:
            # Convert Documents to text strings for embedding
            processed_input = input
            if isinstance(input, list):
                # Extract text from Document objects
                processed_input = []
                for item in input:
                    if hasattr(item, "text"):
                        # It's a Document object, extract text
                        processed_input.append(item.text)
                    elif isinstance(item, str):
                        # It's already a string
                        processed_input.append(item)
                    else:
                        # Try to convert to string
                        processed_input.append(str(item))
            elif hasattr(input, "text"):
                # Single Document object
                processed_input = input.text
            elif isinstance(input, str):
                # Single string
                processed_input = input
            else:
                # Convert to string as fallback
                processed_input = str(input)

            # Filter out batch_size and other unsupported parameters for DashScope API
            filtered_kwargs = {
                k: v for k, v in final_model_kwargs.items() if k not in ["batch_size"]
            }

            api_kwargs = {"input": processed_input, **filtered_kwargs}

            # Ensure model parameter is included for DashScope API
            if "model" not in api_kwargs and hasattr(self, "model"):
                api_kwargs["model"] = self.model

            # Add workspace ID to headers if available
            workspace_id = getattr(self.sync_client, "_workspace_id", None) or getattr(
                self.async_client, "_workspace_id", None
            )
            if workspace_id:
                if "extra_headers" not in api_kwargs:
                    api_kwargs["extra_headers"] = {}
                api_kwargs["extra_headers"]["X-DashScope-WorkSpace"] = workspace_id

            return api_kwargs
        else:
            raise ValueError(
                f"clients/dashscope_client.py:model_type {model_type} is not supported"
            )

    def chat(self, api_kwargs: Dict = {}):
        if not api_kwargs.get("stream", False):
            # For non-streaming, enable_thinking must be false.
            # Pass it via extra_body to avoid TypeError from openai client validation.
            extra_body = api_kwargs.get("extra_body", {})
            extra_body["enable_thinking"] = False
            api_kwargs["extra_body"] = extra_body

        completion = self.sync_client.chat.completions.create(**api_kwargs)

        return completion

    async def achat(self, api_kwargs: Dict = {}):
        """Async version of chat method."""
        if not api_kwargs.get("stream", False):
            # For non-streaming, enable_thinking must be false.
            # Pass it via extra_body to avoid TypeError from openai client validation.
            extra_body = api_kwargs.get("extra_body", {})
            extra_body["enable_thinking"] = False
            api_kwargs["extra_body"] = extra_body

        completion = await self.async_client.chat.completions.create(**api_kwargs)

        return completion

    def embeddings(self, api_kwargs: Dict = {}):
        # Extract input texts from api_kwargs
        texts = api_kwargs.get("input", [])

        if not texts:
            log.warning("😭 No input texts provided")
            return EmbedderOutput(
                data=[], error="No input texts provided", raw_response=None
            )

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Filter out empty or None texts - following HuggingFace client pattern
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                log.warning(
                    f"🔍 Skipping empty or invalid text at index {i}: type={type(text)}, length={len(text) if hasattr(text, '__len__') else 'N/A'}, repr={repr(text)[:100]}"
                )

        if not valid_texts:
            log.error("😭 No valid texts found after filtering")
            return EmbedderOutput(
                data=[],
                error="No valid texts found after filtering",
                raw_response=None,
            )

        if len(valid_texts) != len(texts):
            filtered_count = len(texts) - len(valid_texts)
            log.warning(
                f"🔍 Filtered out {filtered_count} empty/invalid texts out of {len(texts)} total texts"
            )

        # Create modified api_kwargs with only valid texts
        filtered_api_kwargs = api_kwargs.copy()
        filtered_api_kwargs["input"] = valid_texts

        response = self.sync_client.embeddings.create(**filtered_api_kwargs)
        result = self.parse_embedding_response(response)

        # If we filtered texts, we need to create embeddings for the original indices
        if len(valid_texts) != len(texts):
            # Get the correct embedding dimension from the first valid embedding
            embedding_dim = None  # Must be determined from a successful response
            if (
                result.data
                and len(result.data) > 0
                and hasattr(result.data[0], "embedding")
            ):
                embedding_dim = len(result.data[0].embedding)

            final_data = []
            valid_idx = 0
            for i in range(len(texts)):
                if i in valid_indices:
                    # Use the embedding from valid texts
                    final_data.append(result.data[valid_idx])
                    valid_idx += 1
                else:
                    # Create zero embedding for filtered texts with correct dimension
                    log.warning(
                        f"🔍 Creating zero embedding for filtered text at index {i}"
                    )
                    final_data.append(
                        Embedding(
                            embedding=[0.0] * embedding_dim,  # Use correct embedding dimension
                            index=i,
                        )
                    )

            result = EmbedderOutput(
                data=final_data, error=None, raw_response=result.raw_response
            )

        return result

    async def aembeddings(self, api_kwargs: Dict = {}):
        """Async version of embeddings method."""
        # Extract input texts from api_kwargs
        texts = api_kwargs.get("input", [])

        if not texts:
            log.warning("😭 No input texts provided")
            return EmbedderOutput(
                data=[], error="No input texts provided", raw_response=None
            )

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Filter out empty or None texts - following HuggingFace client pattern
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                log.warning(
                    f"🔍 Skipping empty or invalid text at index {i}: type={type(text)}, length={len(text) if hasattr(text, '__len__') else 'N/A'}, repr={repr(text)[:100]}"
                )

        if not valid_texts:
            log.error("😭 No valid texts found after filtering")
            return EmbedderOutput(
                data=[],
                error="No valid texts found after filtering",
                raw_response=None,
            )

        if len(valid_texts) != len(texts):
            filtered_count = len(texts) - len(valid_texts)
            log.warning(
                f"🔍 Filtered out {filtered_count} empty/invalid texts out of {len(texts)} total texts"
            )

        # Create modified api_kwargs with only valid texts
        filtered_api_kwargs = api_kwargs.copy()
        filtered_api_kwargs["input"] = valid_texts

        response = await self.async_client.embeddings.create(**filtered_api_kwargs)
        result = self.parse_embedding_response(response)

        # If we filtered texts, we need to create embeddings for the original indices
        if len(valid_texts) != len(texts):
            # Get the correct embedding dimension from the first valid embedding
            embedding_dim = None  # Must be determined from a successful response
            if (
                result.data
                and len(result.data) > 0
                and hasattr(result.data[0], "embedding")
            ):
                embedding_dim = len(result.data[0].embedding)

            final_data = []
            valid_idx = 0
            for i in range(len(texts)):
                if i in valid_indices:
                    # Use the embedding from valid texts
                    final_data.append(result.data[valid_idx])
                    valid_idx += 1
                else:
                    # Create zero embedding for filtered texts with correct dimension
                    log.warning(
                        f"🔍 Creating zero embedding for filtered text at index {i}"
                    )
                    final_data.append(
                        Embedding(
                            embedding=[0.0] * embedding_dim,  # Use correct embedding dimension
                            index=i,
                        )
                    )

            result = EmbedderOutput(
                data=final_data, error=None, raw_response=result.raw_response
            )

        return result


    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(
        self,
        api_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
        **kwargs,
    ):
        """Call the Dashscope API."""
        if model_type == ModelType.LLM:
            return self.chat(api_kwargs)
        elif model_type == ModelType.EMBEDDER:
            return self.embeddings(api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """Async call to the Dashscope API."""

        if model_type == ModelType.LLM:
            return await self.achat(api_kwargs)
        elif model_type == ModelType.EMBEDDER:
            return await self.aembeddings(api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an instance from a dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "api_key": self._api_key,
            "workspace_id": self._workspace_id,
            "base_url": self.base_url,
            "input_type": self._input_type,
        }

    def __getstate__(self):
        """
        Customize serialization to exclude non-picklable client objects.
        This method is called by pickle when saving the object's state.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable client instances
        if "sync_client" in state:
            del state["sync_client"]
        if "async_client" in state:
            del state["async_client"]
        return state

    def __setstate__(self, state):
        """
        Customize deserialization to re-create the client objects.
        This method is called by pickle when loading the object's state.
        """
        self.__dict__.update(state)
        # Re-initialize the clients after unpickling
        self.sync_client = self.init_sync_client()
        self.async_client = self.init_async_client()  # It will be lazily initialized when acall is used


class DashScopeEmbedder(adal.Embedder):
    r"""
    A user-facing component that orchestrates an embedder model via the DashScope model client and output processors.

    Args:
        model_client (ModelClient): The DashScope model client to use for the embedder.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}.
        output_processors (Optional[Component], optional): The output processors after model call. Defaults to None.
    """

    model_type: ModelType = ModelType.EMBEDDER

    def __init__(
        self,
        *,
        model_kwargs: Dict[str, Any] = {},
        output_processors: Optional[DataComponent] = None,
    ) -> None:

        super().__init__(model_client=DashScopeClient(), model_kwargs=model_kwargs, output_processors=output_processors)
        if not isinstance(model_kwargs, Dict):
            raise TypeError(
                f"clients/dashscope_client.py:{type(self).__name__} requires a dictionary for model_kwargs, not a string"
            )

    def call(
        self,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        log.debug(f"Calling {self.__class__.__name__} with input: {input}")
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=input,
            model_kwargs=self._compose_model_kwargs(**model_kwargs),
            model_type=self.model_type,
        )
        output = self.model_client.call(
            api_kwargs=api_kwargs, model_type=self.model_type
        )

        return output

    async def acall(
        self,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        log.debug(f"Calling {self.__class__.__name__} with input: {input}")
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=input,
            model_kwargs=self._compose_model_kwargs(**model_kwargs),
            model_type=self.model_type,
        )
        output: EmbedderOutputType = None
        try:
            response = await self.model_client.acall(
                api_kwargs=api_kwargs, model_type=self.model_type
            )
            output = self.model_client.parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error calling the DashScope model: {e}")
            output = EmbedderOutput(error=str(e))
            raise

        output.input = [input] if isinstance(input, str) else input
        log.debug(f"Output from {self.__class__.__name__}: {output}")
        return output

    def _compose_model_kwargs(self, **model_kwargs) -> Dict[str, object]:
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)


# Batch Embedding Components for DashScope
class DashScopeBatchEmbedder(adal.BatchEmbedder):
    """Batch embedder specifically designed for DashScope API"""

    def __init__(self, embedder, batch_size: int = 100) -> None:
        super().__init__(embedder=embedder, batch_size=batch_size)
        if self.batch_size > 10:
            log.warning(
                f"DashScope batch embedder initialization, batch size: {self.batch_size}, note that DashScope batch embedding size cannot exceed 25, automatically set to 10"
            )
            self.batch_size = 10

    def call(
        self, input: BatchEmbedderInputType, model_kwargs: Optional[Dict] = {}
    ) -> BatchEmbedderOutputType:
        """
        Batch call to DashScope embedder

        Args:
            input: List of input texts
            model_kwargs: Model parameters

        Returns:
            Batch embedding output
        """
        # Check cache first

        if isinstance(input, str):
            input = [input]

        n = len(input)
        embeddings: List[EmbedderOutput] = []

        log.info(
            f"Starting DashScope batch embedding processing, total {n} texts, batch size: {self.batch_size}"
        )

        for i in tqdm(
            range(0, n, self.batch_size),
            desc="DashScope batch embedding",
            disable=False,
        ):
            batch_input = input[i : min(i + self.batch_size, n)]

            try:
                batch_output = self.embedder(
                    input=batch_input, model_kwargs=model_kwargs
                )
                embeddings.append(batch_output)

                # Validate batch output
                if batch_output.error:
                    log.error(
                        f"Batch {i//self.batch_size + 1} embedding failed: {batch_output.error}"
                    )
                elif batch_output.data:
                    log.debug(
                        f"Batch {i//self.batch_size + 1} successfully generated {len(batch_output.data)} embedding vectors"
                    )
                else:
                    log.warning(
                        f"Batch {i//self.batch_size + 1} returned no embedding data"
                    )

            except Exception as e:
                log.error(f"Batch {i//self.batch_size + 1} processing exception: {e}")
                # Create error embedding output
                error_output = EmbedderOutput(data=[], error=str(e), raw_response=None)
                embeddings.append(error_output)
                raise

        log.info(
            f"DashScope batch embedding completed, processed {len(embeddings)} batches"
        )

        return embeddings

    def __call__(
        self,
        input: BatchEmbedderInputType,
        model_kwargs: Optional[Dict] = {},
        force_recreate: bool = False,
    ) -> BatchEmbedderOutputType:
        """
        Call operator interface, delegates to call method
        """
        return self.call(input=input, model_kwargs=model_kwargs)