from dataclasses import dataclass
from typing import List, Dict
from uuid import uuid4

import adalflow as adal
from adalflow.core.types import RetrieverOutput
from adalflow.core.types import Document

from deepwiki_cli.configs import configs
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from deepwiki_cli.rag.retriever import HybridRetriever
from deepwiki_cli.rag.data_pipeline import DatabaseManager
from deepwiki_cli.core.types import DualVectorDocument

# Configure logging
logger = get_tqdm_compatible_logger(__name__)

system_prompt = r"""
You are a code assistant which answers user questions on a Github Repo or a local repo.
You will receive user query, relevant context, and past conversation history.

LANGUAGE DETECTION AND RESPONSE:
- Detect the language of the user's query
- Respond in the SAME language as the user's query
- IMPORTANT:If a specific language is requested in the prompt, prioritize that language over the query language

FORMAT YOUR RESPONSE USING MARKDOWN:
- Use proper markdown syntax for all formatting
- For code blocks, use triple backticks with language specification (```python, ```javascript, etc.)
- Use ## headings for major sections
- Use bullet points or numbered lists where appropriate
- Format tables using markdown table syntax when presenting structured data
- Use **bold** and *italic* for emphasis
- When referencing file paths, use `inline code` formatting

IMPORTANT FORMATTING RULES:
1. DO NOT include ```markdown fences at the beginning or end of your answer
2. Start your response directly with the content
3. The content will already be rendered as markdown, so just provide the raw markdown content

Think step by step and ensure your answer is well-structured and visually organized.
"""

# Template for RAG
RAG_TEMPLATE = r"""<START_OF_SYS_PROMPT>
{{system_prompt}}
{{output_format_str}}
<END_OF_SYS_PROMPT>
{# OrderedDict of DialogTurn #}
{% if conversation_history %}
<START_OF_CONVERSATION_HISTORY>
{% for key, dialog_turn in conversation_history.items() %}
{{key}}.
User: {{dialog_turn.user_query.query_str}}
You: {{dialog_turn.assistant_response.response_str}}
{% endfor %}
<END_OF_CONVERSATION_HISTORY>
{% endif %}
{% if contexts %}
<START_OF_CONTEXT>
{% for context in contexts %}
{{loop.index}}.
File Path: {{context.meta_data.get('file_path', 'unknown')}}
{% if context.meta_data.get('understanding_text') %}
Code Summary:
---
{{context.meta_data.get('understanding_text')}}
---
{% endif %}
Code Snippet:
```
{{context.text}}
```
{% endfor %}
<END_OF_CONTEXT>
{% endif %}
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
"""

from dataclasses import dataclass, field


@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(
        default="", metadata={"desc": "Chain of thoughts for the answer."}
    )
    answer: str = field(
        default="",
        metadata={
            "desc": "Answer to the user query, formatted in markdown for beautiful rendering with react-markdown. DO NOT include ``` triple backticks fences at the beginning or end of your answer."
        },
    )

    __output_fields__ = ["rationale", "answer"]


class RAG(adal.Component):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_path) first."""

    def __init__(self):
        """
        Initialize the RAG component.
        """
        super().__init__()
        assert "generator" in configs(), "configs must contain generator section"
        generator_config = configs()["generator"]
        assert (
            "provider" in generator_config
        ), "generator_config must contain provider section"
        assert (
            "model" in generator_config
        ), "generator_config must contain model section"
        model = generator_config["model"]
        assert (
            "model_client" in generator_config
        ), "generator_config must contain model_client section"
        model_client_class = generator_config["model_client"]
        model_client = model_client_class()
        assert (
            "model_kwargs" in generator_config
        ), "generator_config must contain model_kwargs section"
        model_kwargs = generator_config["model_kwargs"]
        model_kwargs["model"] = model

        # Format instructions for natural language output (no structured parsing)
        format_instructions = """
Please provide a comprehensive answer to the user's question based on the provided context.

IMPORTANT FORMATTING RULES:
1. Respond in the same language as the user's question
2. Format your response using markdown for better readability
3. Use code blocks, bullet points, headings, and other markdown features as appropriate
4. Be clear, concise, and helpful
5. If you use code examples, make sure they are properly formatted with language-specific syntax highlighting
6. Structure your answer logically with clear sections if the question is complex"""

        # Set up the main generator (no output processors to avoid JSON parsing issues)
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "conversation_history": None,  # No conversation history
                "system_prompt": system_prompt,
                "contexts": None,
            },
            model_client=model_client,
            model_kwargs=model_kwargs,
        )

        self.retriever = None
        self.db_manager = None
        self.documents = None

    def initialize_db_manager(self, repo_path: str):
        """Initialize the database manager with local storage"""
        self.db_manager = DatabaseManager(repo_path)
        self.documents = []

    def _validate_and_filter_embeddings(
        self, documents: List[Document | DualVectorDocument]
    ) -> List:
        """
        Validate embeddings and filter out documents with invalid or mismatched embedding sizes.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of documents with valid embeddings of consistent size
        """
        if not documents:
            logger.warning("No documents provided for embedding validation")
            return []

        valid_documents = []
        embedding_sizes = {}
        code_embedding_sizes = {}
        understanding_embedding_sizes = {}
        is_dual_vector = False
        # First pass: collect all embedding sizes and count occurrences
        for i, doc in enumerate(documents):
            if isinstance(doc, Document):
                if not hasattr(doc, "vector") or doc.vector is None:
                    logger.warning(
                        f"❓Document {i} has no embedding vector, skipping...\n doc: {doc}"
                    )
                    continue

                try:
                    if isinstance(doc.vector, list):
                        embedding_size = len(doc.vector)
                    elif hasattr(doc.vector, "shape"):
                        embedding_size = (
                            doc.vector.shape[0]
                            if len(doc.vector.shape) == 1
                            else doc.vector.shape[-1]
                        )
                    elif hasattr(doc.vector, "__len__"):
                        embedding_size = len(doc.vector)
                    else:
                        logger.warning(
                            f"Document {i} has invalid embedding vector type: {type(doc.vector)}, skipping"
                        )
                        continue

                    if embedding_size == 0:
                        logger.warning(
                            f"❓Document {i} has empty embedding vector, skipping...\n doc: {doc}"
                        )
                        continue

                    embedding_sizes[embedding_size] = (
                        embedding_sizes.get(embedding_size, 0) + 1
                    )

                except Exception as e:
                    logger.warning(
                        f"Error checking embedding size for document {i}: {str(e)}, skipping"
                    )
                    continue
            elif isinstance(doc, DualVectorDocument):
                is_dual_vector = True
                if hasattr(doc, "code_embedding"):
                    code_embedding_sizes[len(doc.code_embedding)] = (
                        code_embedding_sizes.get(len(doc.code_embedding), 0) + 1
                    )
                if hasattr(doc, "understanding_embedding"):
                    understanding_embedding_sizes[len(doc.understanding_embedding)] = (
                        understanding_embedding_sizes.get(
                            len(doc.understanding_embedding), 0
                        )
                        + 1
                    )
            else:
                raise ValueError(
                    f"❓Document {i} has invalid type: {type(doc)}, skipping...\n"
                )

        if not is_dual_vector:
            if not embedding_sizes:
                logger.error("No valid embeddings found in any documents")
                return []

            # Find the most common embedding size (this should be the correct one)
            target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
            logger.info(
                f"Target embedding size: {target_size} (found in {embedding_sizes[target_size]} documents)"
            )

            # Log all embedding sizes found
            for size, count in embedding_sizes.items():
                if size != target_size:
                    logger.warning(
                        f"Found {count} documents with incorrect embedding size {size}, will be filtered out"
                    )

            # Second pass: filter documents with the target embedding size
            for i, doc in enumerate(documents):
                if not hasattr(doc, "vector") or doc.vector is None:
                    continue

                try:
                    if isinstance(doc.vector, list):
                        embedding_size = len(doc.vector)
                    elif hasattr(doc.vector, "shape"):
                        embedding_size = (
                            doc.vector.shape[0]
                            if len(doc.vector.shape) == 1
                            else doc.vector.shape[-1]
                        )
                    elif hasattr(doc.vector, "__len__"):
                        embedding_size = len(doc.vector)
                    else:
                        continue

                    if embedding_size == target_size:
                        valid_documents.append(doc)
                    else:
                        # Log which document is being filtered out
                        file_path = getattr(doc, "meta_data", {}).get(
                            "file_path", f"document_{i}"
                        )
                        logger.warning(
                            f"Filtering out document '{file_path}' due to embedding size mismatch: {embedding_size} != {target_size}"
                        )

                except Exception as e:
                    file_path = getattr(doc, "meta_data", {}).get(
                        "file_path", f"document_{i}"
                    )
                    logger.warning(
                        f"Error validating embedding for document '{file_path}': {str(e)}, skipping"
                    )
                    continue

            logger.info(
                f"Embedding validation complete: {len(valid_documents)}/{len(documents)} documents have valid embeddings"
            )

            if len(valid_documents) == 0:
                logger.error(
                    "No documents with valid embeddings remain after filtering"
                )
            elif len(valid_documents) < len(documents):
                filtered_count = len(documents) - len(valid_documents)
                logger.warning(
                    f"Filtered out {filtered_count} documents due to embedding issues"
                )

            return valid_documents

        else:
            if not code_embedding_sizes or not understanding_embedding_sizes:
                logger.error("No valid embeddings found in any documents")
                return []
            target_code_embedding_size = max(
                code_embedding_sizes.keys(), key=lambda k: code_embedding_sizes[k]
            )
            # Some understanding text is "" and its embedding size is 0. We need to remove them when calculating the primary embedding size
            target_understanding_embedding_size = max(
                {
                    k: v for k, v in understanding_embedding_sizes.items() if k > 0
                }.keys(),
                key=lambda k: understanding_embedding_sizes[k],
            )
            logger.info(
                f"Target code embedding size: {target_code_embedding_size} (found in {code_embedding_sizes[target_code_embedding_size]} documents)"
            )
            logger.info(
                f"Target understanding embedding size: {target_understanding_embedding_size} (found in {understanding_embedding_sizes[target_understanding_embedding_size]} documents)"
            )
            for i, doc in enumerate(documents):
                assert isinstance(doc, DualVectorDocument)
                if (
                    len(doc.code_embedding) != target_code_embedding_size
                    or len(doc.understanding_embedding)
                    != target_understanding_embedding_size
                    and len(doc.understanding_embedding) > 0
                ):
                    logger.warning(
                        f"Filtering out document '{doc.file_path}' due to embedding size mismatch: {len(doc.code_embedding)} != {target_code_embedding_size} or {len(doc.understanding_embedding)} != {target_understanding_embedding_size}"
                    )
                else:
                    valid_documents.append(doc)
            return valid_documents

    def prepare_retriever(self, repo_path: str):
        """
        Prepare the retriever for a repository.
        Will load database from local storage if available.

        Args:
            repo_path: URL or local path to the repository
        """
        self.initialize_db_manager(repo_path)

        logger.info(f"🔍 Build up database...")
        # self.documents is a list of Document or DualVectorDocument
        self.documents = self.db_manager.prepare_database()
        logger.info(f"✅ Loaded {len(self.documents)} documents for retrieval")
        # Validate and filter embeddings to ensure consistent sizes
        self.documents = self._validate_and_filter_embeddings(self.documents)
        logger.info(f"🎉Validated and filtered {len(self.documents)} documents")
        if not self.documents:
            raise ValueError(
                "No valid documents with embeddings found. Cannot create retriever."
            )

        # Use HybridRetriever which combines BM25 and FAISS
        self.retriever = HybridRetriever(documents=self.documents)

    def call(self, query: str) -> List[RetrieverOutput]:
        """
        Query the RAG system.
        """
        if not self.retriever:
            raise ValueError("Retriever not prepared. Call prepare_retriever first.")

        logger.info(f"🏃 Running RAG for query: '{query}'")

        try:
            retrieved_docs = self.retriever.call(query)
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")
            raise
