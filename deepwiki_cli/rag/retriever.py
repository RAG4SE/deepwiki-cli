import re
from typing import List, Optional, Union, Dict, Any
from rank_bm25 import BM25Okapi

from adalflow.core.types import RetrieverOutput, Document, RetrieverOutputType
from adalflow.components.retriever.faiss_retriever import (
    FAISSRetriever,
    FAISSRetrieverQueriesType,
)

from deepwiki_cli.configs import get_embedder, configs
from deepwiki_cli.core.types import DualVectorDocument
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

class SingleVectorRetriever(FAISSRetriever):
    """A wrapper of FAISSRetriever with the additional feature of supporting docoments feature"""

    def __init__(self, documents: List[Document], *args, **kwargs):
        self.original_doc = documents
        super().__init__(documents=documents, *args, **kwargs)
        logger.info(
            f"SingleVectorRetriever initialized with {len(documents)} documents"
        )

    def call(
        self,
        input: FAISSRetrieverQueriesType,
    ) -> RetrieverOutputType:
        retriever_output = super().call(input, self.top_k)

        # Extract the first result from the list
        if not retriever_output:
            return []

        first_output = retriever_output[0]

        # Get the documents based on the indices
        retrieved_docs = [self.original_doc[i] for i in first_output.doc_indices]

        # Create a new RetrieverOutput with the documents
        return [
            RetrieverOutput(
                doc_indices=first_output.doc_indices,
                doc_scores=first_output.doc_scores,
                query=first_output.query,
                documents=retrieved_docs,
            )
        ]

class DualVectorRetriever:
    """Dual vector retriever: supports dual retrieval from code and summary vectors."""

    def __init__(self, dual_docs: List[DualVectorDocument], embedder, top_k: int = 20):
        """
        Initializes the dual vector retriever.

        Args:
            dual_docs: A list of dual vector documents.
            embedder: The embedder instance.
            top_k: The number of most relevant documents to return.
        """
        self.dual_docs = dual_docs
        self.embedder = embedder
        self.top_k = top_k
        self.doc_map = {doc.original_doc.id: doc for doc in dual_docs}

        # Build the two FAISS indexes
        self._build_indices()
        logger.info(
            f"Dual vector retriever initialization completed, containing {len(dual_docs)} documents"
        )

    def _build_indices(self):
        """Builds the code index and the summary index."""
        if not self.dual_docs:
            logger.warning("No documents available for building indices")
            self.code_retriever = None
            self.understanding_retriever = None
            return

        # 1. Build the code index
        code_docs = []
        for dual_doc in self.dual_docs:
            # Create a document object for FAISS
            faiss_doc = Document(
                text=dual_doc.original_doc.text,
                meta_data=dual_doc.original_doc.meta_data,
                id=f"{dual_doc.original_doc.id}_code",
                vector=dual_doc.code_embedding,
            )
            code_docs.append(faiss_doc)

        self.code_retriever = FAISSRetriever(
            top_k=self.top_k,
            embedder=self.embedder,
            documents=code_docs,
            document_map_func=lambda doc: doc.vector,
        )
        logger.info("Code FAISS index built successfully.")

        # 2. Build the summary index
        understanding_docs = []
        for dual_doc in self.dual_docs:
            faiss_doc = Document(
                text=dual_doc.understanding_text,
                meta_data=dual_doc.original_doc.meta_data,
                id=f"{dual_doc.original_doc.id}_understanding",
                vector=dual_doc.understanding_embedding,
            )
            understanding_docs.append(faiss_doc)

        self.understanding_retriever = FAISSRetriever(
            top_k=self.top_k,
            embedder=self.embedder,
            documents=understanding_docs,
            document_map_func=lambda doc: doc.vector,
        )
        logger.info("Understanding FAISS index built successfully.")

    def call(self, query_str: str) -> RetrieverOutputType:
        """
        Performs dual retrieval.

        Args:
            query_str: The query string.

        Returns:
            A RetrieverOutput object containing the retrieved documents and scores.
        """
        assert isinstance(
            query_str, str
        ), f"Query must be a string, got {type(query_str)}"

        if not self.dual_docs:
            return RetrieverOutput(
                doc_indices=[], doc_scores=[], query=query_str, documents=[]
            )

        # 1. Retrieve from the code index
        code_results = self.code_retriever.call(query_str, top_k=self.top_k)[0]
        # 2. Retrieve from the summary index
        understanding_results = self.understanding_retriever.call(
            query_str, top_k=self.top_k
        )[0]

        # 3. Merge and re-rank the results
        combined_scores = {}

        # Process code results - extract original chunk_id from FAISS document ID
        for i, score in zip(code_results.doc_indices, code_results.doc_scores):
            # Get the document from code retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_code", "")
            combined_scores[original_chunk_id] = score

        # Process understanding results - extract original chunk_id from FAISS document ID
        for i, score in zip(
            understanding_results.doc_indices, understanding_results.doc_scores
        ):
            # Get the document from understanding retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_understanding", "")
            if original_chunk_id not in combined_scores:
                combined_scores[original_chunk_id] = score
            else:
                combined_scores[original_chunk_id] = max(
                    combined_scores[original_chunk_id], score
                )

        # 4. Sort and get the top-k results
        # Sort by the combined score in descending order
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda chunk_id: combined_scores[chunk_id],
            reverse=True,
        )

        # 5. Retrieve the full documents for the top-k chunk_ids and create indices mapping
        top_k_docs = []
        doc_indices = []
        doc_scores = []
        for idx, chunk_id in enumerate(
            sorted_chunk_ids[: min(self.top_k, len(sorted_chunk_ids))]
        ):
            if chunk_id in self.doc_map:
                dual_doc = self.doc_map[chunk_id]
                top_k_docs.append(dual_doc)
                doc_indices.append(idx)
                doc_scores.append(combined_scores[chunk_id])

        logger.info(
            f"Retrieved {len(top_k_docs)} documents after merging code and understanding search results."
        )

        return [
            RetrieverOutput(
                doc_indices=doc_indices,
                doc_scores=doc_scores,
                query=query_str,
                documents=top_k_docs,
            )
        ]

class HybridRetriever:
    """
    Hybrid retriever that combines BM25 keyword filtering with FAISS semantic search.

    The retrieval process:
    1. BM25 first filters documents by exact keyword matches, reducing the search space
    2. FAISS then performs semantic similarity search on this subset
    3. Results are merged and re-ranked for optimal relevance
    """

    def __init__(self, documents: List[Union[Document, DualVectorDocument]], **kwargs):
        """
        Initialize the hybrid retriever.

        Args:
            documents: List of documents to index
            embedder: Embedding model for FAISS
            use_bm25: Whether to enable BM25 filtering
            bm25_top_k: Number of documents to retrieve from BM25 before FAISS
            bm25_k1: BM25 k1 parameter (term frequency saturation)
            bm25_b: BM25 b parameter (field length normalization)
            top_k: Final number of documents to return
            use_dual_vector: Whether to use dual vector retrieval
            **kwargs: Additional arguments for FAISS retriever
        """
        self.documents = documents
        self.embedder = get_embedder()

        rag_config = configs()["rag"]
        assert "embedder" in rag_config, "rag_config must contain embedder section"
        self.use_dual_vector = rag_config["embedder"]["sketch_filling"]
        assert "hybrid" in rag_config, "rag_config must contain hybrid section"
        assert (
            "enabled" in rag_config["hybrid"]
        ), "hybrid_config must contain enabled section"
        self.use_bm25 = rag_config["hybrid"]["enabled"]
        assert "bm25" in rag_config["hybrid"], "hybrid_config must contain bm25 section"
        bm25_config = rag_config["hybrid"]["bm25"]
        assert (bm25_config is None) or (
            "top_k" in bm25_config and "k1" in bm25_config and "b" in bm25_config
        ), "bm25_config must contain top_k, k1, and b parameters"
        self.bm25_top_k = bm25_config["top_k"]
        self.bm25_k1 = bm25_config["k1"]
        self.bm25_b = bm25_config["b"]
        assert (
            "retriever" in rag_config
        ), "embedder_config must contain retriever section"
        retriever_config = rag_config["retriever"]
        assert (
            "top_k" in retriever_config
        ), "retriever_config must contain top_k section"
        self.top_k = retriever_config["top_k"]

        # Initialize BM25 if enabled
        if self.use_bm25:
            self._initialize_bm25()
            # FAISS retriever will be initialized later with filtered documents
            self.faiss_retriever = None
        else:
            # Initialize FAISS retriever with all documents when BM25 is disabled
            self._initialize_faiss_retriever(**kwargs)

        logger.info(
            f"Hybrid retriever initialized with BM25={'enabled' if self.use_bm25 else 'disabled'}, "
            f"dual_vector={'enabled' if self.use_dual_vector else 'disabled'}"
        )

    def _initialize_bm25(self):
        """Initialize BM25 index with document texts."""
        try:
            # Extract text content from documents for BM25 indexing
            corpus = []
            for doc in self.documents:
                if isinstance(doc, DualVectorDocument):
                    # For dual vector documents, combine code and understanding text
                    text = doc.original_doc.text + "\n" + doc.understanding_text
                else:
                    text = doc.text

                # Tokenize text for BM25 (simple whitespace + punctuation splitting)
                tokens = self._tokenize_text(text)
                corpus.append(tokens)

            # Initialize BM25 with custom parameters
            self.bm25 = BM25Okapi(corpus, k1=self.bm25_k1, b=self.bm25_b)
            logger.info(f"BM25 index created with {len(corpus)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.use_bm25 = False
            self.bm25 = None

    def _initialize_faiss_retriever(self, **kwargs):
        """Initialize FAISS retriever based on vector type."""
        if self.use_dual_vector:
            self.faiss_retriever = DualVectorRetriever(
                dual_docs=self.documents,
                embedder=self.embedder,
                top_k=self.top_k,
                **kwargs,
            )
        else:
            self.faiss_retriever = SingleVectorRetriever(
                documents=self.documents,
                embedder=self.embedder,
                top_k=self.top_k,
                document_map_func=lambda doc: doc.vector,
                **kwargs,
            )
        logger.info(f"FAISS retriever initialized successfully")

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25 indexing."""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Split on whitespace and common punctuation, keep alphanumeric tokens
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _bm25_filter(self, query: str) -> List[int]:
        """Filter documents using BM25 and return document indices."""
        if not self.use_bm25 or not self.bm25:
            # If BM25 is disabled, return all document indices
            return list(range(len(self.documents)))

        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)

            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k document indices based on BM25 scores
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )

            # Limit to bm25_top_k documents
            filtered_indices = doc_indices[: self.bm25_top_k]

            logger.info(
                f"BM25 filtered {len(self.documents)} documents to {len(filtered_indices)} candidates"
            )
            return filtered_indices

        except Exception as e:
            logger.error(f"BM25 filtering failed: {e}, falling back to all documents")
            raise

    def call(self, query: str, top_k: Optional[int] = None) -> List[RetrieverOutput]:
        """Perform hybrid retrieval combining BM25 and FAISS."""
        if top_k is None:
            top_k = self.top_k

        try:
            if self.use_bm25:
                # Step 1: BM25 filtering
                bm25_indices = self._bm25_filter(query)
                # Step 2: Filter documents
                self.documents = [self.documents[i] for i in bm25_indices]
                # Step 3: Initialize FAISS retriever with filtered documents
                self._initialize_faiss_retriever()

                if not bm25_indices:
                    logger.warning(
                        "BM25 returned no results, falling back to full FAISS search"
                    )
                    # Initialize FAISS with all documents as fallback
                    return self.faiss_retriever.call(query)

                faiss_results = self.faiss_retriever.call(query)

                return faiss_results

            else:
                # BM25 disabled, use pure FAISS search
                logger.info("BM25 disabled, using pure FAISS search")
                return self.faiss_retriever.call(query)

        except Exception as e:
            logger.error(
                f"Hybrid retrieval failed: {e}, falling back to pure FAISS search"
            )
            raise
            # Fall back to pure FAISS search - ensure retriever is initialized
            if self.faiss_retriever is None:
                self._initialize_faiss_retriever()
            return self.faiss_retriever.call(query)
