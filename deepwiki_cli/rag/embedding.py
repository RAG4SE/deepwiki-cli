from typing import List
from copy import deepcopy
from tqdm import tqdm

import adalflow as adal
from adalflow.core.types import Document, EmbedderOutput
from adalflow.core.component import DataComponent

from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from deepwiki_cli.rag.code_understanding import CodeUnderstandingGenerator
from deepwiki_cli.core.types import DualVectorDocument


logger = get_tqdm_compatible_logger(__name__)

class ToEmbeddings(DataComponent):
    """Component that converts document sequences to embedding vector sequences"""

    def __init__(self, embedder: adal.Embedder | adal.BatchEmbedder) -> None:
        super().__init__()
        self.embedder = embedder

    def __call__(self, input: List[Document]) -> List[Document]:
        """
        Process list of documents, generating embedding vectors for each document

        Args:
            input: List of input documents

        Returns:
            List of documents containing embedding vectors
        """
        output = deepcopy(input)

        # Convert to text list
        embedder_input: List[str] = [chunk.text for chunk in output]

        logger.info(f"Starting to process embeddings for {len(embedder_input)} documents")
        
        # Batch process embeddings
        outputs: List[EmbedderOutput] = self.embedder(input=embedder_input)

        if isinstance(self.embedder, adal.BatchEmbedder):
            for batch_idx, batch_output in enumerate(
                tqdm(
                    outputs,
                    desc="Adding embeddings to documents from batch",
                    disable=False,
                    total=len(outputs),
                )
            ):
                for idx, embedding in enumerate(batch_output.data):
                    output[batch_idx * self.embedder.batch_size + idx].vector = embedding.embedding
        elif isinstance(self.embedder, adal.Embedder):
            for idx, idx_output in enumerate(
                tqdm(
                    outputs,
                    desc="Assigning embedding vectors to documents",
                    disable=False,
                    total=len(outputs),
                )
            ):
                assert len(idx_output.data) == 1, "DashScope embedder should return a single embedding"
                output[idx].vector = idx_output.data[0].embedding
        else:
            raise ValueError(f"Unsupported embedder type: {type(self.embedder)}")
        return output

class DashScopeToEmbeddings(ToEmbeddings):
    """Component that converts document sequences to embedding vector sequences, specifically optimized for DashScope API"""

    def __init__(self, embedder: adal.Embedder | adal.BatchEmbedder) -> None:
        super().__init__(embedder)
        self.embedder = embedder

    def __call__(self, input: List[Document]) -> List[Document]:
        return super().__call__(input)

class HuggingfaceToEmbeddings(ToEmbeddings):
    """Component that converts document sequences to embedding vector sequences, specifically optimized for Huggingface API"""

    def __init__(self, embedder) -> None:
        super().__init__(embedder)
        self.embedder = embedder

    def __call__(self, input: List[Document]) -> List[Document]:
        return super().__call__(input)


class DualVectorToEmbeddings(ToEmbeddings):
    """
    A data component that transforms documents into dual-vector embeddings,
    including both code and understanding vectors.
    """

    def __init__(self, embedder: adal.Embedder | adal.BatchEmbedder, generator: CodeUnderstandingGenerator):
        """
        Initialize the DualVectorToEmbeddings component.

        Args:
            embedder: the embedder instance
            generator: the code understanding generator instance
        """
        super().__init__(embedder=embedder)
        self.code_generator = generator

    def __call__(self, documents: List[Document]) -> List[DualVectorDocument]:
        """
        Processes a list of documents to generate and cache dual-vector embeddings.
        """
        logger.info(
            "Generating dual-vector embeddings for %s documents", len(documents)
        )
        output = super().__call__(documents)
        code_vectors = [doc.vector for doc in output]
        assert len(code_vectors) == len(documents), "The number of code vectors should be the same as the number of documents"
        
        understanding_texts = []
        
        for idx, doc in enumerate(
            tqdm(
                documents,
                desc="Generating code understanding",
                disable=False,
                total=len(documents),
            )
        ):
            assert (
                "is_code" in doc.meta_data
            ), f"rag/dual_vector_pipeline.py:No `is_code` key in meta_data: {doc.meta_data}"
            understanding_text = self.code_generator.generate_code_understanding(
                doc.text, doc.meta_data.get("file_path")
            )
            understanding_texts.append(understanding_text)
        
        summary_vectors = super().__call__([Document(text=text) for text in understanding_texts])
        summary_vectors = [doc.vector for doc in summary_vectors]
        assert len(summary_vectors) == len(code_vectors), f"The number of summary vectors ({len(summary_vectors)}) should be the same as the number of code vectors ({len(code_vectors)})"
        
        dual_docs = [
            DualVectorDocument(
                original_doc=doc,
                code_embedding=code_vectors[idx],
                understanding_embedding=summary_vectors[idx],
                understanding_text=understanding_texts[idx],
            )
            for idx, doc in enumerate(documents)
        ]
        
        logger.info("Successfully generated %s dual-vector documents.", len(dual_docs))
        return dual_docs