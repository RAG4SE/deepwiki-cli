"""Dynamic splitter transformer that selects appropriate splitter based on document type."""

import os
from typing import List, Union, Any

from adalflow.core.types import Document
from adalflow.core.component import Component
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from deepwiki_cli.rag.splitter_factory import get_splitter_factory
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class DynamicSplitterTransformer(Component):
    """Transformer that dynamically selects appropriate splitter based on document type."""

    def __init__(self, batch_size: int = None):
        """Initialize the dynamic splitter transformer."""
        super().__init__()
        self.splitter_factory = get_splitter_factory()
        self.batch_size = batch_size
        if not self.batch_size:
            self.batch_size = os.cpu_count()
        logger.info("Initialized DynamicSplitterTransformer")

    def _process_batch(self, documents: List[Document]) -> List[Document]:
        """Process a batch of documents with appropriate splitters using multiprocessing."""
        result_documents = []
        
        # Use ProcessPoolExecutor to parallelize document processing
        with ProcessPoolExecutor(max_workers=min(len(documents), mp.cpu_count())) as executor:
            # Submit all documents for processing
            futures = [executor.submit(self._process_single_document, doc) for doc in documents]
            
            # Collect results as they complete
            for future in futures:
                try:
                    result_documents.extend(future.result())
                except Exception as e:
                    logger.error(f"Error processing document in parallel: {e}")
        
        return result_documents
    
    def _process_single_document(self, doc: Document) -> List[Document]:
        """Process a single document with appropriate splitter."""
        splitter = self.splitter_factory.get_splitter(
            content=doc.text,
            file_path=getattr(doc, "meta_data", {}).get("file_path", ""),
        )
        return splitter.call([doc])

    def call(self, documents: List[Document]) -> List[Document]:
        """Process documents with appropriate splitters using batch optimization.

        Args:
            documents (List[Document]): Input documents

        Returns:
            List[Document]: Split documents
        """
        if not documents:
            return []

        result_documents = []

        for start_idx in tqdm(
            range(0, len(documents), self.batch_size),
            desc="Splitting Documents in Batches",
        ):
            end_idx = start_idx + self.batch_size
            batch_documents = documents[start_idx:end_idx]
            result_documents.extend(self._process_batch(batch_documents))
        return result_documents

        # Step 1: Group documents by splitter type
        splitter_groups = (
            {}
        )  # key: splitter_key, value: {'splitter': splitter, 'docs': [docs]}

        for doc in tqdm(documents, desc="Grouping documents by splitter type"):
            try:
                # Get appropriate splitter for this document
                file_path = getattr(doc, "meta_data", {}).get("file_path", "")
                splitter = self.splitter_factory.get_splitter(
                    content=doc.text,
                    file_path=file_path,
                )

                # Create splitter key
                splitter_key = splitter.get_key()

                # Group documents by splitter
                if splitter_key not in splitter_groups:
                    splitter_groups[splitter_key] = {
                        "splitter": splitter,
                        "docs": [],
                    }

                splitter_groups[splitter_key]["docs"].append(doc)

            except Exception as e:
                logger.error(
                    f"Error analyzing document {getattr(doc, 'meta_data', {}).get('file_path', 'unknown')}: {e}"
                )
                raise

        # Step 2: Process each splitter group in batch
        result_documents = []
        logger.info(f"Found {len(splitter_groups)} unique splitter configurations")

        for splitter_key, group_info in splitter_groups.items():
            logger.info(f"Processing splitter group {splitter_key}")
            splitter = group_info["splitter"]
            docs = group_info["docs"]

            try:
                # Batch split all documents for this splitter
                split_docs = splitter.call(docs)
                result_documents.extend(split_docs)

            except Exception as e:
                # Skip this unsplittable document
                logger.warning(
                    f"Error processing splitter group {splitter_key}: {e}"
                )
                continue

        logger.info(
            f"Processed {len(documents)} documents into {len(result_documents)} chunks using {len(splitter_groups)} splitter groups"
        )

        return result_documents

    def __call__(self, documents: List[Document]) -> List[Document]:
        """Make the transformer callable.

        Args:
            documents (List[Document]): Input documents

        Returns:
            List[Document]: Split documents
        """
        return self.call(documents)
