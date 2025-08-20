#!/usr/bin/env python3
"""
RAGalyze Query Module
"""

import os
import sys
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
import hydra

from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from deepwiki_cli.rag.rag import RAG
from deepwiki_cli.configs import *
from deepwiki_cli.core.types import DualVectorDocument

# Setup logging
logger = get_tqdm_compatible_logger(__name__)


def save_query_results(result: Dict[str, Any], repo_path: str, question: str) -> str:
    """
    Save query results to reply folder with timestamp

    Args:
        result: Query result dictionary
        repo_path: Repository path that was queried
        question: Question that was asked

    Returns:
        Path to the created output directory
    """
    # Create timestamp-based folder name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("reply") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save response
    response_file = output_dir / "response.txt"
    with open(response_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("QUERY INFORMATION\n")
        f.write("=" * 50 + "\n")
        f.write(f"Repository: {repo_path}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=" * 50 + "\n")
        f.write("RESPONSE\n")
        f.write("=" * 50 + "\n")
        # Convert escaped newlines to actual line breaks
        formatted_response = result["response"].replace("\\n", "\n")
        f.write(formatted_response + "\n")

    # Save retrieved documents if available
    if result.get("retrieved_documents"):
        docs_file = output_dir / "retrieved_documents.txt"
        with open(docs_file, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("RETRIEVED DOCUMENTS\n")
            f.write("=" * 50 + "\n")
            for i, doc in enumerate(result["retrieved_documents"], 1):
                if isinstance(doc, DualVectorDocument):
                    f.write(
                        f"\n{i}. File: {getattr(doc.original_doc, 'meta_data', {}).get('file_path', 'Unknown')}\n"
                    )
                    f.write("Full Content:\n")
                    f.write(doc.original_doc.text + "\n")
                    understanding_text = getattr(doc, "understanding_text", "")
                    if understanding_text:
                        f.write("Code Understanding:\n")
                        f.write("---\n")
                        f.write(understanding_text + "\n")
                        f.write("---\n\n")
                    f.write("=" * 80 + "\n")
                else:
                    f.write(
                        f"\n{i}. File: {getattr(doc, 'meta_data', {}).get('file_path', 'Unknown')}\n"
                    )

                    f.write("Full Content:\n")
                    f.write(doc.text + "\n")

    # Save metadata
    metadata_file = output_dir / "metadata.txt"
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write("Query Metadata\n")
        f.write("=" * 30 + "\n")
        f.write(f"Repository: {repo_path}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Error Message: {result.get('error_msg', 'None')}\n")
        f.write(
            f"Retrieved Documents Count: {len(result.get('retrieved_documents', []))}\n"
        )

        # Count documents with understanding text
        docs_with_understanding = 0
        if result.get("retrieved_documents"):
            for doc in result["retrieved_documents"]:
                understanding_text = getattr(doc, "meta_data", {}).get(
                    "understanding_text", ""
                )
                if understanding_text:
                    docs_with_understanding += 1
        f.write(f"Documents with Code Understanding: {docs_with_understanding}\n")

    return str(output_dir)


def analyze_repository(repo_path: str) -> RAG:
    """
    Analyze a code repository and return a RAG instance.

    Args:
        repo_path: Path to the repository

    Returns:
        RAG: Initialized RAG instance

    Raises:
        ValueError: If repository path doesn't exist
        Exception: If analysis fails
    """
    repo_path = os.path.abspath(repo_path)

    # Check if path exists
    if not os.path.exists(repo_path):
        raise ValueError(f"core/query.py:Repository path does not exist: {repo_path}")

    logger.info(f"🚀 Starting new analysis for: {repo_path}")

    # Initialize RAG with dual vector flag
    rag = RAG()

    # Prepare retriever
    rag.prepare_retriever(repo_path)

    logger.info(f"✅ Analysis complete for: {repo_path}")

    return rag


def query_repository(repo_path: str, question: str) -> Dict[str, Any]:
    """
    Query a repository about a question and return structured results.

    Args:
        repo_path: Path to the repository to analyze
        question: Question to ask about the repository

    Returns:
        Dict containing:
            - response: The assistant's response text
            - retrieved_documents: List of retrieved documents with metadata
            - error_msg: Error message if any (empty string if no error)
    """
    # Analyze repository
    rag = analyze_repository(repo_path=repo_path)

    if question == "":
        return None

    logger.info(f"🔍 Processing question: {question}")

    # Call RAG system to retrieve documents
    result = rag.call(question)

    if result and len(result) > 0:
        # result[0] is RetrieverOutput
        retriever_output = result[0]
        retrieved_docs = (
            retriever_output.documents if hasattr(retriever_output, "documents") else []
        )

        # Generate answer
        logger.info("🤖 Generating answer...")
        try:
            if isinstance(retrieved_docs[0], DualVectorDocument):
                contexts = [doc.original_doc for doc in retrieved_docs]
            else:
                contexts = retrieved_docs
            generator_result = rag.generator(
                prompt_kwargs={"input_str": question, "contexts": contexts}
            )
        except Exception as e:
            logger.error(f"Error calling generator: {e}")
            raise

        # Only use the content of the first choice
        try:
            rag_answer = generator_result.data.strip()
        except Exception as e:
            logger.error(f"Error catching generator result: {e}")
            raise

        assert rag_answer, "Generator result is empty"

        logger.info(f"✅ Final answer length: {len(rag_answer)} characters")

        return {
            "response": rag_answer,
            "retrieved_documents": retrieved_docs,
            "error_msg": "",
        }

    else:
        logger.error("❌ No rag output retrieved")
        raise


def print_result(result: Dict[str, Any]) -> None:
    if result["error_msg"]:
        print(f"Error: {result['error_msg']}")
        sys.exit(1)
    else:
        print("\n" + "=" * 50)
        print("RESPONSE:")
        print("=" * 50)
        # Convert escaped newlines to actual line breaks for better readability
        formatted_response = result["response"].replace("\\n", "\n")
        print(formatted_response)

        if result["retrieved_documents"]:
            print("\n" + "=" * 50)
            print("RETRIEVED DOCUMENTS:")
            print("=" * 50)
            for i, doc in enumerate(result["retrieved_documents"], 1):
                if isinstance(doc, DualVectorDocument):
                    print(
                        f"\n{i}. File: {getattr(doc.original_doc, 'meta_data', {}).get('file_path', 'Unknown')}"
                    )
                    print(
                        f"   Preview: {(doc.original_doc.text[:200] + '...') if len(doc.original_doc.text) > 200 else doc.original_doc.text}"
                    )
                else:
                    print(
                        f"\n{i}. File: {getattr(doc, 'meta_data', {}).get('file_path', 'Unknown')}"
                    )
                    print(
                        f"   Preview: {(doc.text[:200] + '...') if len(doc.text) > 200 else doc.text}"
                    )
        print("For more details, please check the reply folder")


@hydra.main(version_base=None, config_path="configs", config_name="main")
def hydra_wrapped_query_repository(cfg: DictConfig) -> None:

    assert cfg.repo_path, "repo_path must be set"
    # print("force_embedding:", configs()["rag"]["embedder"]["force_embedding"])
    load_all_configs(cfg)
    try:
        result = query_repository(repo_path=cfg.repo_path, question=cfg.question)

        if result is None:
            logger.warning("❌ No question provided, only embedding the repository")
            return

        # Save results to reply folder
        output_dir = save_query_results(result, cfg.repo_path, cfg.question)

        print_result(result)

        # Inform user about saved files
        print(f"\n📁 Results saved to: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    hydra_wrapped_query_repository()
