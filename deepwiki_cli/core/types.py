from dataclasses import dataclass, field
from adalflow.core.types import Document, List


@dataclass
class DualVectorDocument:
    """
    Data structure to store the code, its summary, and the corresponding dual vector embeddings.
    """

    original_doc: Document = None
    code_embedding: List[float] = field(default_factory=list)
    understanding_text: str = ""
    understanding_embedding: List[float] = field(default_factory=list)
    file_path: str = ""
    chunk_id: str = ""
