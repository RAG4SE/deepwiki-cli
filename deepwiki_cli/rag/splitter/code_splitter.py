"""Smart text splitter that finds appropriate stopping points near chunk boundaries."""

from typing import List, Literal
from adalflow.components.data_process import TextSplitter
from adalflow.components.data_process.text_splitter import (
    DocumentSplitterInputType,
    DocumentSplitterOutputType,
)
from adalflow.core.types import Document
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from copy import deepcopy
from tqdm import tqdm

try:
    from tree_sitter import Language, Parser

    # Import language parsers in a more organized way
    LANGUAGE_MODULES = {}
    try:
        import tree_sitter_python as tspython

        LANGUAGE_MODULES["python"] = tspython
    except ImportError:
        pass

    try:
        import tree_sitter_javascript as tsjavascript

        LANGUAGE_MODULES["javascript"] = tsjavascript
        LANGUAGE_MODULES["typescript"] = tsjavascript  # TypeScript uses JS parser
    except ImportError:
        pass

    try:
        import tree_sitter_java as tsjava

        LANGUAGE_MODULES["java"] = tsjava
    except ImportError:
        pass

    try:
        import tree_sitter_cpp as tscpp

        LANGUAGE_MODULES["cpp"] = tscpp
        LANGUAGE_MODULES["c"] = tscpp  # C uses CPP parser
    except ImportError:
        pass

    try:
        import tree_sitter_go as tsgo

        LANGUAGE_MODULES["go"] = tsgo
    except ImportError:
        pass

    try:
        import tree_sitter_rust as tsrust

        LANGUAGE_MODULES["rust"] = tsrust
    except ImportError:
        pass

    try:
        import tree_sitter_markdown as tsmarkdown

        LANGUAGE_MODULES["markdown"] = tsmarkdown
    except ImportError:
        pass

    try:
        import tree_sitter_rst as tsrst

        LANGUAGE_MODULES["rst"] = tsrst
    except ImportError:
        pass

    try:
        import tree_sitter_yaml as tsyaml

        LANGUAGE_MODULES["yaml"] = tsyaml
        LANGUAGE_MODULES["yml"] = tsyaml  # Both .yaml and .yml extensions
    except ImportError:
        pass

    try:
        import tree_sitter_json as tsjson

        LANGUAGE_MODULES["json"] = tsjson
    except ImportError:
        pass

    TREE_SITTER_AVAILABLE = len(LANGUAGE_MODULES) > 0
except ImportError:
    TREE_SITTER_AVAILABLE = False
    LANGUAGE_MODULES = {}

logger = get_tqdm_compatible_logger(__name__)


class CodeSplitter(TextSplitter):
    __doc__ = r"""Enhanced TextSplitter that finds intelligent stopping points near chunk boundaries for code.
    This class provides intelligent code splitting functionality by finding appropriate stopping points near chunk boundaries.
    It uses tree-sitter for parsing code and finding statement-level boundaries to avoid splitting in the middle of code constructs.

    Key features:
    - Smart boundary detection using tree-sitter parsing
    - Language-specific statement recognition
    - Adjustable chunk sizes and overlap
    - Fallback mechanisms when tree-sitter is not available
    
    The splitter supports multiple programming languages including:
    - Python (.py)
    - JavaScript/TypeScript (.js/.ts) 
    - Java (.java)
    - C/C++ (.cpp/.c)
    - Go (.go)
    - Rust (.rs)
    - Markdown (.md/.markdown)
    - reStructuredText (.rst)
    - YAML (.yaml/.yml)
    - JSON (.json)

    Args:
        split_by (str): Splitting method - "word", "sentence", "page", "passage", or "token"
        chunk_size (int): Target size for each chunk in tokens/words
        chunk_overlap (int): Number of tokens/words to overlap between chunks
        batch_size (int): Number of documents to process at once
        separators (dict): Custom separators for different splitting methods
        smart_boundary_ratio (float): When to start looking for smart boundaries (0.8 = 80% of chunk_size)
        file_extension (str): File extension to determine the programming language
    """

    # Language mappings for tree-sitter
    SUFFIX_TO_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "rst",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
    }

    # Dynamically determine supported languages based on available modules
    @property
    def supported_languages(self):
        return set(LANGUAGE_MODULES.keys()) if TREE_SITTER_AVAILABLE else set()

    def __init__(
        self,
        split_by: Literal["word", "sentence", "page", "passage", "token"] = "word",
        chunk_size: int = 1024,
        chunk_overlap: int = 64,
        batch_size: int = 1000,
        smart_boundary_ratio: float = 0.8,
        file_extension: str = None,
    ):
        """
        Initialize CodeSplitter.

        Args:
            split_by: Same as TextSplitter
            chunk_size: Same as TextSplitter
            chunk_overlap: Same as TextSplitter
            batch_size: Same as TextSplitter
            smart_boundary_ratio: When to start looking for smart boundaries (0.8 = 80% of chunk_size)
            file_extension: The file extension to use for parser selection
        """

        # TODO: if file_extension is None, determine from document content or metadata
        assert file_extension is not None, "File extension must be provided"
        assert (
            file_extension in self.SUFFIX_TO_LANG
        ), f"Unsupported file extension {file_extension}"

        assert (
            split_by == "word" or split_by == "token"
        ), f"CodeSplitter only supports split_by='word' or split_by='token', but got {split_by}"

        super().__init__(split_by, chunk_size, chunk_overlap, batch_size)
        self.smart_boundary_ratio = smart_boundary_ratio

        self.file_extension = file_extension
        self.smart_boundary_threshold = int(chunk_size * smart_boundary_ratio)

        # Initialize tree-sitter parsers
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._init_parsers()
        else:
            logger.warning(
                "Tree-sitter not available. Code splitting will use fallback method."
            )

    def get_key(self) -> str:
        """Generate a unique key for this splitter configuration.

        Returns:
            str: Unique key representing this splitter's configuration
        """
        key_params = [
            f"split_by:{self.split_by}",
            f"batch_size:{self.batch_size}",
            f"separator:{self.separators[self.split_by]}",
            f"chunk_size:{self.chunk_size}",
            f"chunk_overlap:{self.chunk_overlap}",
            f"smart_boundary_ratio:{self.smart_boundary_ratio}",
            f"file_extension:{self.file_extension}",
        ]
        return f"{self.__class__.__name__}({','.join(key_params)})"

    def __getstate__(self):
        """Exclude non-serializable attributes from pickling."""
        state = self.__dict__.copy()
        # Remove the non-serializable tree_sitter.Parser objects
        if "parsers" in state:
            del state["parsers"]
        if "parser" in state:
            del state["parser"]
        # Remove the non-serializable tree_sitter.Tree object
        if "tree" in state:
            del state["tree"]
        return state

    def __setstate__(self, state):
        """Re-initialize non-serializable attributes after unpickling."""
        self.__dict__.update(state)
        # Re-initialize the tree_sitter.Parser objects
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._init_parsers()

    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning(
                "Tree-sitter not available. Code splitting will use fallback method."
            )
            return

        if not self.file_extension:
            logger.warning(
                "No file extension provided. Code splitting will use fallback method."
            )
            return

        lang_name = self.SUFFIX_TO_LANG.get(self.file_extension)
        if not lang_name or lang_name not in self.supported_languages:
            logger.warning(
                f"Unsupported file extension {self.file_extension} for tree-sitter parsing, using fallback method for code split."
            )
            return

        try:
            # Get the language module and create parser
            lang_module = LANGUAGE_MODULES[lang_name]
            language = Language(lang_module.language())
            parser = Parser(language)
            self.parsers[lang_name] = parser
            logger.info(f"Initialized tree-sitter parser for {lang_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize parser for {lang_name}: {e}")

    def _is_statement_node(self, node, language: str) -> bool:
        """
        Check if a tree-sitter node represents a complete statement.
        This ensures we don't split at partial code elements like single parentheses.
        """
        # Define statement-level node types for different languages
        statement_types = {
            "python": {
                "expression_statement",
                "function_definition",
                "class_definition",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "with_statement",
                "import_statement",
                "import_from_statement",
                "return_statement",
                "break_statement",
                "continue_statement",
                "pass_statement",
                "del_statement",
                "raise_statement",
                "assert_statement",
                "global_statement",
                "nonlocal_statement",
                "decorated_definition",
                "match_statement",
            },
            "javascript": {
                "expression_statement",
                "function_declaration",
                "class_declaration",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "return_statement",
                "break_statement",
                "continue_statement",
                "throw_statement",
                "switch_statement",
                "variable_declaration",
                "import_statement",
                "export_statement",
            },
            "java": {
                "expression_statement",
                "method_declaration",
                "class_declaration",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "return_statement",
                "break_statement",
                "continue_statement",
                "throw_statement",
                "switch_statement",
                "local_variable_declaration",
                "import_declaration",
                "package_declaration",
            },
            "cpp": {
                "expression_statement",
                "function_definition",
                "class_specifier",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "return_statement",
                "break_statement",
                "continue_statement",
                "throw_statement",
                "switch_statement",
                "declaration",
            },
            "go": {
                "expression_statement",
                "function_declaration",
                "type_declaration",
                "if_statement",
                "for_statement",
                "switch_statement",
                "return_statement",
                "break_statement",
                "continue_statement",
                "go_statement",
                "defer_statement",
                "var_declaration",
                "const_declaration",
                "import_declaration",
                "package_clause",
            },
            "rust": {
                "expression_statement",
                "function_item",
                "struct_item",
                "if_expression",
                "loop_expression",
                "while_expression",
                "for_expression",
                "match_expression",
                "return_expression",
                "break_expression",
                "continue_expression",
                "let_declaration",
                "use_declaration",
                "mod_item",
                "impl_item",
            },
            "markdown": {
                "atx_heading",
                "setext_heading",
                "paragraph",
                "code_block",
                "fenced_code_block",
                "list_item",
                "block_quote",
                "thematic_break",
                "html_block",
                "link_reference_definition",
            },
            "rst": {
                "section",
                "title",
                "paragraph",
                "literal_block",
                "code_block",
                "block_quote",
                "bullet_list",
                "enumerated_list",
                "list_item",
                "definition_list",
                "definition_list_item",
                "field_list",
                "field",
                "option_list",
                "option_list_item",
                "doctest_block",
                "line_block",
                "table",
                "directive",
                "comment",
                "substitution_definition",
                "target",
                "transition",
            },
            "yaml": {
                "document",
                "block_mapping",
                "block_sequence",
                "flow_mapping",
                "flow_sequence",
                "block_mapping_pair",
                "block_sequence_item",
                "flow_mapping_pair",
                "flow_sequence_item",
                "plain_scalar",
                "single_quote_scalar",
                "double_quote_scalar",
                "literal_scalar",
                "folded_scalar",
                "alias_node",
                "anchor_node",
                "tag",
                "directive",
                "comment",
            },
            "json": {
                "document",
                "object",
                "array",
                "pair",
                "string",
                "number",
                "true",
                "false",
                "null",
            },
        }

        return node.type in statement_types.get(language, set())

    def _find_next_code_boundary_with_treesitter(
        self, text: bytes, start_pos: int, max_pos: int
    ) -> int:
        """
        Given a utf-8 encoded text, find the best code boundary using tree-sitter.
        The "best" boundary is the one that is closest to max_pos, but not less than start_pos.
        Only considers statement-level nodes to ensure complete code semantics.
        """
        assert isinstance(text, bytes), "text must be utf-8 encoded bytes"
        assert self.language, "language must be set"
        assert self.tree, "tree must be set"

        try:

            # Find nodes that end near our target position
            best_boundary = start_pos

            def find_good_boundary(node):
                nonlocal best_boundary

                # Only consider statement-level nodes for boundaries
                if self._is_statement_node(node, self.language):
                    node_end = node.end_byte
                    # Check if this node ends in our target range
                    if start_pos <= node_end <= max_pos:
                        best_boundary = max(best_boundary, node_end)

                # Recursively check children
                for child in node.children:
                    find_good_boundary(child)

            find_good_boundary(self.tree.root_node)
            return min(best_boundary, max_pos)

        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {self.language}: {e}")
            raise

    def _find_prev_code_boundary_with_treesitter(
        self, text: bytes, desired_start_char: int
    ):
        """
        Given a utf-8 encoded text, find the best code boundary using tree-sitter.
        The "best" boundary is the one that is closest to desired_start_char, but not less than desired_start_char.
        The range of the boundary is [desired_start_char, len(text)].
        """
        assert isinstance(text, bytes), "text must be utf-8 encoded bytes"
        assert self.language is not None, "language must be set"
        assert self.tree is not None, "tree must be set"
        try:
            best_start = len(
                text
            )  # default to the end of the text, meaning zero overlap

            def find_best_start_recursive(node):
                nonlocal best_start
                # Check if the current node is a candidate and is a statement
                if node.start_byte >= desired_start_char and self._is_statement_node(
                    node, self.language
                ):
                    best_start = min(best_start, node.start_byte)

                # Recursively check children only if they can potentially contain a better candidate.
                if node.start_byte < best_start:
                    for child in node.children:
                        find_best_start_recursive(child)

            find_best_start_recursive(self.tree.root_node)
            return best_start

        except Exception as e:
            logger.error(
                f"Tree-sitter parsing for overlap failed for {self.language}: {e}"
            )
            raise

    def _merge_units_to_chunks_by_words(
        self, splits: List[str], chunk_size: int, chunk_overlap: int, separator: str
    ):
        chunks = []
        idx = 0

        while idx < len(splits):
            # Calculate the end position for this chunk
            chunk_end = min(idx + chunk_size, len(splits))
            prev_chunk_bytes = (
                separator.join(splits[:idx]).encode("utf-8", errors="ignore") if idx > 0 else b""
            )
            chunk_text = separator.join(splits[idx:chunk_end])
            chunk_text_bytes = chunk_text.encode("utf-8", errors="ignore")
            # If this is not the last chunk and we have room for smart boundary detection
            if chunk_end < len(splits):
                # Find smart boundary in the last portion of the chunk
                search_start_pos = int(
                    len(chunk_text_bytes) * self.smart_boundary_ratio
                )
                smart_boundary_pos = self._find_next_code_boundary_with_treesitter(
                    self.bytes,
                    search_start_pos + len(prev_chunk_bytes),
                    len(chunk_text_bytes) + len(prev_chunk_bytes),
                )
                smart_boundary_pos -= len(prev_chunk_bytes)
                # If we found a good boundary, adjust the chunk end
                if smart_boundary_pos < len(chunk_text_bytes):
                    # Re-encode to find the token boundary
                    boundary_text = chunk_text_bytes[:smart_boundary_pos].decode(
                        "utf-8"
                    )
                    chunk_end = idx + len(boundary_text.split(separator))
                    # Update chunk_text_bytes to reflect the boundary adjustment
                    chunk_text_bytes = boundary_text.encode("utf-8", errors="ignore")

                    logger.debug(
                        f"Found smart boundary at position {smart_boundary_pos} (token {chunk_end})"
                    )

            if chunk_text_bytes.strip():  # Only add non-empty chunks
                last_line_break = prev_chunk_bytes.rfind(b"\n")
                if last_line_break != -1:
                    chunk_text = (
                        b" " * (len(prev_chunk_bytes) - last_line_break)
                        + chunk_text_bytes
                    ).decode("utf-8")
                else:
                    chunk_text = chunk_text_bytes.decode("utf-8")
                chunks.append(chunk_text)

            # Smart overlap calculation
            # Use the actual chunk tokens that were used (may be adjusted by boundary detection)
            actual_chunk_words = chunk_text_bytes.decode("utf-8").split(separator)
            num_non_overlap_words = len(actual_chunk_words) - chunk_overlap

            if num_non_overlap_words > 0 and chunk_end < len(splits):
                bytetext_before_overlap = separator.join(
                    actual_chunk_words[:num_non_overlap_words]
                ).encode("utf-8", errors="ignore")
                desired_start_byte = len(bytetext_before_overlap)
                best_start_byte = self._find_prev_code_boundary_with_treesitter(
                    self.bytes, desired_start_byte + len(prev_chunk_bytes)
                )
                best_start_byte -= len(prev_chunk_bytes)

                if best_start_byte < len(chunk_text_bytes):
                    new_overlap_text = chunk_text_bytes[best_start_byte:].decode(
                        "utf-8"
                    )
                    new_overlap_words_count = len(new_overlap_text.split(separator))
                    adjusted_overlap = new_overlap_words_count

                    # Find the token that starts with the overlap text
                    next_idx = chunk_end - adjusted_overlap  # fallback

                else:
                    adjusted_overlap = 0
                    next_idx = chunk_end - adjusted_overlap
            else:
                adjusted_overlap = 0
                next_idx = chunk_end - adjusted_overlap
            assert (
                next_idx > idx
            ), f"next_idx ({next_idx}) must be greater than idx ({idx})"
            idx = next_idx

        return chunks

    def _merge_units_to_chunks_by_tokens(
        self, splits: List[str], chunk_size: int, chunk_overlap: int, separator: str
    ) -> List[str]:
        """Enhanced merge method with smart boundary detection."""

        chunks = []
        idx = 0

        while idx < len(splits):
            # Calculate the end position for this chunk
            chunk_end = min(idx + chunk_size, len(splits))
            chunk_tokens = splits[idx:chunk_end]
            prev_chunk_bytes = (
                self.tokenizer.decode(splits[:idx]).encode("utf-8", errors="ignore") if idx > 0 else b""
            )
            chunk_text_bytes = self.tokenizer.decode(chunk_tokens).encode("utf-8", errors="ignore")
            # If this is not the last chunk and we have room for smart boundary detection
            if (
                chunk_end < len(splits)
                and chunk_end - idx >= self.smart_boundary_threshold
            ):
                # Find smart boundary in the last portion of the chunk
                search_start_pos = int(
                    len(chunk_text_bytes) * self.smart_boundary_ratio
                )
                smart_boundary_pos = self._find_next_code_boundary_with_treesitter(
                    self.bytes,
                    search_start_pos + len(prev_chunk_bytes),
                    len(chunk_text_bytes) + len(prev_chunk_bytes),
                )
                smart_boundary_pos -= len(prev_chunk_bytes)

                # If we found a good boundary, adjust the chunk end
                if smart_boundary_pos < len(chunk_text_bytes):
                    # Re-encode to find the token boundary
                    boundary_text = chunk_text_bytes[:smart_boundary_pos].decode(
                        "utf-8"
                    )
                    boundary_tokens = self.tokenizer.encode(boundary_text)
                    chunk_end = idx + len(boundary_tokens)
                    # Update chunk_text_bytes to reflect the boundary adjustment
                    chunk_text_bytes = boundary_text.encode("utf-8", errors="ignore")

                    logger.debug(
                        f"Found smart boundary at position {smart_boundary_pos} (token {chunk_end})"
                    )

            if chunk_text_bytes.strip():  # Only add non-empty chunks
                chunk_text = chunk_text_bytes.decode("utf-8")
                chunks.append(chunk_text)

            # Smart overlap calculation
            # Use the actual chunk tokens that were used (may be adjusted by boundary detection)
            actual_chunk_tokens = self.tokenizer.encode(
                chunk_text_bytes.decode("utf-8")
            )
            num_non_overlap_tokens = len(actual_chunk_tokens) - chunk_overlap

            if num_non_overlap_tokens > 0 and chunk_end < len(splits):
                bytetext_before_overlap = self.tokenizer.decode(
                    actual_chunk_tokens[:num_non_overlap_tokens]
                ).encode("utf-8", errors="ignore")
                desired_start_byte = len(bytetext_before_overlap)

                best_start_byte = self._find_prev_code_boundary_with_treesitter(
                    self.bytes, desired_start_byte + len(prev_chunk_bytes)
                )
                best_start_byte -= len(prev_chunk_bytes)
                if best_start_byte < len(chunk_text_bytes):
                    new_overlap_text = chunk_text_bytes[best_start_byte:].decode(
                        "utf-8"
                    )
                    new_overlap_tokens_count = len(
                        self.tokenizer.encode(new_overlap_text)
                    )
                    adjusted_overlap = new_overlap_tokens_count

                    # Calculate next_idx based on the smart overlap position
                    # Find the token position in the original splits that corresponds to the overlap start
                    overlap_start_text = chunk_text_bytes[best_start_byte:].decode(
                        "utf-8"
                    )

                    # Find the token that starts with the overlap text
                    next_idx = chunk_end - adjusted_overlap  # fallback

                    # Search through tokens to find the one that matches the overlap start
                    for token_idx in range(idx, min(len(splits), chunk_end + 5)):
                        # Decode from this token to the end to see if it matches our overlap start
                        remaining_tokens = splits[token_idx:]
                        remaining_text = self.tokenizer.decode(remaining_tokens)

                        # Check if this token position gives us the overlap text we want
                        # Strip leading whitespace for comparison since tokenizer may add spaces
                        if remaining_text.lstrip().startswith(
                            overlap_start_text.lstrip()[
                                : min(50, len(overlap_start_text.lstrip()))
                            ]
                        ):
                            next_idx = token_idx
                            break
                else:
                    adjusted_overlap = 0
                    next_idx = chunk_end - adjusted_overlap
            else:
                adjusted_overlap = 0
                next_idx = chunk_end - adjusted_overlap
            assert (
                next_idx > idx
            ), f"next_idx ({next_idx}) must be greater than idx ({idx})"
            idx = next_idx

        logger.info(
            f"Smart splitting created {len(chunks)} chunks from {len(splits)} tokens"
        )
        return chunks

    def _merge_units_to_chunks(
        self, splits: List[str], chunk_size: int, chunk_overlap: int, separator: str
    ):
        """Enhanced merge method with smart boundary detection."""
        if self.split_by == "word":
            return self._merge_units_to_chunks_by_words(
                splits, chunk_size, chunk_overlap, separator
            )
        if self.split_by == "token":
            # For non-token splitting, use the original method
            return self._merge_units_to_chunks_by_tokens(
                splits, chunk_size, chunk_overlap, separator
            )

        return super()._merge_units_to_chunks(
            splits, chunk_size, chunk_overlap, separator
        )

    def split_text(self, text: str) -> List[str]:
        """
        Initialize parser
        Split the text into chunks using the smart boundary detection.
        """
        self.language = self.SUFFIX_TO_LANG[self.file_extension]
        self.text = text
        self.bytes = text.encode("utf-8", errors="ignore")
        self.parser = self.parsers[self.language]
        self.tree = self.parser.parse(self.bytes)

        return super().split_text(text)

    def call(self, documents: DocumentSplitterInputType) -> DocumentSplitterOutputType:
        """
        Process the splitting task on a list of documents in batch.

        Batch processes a list of documents, splitting each document's text according to the configured
        split_by, chunk size, and chunk overlap.

        Compared with TextSplitter.call, this method remvoes redundant logging and handles errors in split_text.

        Args:
            documents (List[Document]): A list of Document objects to process.

        Returns:
            List[Document]: A list of new Document objects, each containing a chunk of text from the original documents.

        Raises:
            TypeError: If 'documents' is not a list or contains non-Document objects.
            ValueError: If any document's text is None.
        """

        if not isinstance(documents, list) or any(
            not isinstance(doc, Document) for doc in documents
        ):
            raise TypeError("Input should be a list of Documents.")

        split_docs = []
        # Using range and batch_size to create batches
        for start_idx in tqdm(
            range(0, len(documents), self.batch_size),
            desc="Splitting Documents in Batches",
        ):
            batch_docs = documents[start_idx : start_idx + self.batch_size]

            for doc in batch_docs:
                if not isinstance(doc, Document):
                    raise TypeError(
                        f"Each item in documents should be an instance of Document, but got {type(doc).__name__}."
                    )

                if doc.text is None:
                    raise ValueError(f"Text should not be None. Doc id: {doc.id}")

                try:
                    text_splits = self.split_text(doc.text)
                except Exception as e:
                    logger.warning(f"Error splitting document {doc.id}: {e}")
                    continue

                meta_data = deepcopy(doc.meta_data)

                split_docs.extend(
                    [
                        Document(
                            text=txt,
                            meta_data=meta_data,
                            parent_doc_id=f"{doc.id}",
                            order=i,
                            vector=[],
                        )
                        for i, txt in enumerate(text_splits)
                    ]
                )
        return split_docs
