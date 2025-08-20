# DeepWiki CLI

Repository Analysis and Query CLI Tool with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Repository Analysis**: Analyze code repositories and extract meaningful information
- **RAG-powered Q&A**: Ask questions about your codebase and get intelligent answers
- **Dual Usage**: Use as a Python library or command-line tool
- **Configurable**: Flexible configuration system using Hydra
- **Advanced Retrieval**: Supports dual-vector embedding (code + semantic interpretation) and hybrid search (BM25 + FAISS) for enhanced precision

## Installation

### From PyPI

```bash
pip install deepwiki_cli
```

### From Source

```bash
git clone git@github.com:RAG4SE/deepwiki-cli.git
cd deepwiki-cli
pip install .

```

## A Quick Start

### Set up API keys of LLM providers

DeepWiki-cli requires API keys from LLM providers to function properly. The tool has been thoroughly tested with the following model configurations:

**Embedding Models:**
- Huggingface: `intfloat/multilingual-e5-large-instruct`
- Dashscope: `text-embedding-v4`

**Question Answering Models:**
- Dashscope: `qwen-plus`, `qwen-max`, `qwen-turbo`
- Google: `gemini-2.5-flash-preview-05-20`

**Code Understanding Model (for dual-vector embedding):**
- Dashscope: `qwen3-32b`

> **Note**: Support for additional mainstream models and providers is planned for future releases.

You can refer to the following coarse-grained table to get the corresponding API keys and store them into your env.

| Provider | Required Environment Variables | How to Get API Key |
|----------|-------------------------------|-------------------|
| **Google Gemini** | `GOOGLE_API_KEY` | [Get API Key](https://aistudio.google.com/app/apikey) |
| **OpenAI** | `OPENAI_API_KEY` | [Get API Key](https://platform.openai.com/api-keys) |
| **DeepSeek** | `DEEPSEEK_API_KEY` | [Get API Key](https://platform.deepseek.com/api_keys) |
| **Dashscope (Qwen)** | `DASHSCOPE_API_KEY`, `DASHSCOPE_WORKSPACE_ID` (optional) | [Get API Key and WorkSpace](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.6ebe48238qeoit&tab=api#/api)  |
| **SiliconFlow** | `SILICONFLOW_API_KEY` | [Get API Key](https://cloud.siliconflow.cn/i/api-keys) |

### As a Command-Line Tool

After installation, you can use the `deepwiki` command to query about a local codebase:

```bash
deepwiki repo_path=/path/to/repository question="What does this project do?"
```

Or, you can only embed the codebase with the following command:

```bash
deepwiki repo_path=/path/to/repository
```

**Important Note**: If you want to exclude additional folders from the embedding process, avoid specifying them directly like `repo.file_filters.extra_excluded_patterns=["reply/"]`. 
Shell quoting may cause Hydra to not recognize this pattern correctly. Instead, use the following format:

```bash
deepwiki repo_path=/path/to/repository question="What does this project do?" 'repo.file_filters.extra_excluded_patterns=[pattern1, pattern2]'
```

### As a Python Library

```python
from deepwiki_cli import *
from deepwiki_cli import configs

repo_path = "/path/to/repository"
question = "What does this project do?"

# Basic usage
result = query_repository(
    repo_path=repo_path,
    question=question
)

print_result(result)
save_query_results(result, repo_path, question)

# Load default DictConfig
dict_config = load_default_config()
# Add custom config
# Use qwen-plus instead of the default qwen-coder to reply
dict_config.generator.model = "qwen-plus"
# Disable the use of bm25
dict_config.rag.hybrid.enabled = False
# Disable semantic interpretation in dual-vector embedding, only embed code snippets
dict_config.rag.embedder.sketch_filling = False
# Modify the global dict-type configs
configs.configs = load_all_configs(dict_config)

result = query_repository(
    repo_path=repo_path,
    question=question,
)

print_result(result)
save_query_results(result, repo_path, question)
```

Similar to the second use of `deepwiki` command, you can also use deepwiki_cli library to only embed to codebase

```python
from deepwiki_cli import *

repo_path = "/path/to/repository"

analyze_repository(repo_path=repo_path)

```

## Acknowledgements

[deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open).