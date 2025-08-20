#!/usr/bin/env python3
"""
Simple example of loading all YAML configurations from the configs folder using Hydra.
This example works with the existing YAML files without requiring dataclass definitions.
"""

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

#! Though the following imports are not directly used, they are stored in globals() and will be used implicitly. So DO NOT REMOVE THEM!!
from deepwiki_cli.clients import *
from adalflow import GoogleGenAIClient
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

global_configs = None
# Setup logging
logger = get_tqdm_compatible_logger(__name__)

def load_default_config() -> DictConfig:
    """
    Load all YAML configurations programmatically using Hydra's compose API.
    """
    with initialize(config_path=".", version_base=None):
        return compose(config_name="main")

def load_all_configs(cfg: DictConfig = None):
    if cfg is None:
        cfg = load_default_config()
    all_configs = OmegaConf.to_container(cfg, resolve=True)
    load_generator_config(all_configs)
    load_rag_config(all_configs)
    global global_configs
    global_configs = all_configs

def configs():
    if global_configs is None:
        raise ValueError("May use global_configs before loading all configs. Probably deepwiki_cli has incorrect initialization.")
    return global_configs


PROVIDER_NAME_TO_CLASS = {
    "google": GoogleGenAIClient,
    "dashscope": DashScopeClient,
}


def load_generator_config(configs: dict):
    # Add client classes to each provider
    assert (
        "provider" in configs["generator"]
    ), "generator config must contain 'provider'"
    assert "model" in configs["generator"], "generator config must contain 'model'"
    assert (
        "model_kwargs" in configs["generator"]
    ), "generator config must contain 'model_kwargs'"
    assert (
        "temperature" in configs["generator"]["model_kwargs"]
    ), "generator config must contain 'temperature'"
    assert (
        "top_p" in configs["generator"]["model_kwargs"]
    ), "generator config must contain 'top_p'"

    configs["generator"]["model_client"] = PROVIDER_NAME_TO_CLASS[
        configs["generator"]["provider"]
    ]

def load_rag_config(configs: dict):
    # Process client classes
    embedder_provider = configs["rag"]["embedder"]["provider"]
    if embedder_provider == "dashscope":
        embedder_class_name = "DashScopeEmbedder"
        batch_embedder_class_name = "DashScopeBatchEmbedder"
    elif embedder_provider == "huggingface":
        embedder_class_name = "HuggingfaceEmbedder"
        batch_embedder_class_name = "HuggingfaceBatchEmbedder"
    else:
        raise ValueError(f"Unknown provider: {embedder_provider}")
    assert (
        batch_embedder_class_name in globals() and
        embedder_class_name in globals()
    ), f"load_rag_config: {batch_embedder_class_name} or {embedder_class_name} not in globals()  {globals()}"
    configs["rag"]["embedder"]["model_client"] = globals()[embedder_class_name]
    configs["rag"]["embedder"]["batch_model_client"] = globals()[batch_embedder_class_name]
    code_understanding_config = configs["rag"]["code_understanding"]
    # Get client class
    code_understanding_provider = code_understanding_config["provider"]
    if code_understanding_provider == "dashscope":
        class_name = "DashScopeClient"
    elif code_understanding_provider == "huggingface":
        class_name = "HuggingfaceClient"
    else:
        raise ValueError(f"Unknown provider: {code_understanding_provider}")
    # Map client class to actual class
    model_client = globals().get(class_name)
    if not model_client:
        raise ValueError(f"Unknown client class: {class_name}")

    # Set model client
    code_understanding_config["model_client"] = model_client


def main():
    """
    Main function demonstrating different ways to load all configs.
    """
    print("Hydra All Configs Example")
    print("=" * 50)

    try:
        # Load all configurations
        all_configs = load_all_configs()
        print(all_configs)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
