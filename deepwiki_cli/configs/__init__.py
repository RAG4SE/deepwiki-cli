from .compose_hydra import load_all_configs, load_default_config, configs

import adalflow as adal


def get_batch_embedder() -> adal.BatchEmbedder:
    embedder_config = configs()["rag"]["embedder"]
    model_client_class = embedder_config["model_client"]
    batch_model_client_class = embedder_config["batch_model_client"]
    model_kwargs = embedder_config[
        "model_kwargs"
    ].copy()  # Create a copy to avoid modifying original
    if "model" not in model_kwargs:
        assert "model" in embedder_config, "embedder_config must contain model"
        model_kwargs["model"] = embedder_config["model"]
    model_client = model_client_class(model_kwargs=model_kwargs)
    batch_model_client = batch_model_client_class(embedder=model_client, batch_size=embedder_config["batch_size"])
    return batch_model_client


def get_embedder() -> adal.Embedder:
    embedder_config = configs()["rag"]["embedder"]
    model_client_class = embedder_config["model_client"]
    model_kwargs = embedder_config[
        "model_kwargs"
    ].copy()  # Create a copy to avoid modifying original
    if "model" not in model_kwargs:
        assert "model" in embedder_config, "embedder_config must contain model"
        model_kwargs["model"] = embedder_config["model"]
    model_client = model_client_class(model_kwargs=model_kwargs)
    return model_client
