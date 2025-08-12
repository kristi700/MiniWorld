import torch

from models.tokenizer import VQ_VAE

def build_model(config):
    """
    Builds the appropriate model based on the configuration.
    This function acts as a single entry point for creating any model in the project.

    Args:
        config (OmegaConf): The global configuration object.

    Returns:
        torch.nn.Module: The constructed model, potentially with loaded weights.
    """

    mode = config.get("training", {}).get("type", None) or config.get("eval", {}).get(
        "mode", None
    )
    if mode is None:
        raise ValueError(
            "Could not determine mode. Set either 'training.type' or 'eval.mode' in config."
        )
    mode = mode.lower()

    image_shape = (
        config.model.in_channels,
        config.data.img_size,
        config.data.img_size,
    )

    model = None

    if mode in ["tokenizer"]:
        # TODO - configurable depth!
        model = VQ_VAE(
            image_shape = image_shape,
            embedding_dim=config.model.embedding_dim,
            codebook_size=config.model.codebook_size,
        )
    else:
        raise ValueError(f"Unknown model-building mode: {mode}")

    if hasattr(torch, "compile"):
        return torch.compile(model)
    return model