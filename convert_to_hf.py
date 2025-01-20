from dataclasses import dataclass
import os
import logging
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig
from typing import Optional
from omegaconf import OmegaConf
import torch
from apps.main.generate import load_consolidated_model_and_tokenizer
from apps.mtp.transformer import LMTransformer, LMMTPArgs
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    DistributedArgs,
    get_global_rank,
    setup_torch_distributed,
)
from huggingface_hub import login

# Directly pass your Hugging Face token
HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()

@dataclass
class UploadArgs:
    username: str = ""
    repo_name: str = ""

    name: str = "upload"
    dump_dir: Optional[str] = None
    ckpt_dir: str = ""
    model_dir: str = ""


class CustomModelConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(self, hidden_size=256, num_attention_heads=8, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers

def save_config(output_dir: str, **kwargs):
    """Creates and saves HF File"""
    config = CustomModelConfig(**kwargs)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(output_dir) / "config.json"
    config.save_pretrained(output_dir)
    logger.info(f"Configuration saved at {config_path}")
    return config

class LinguaModelHub(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def upload(cfg: UploadArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())

    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    torch.distributed.barrier()
    logger.info("Loading model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        model_args_cls=LMMTPArgs,
    )
    logger.info("Model loaded")

    # Save Hugging Face-compatible configuration
    config = save_config(
        output_dir=cfg.model_dir,
        hidden_size=train_cfg.model.hidden_size,
        num_attention_heads=train_cfg.model.num_attention_heads,
        num_layers=train_cfg.model.num_layers,
    )

    # Pushing model to HF
    lingua_model = LinguaModelHub(model)
    lingua_model.push_to_hub(f"{cfg.username}/{cfg.repo_name}", private=True)
    logger.info("Model pushed")
    
    # Push tokenizer to HF
    logger.info("Pushing tokenizer to HF")
    tokenizer_path = Path("./tokenizers/llama3/original")

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    hf_tokenizer.push_to_hub(f"{cfg.username}/{cfg.repo_name}", private=True)
    logger.info("Tokenizer pushed")

def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    For example, if the config is the following
    model:
        dim: 128
        n_layers: 4
    and you call upload.py with upload.py model.dim=64
    Then the final UploadArgs will have
    model:
        dim: 64
        n_layers: 4
    Plus all the default values in UploadArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config  # Remove unsupported attribute

    default_cfg = OmegaConf.structured(UploadArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    upload(cfg)

if __name__ == "__main__":
    main()
