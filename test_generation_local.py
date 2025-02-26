from dataclasses import dataclass
import os
import logging
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig, AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
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
from lingua_config import LinguaModelConfig
from lingua_model import LinguaModelHub
from huggingface_hub import login
import torch

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


@dataclass
class LoadArgs:
    username: str = ""
    repo_name: str = ""

    name: str = "upload"
    dump_dir: Optional[str] = None
    ckpt_dir: str = ""
    model_dir: str = ""

def load_model(cfg: LoadArgs):
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
    model, tokenizer = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        model_args_cls=LMMTPArgs,
    )
    return model
    
AutoConfig.register("lingua_model", LinguaModelConfig)  
# model_type, config class
AutoModel.register(LinguaModelConfig, LinguaModelHub) 
# config class, model class



def main():
    # AutoConfig.register("lingua_model", LinguaModelConfig)  
    # AutoModel.register(LinguaModelConfig, LinguaModelHub)
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config  # Remove unsupported attribute

    default_cfg = OmegaConf.structured(LoadArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    
    # catch model
    original_model = load_model(cfg)
    
    # create config
    base_path = "umlauf/llama_try"
    config = AutoConfig.from_pretrained(base_path)
    # Load from Config or form pre_trained??? -> from config to load not initialized weights
    # model = AutoModel.from_config(config)
    model = AutoModel.from_config(config)
    model.set_model(original_model)
    print("model", model)
    
if __name__ == "__main__":
    main()
    