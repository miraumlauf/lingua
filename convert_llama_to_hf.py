from dataclasses import dataclass
import os
import logging
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
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
    print("Script is executing! Before loading State dict")
    # extracting weights and saving pytorch_model.bin
    file_path = Path("./apps/mtp/llama_babylm_lr_min/checkpoints/0000009000/consolidated/consolidated.pth")
    state_dict = torch.load(file_path)
    print("Script is executing! After loading State dict")
    ## saving pytorch_model.bin file of only the weights
    # print("State Dict Keys", state_dict.keys()) # model, optimizer
    # model_weights = state_dict["model"]
    # torch.save(model_weights, "./example_model/pytorch_model.bin")
    
    #torch.distributed.barrier()
    print("Script is executing! Getting Model weights")
    model_weights = state_dict["model"]
    print("Model Weights/State Dict keys: keys my model provides", model_weights.keys())

    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        rotary_embedding_base=10000.0,
        bos_token_id=128000,
        pad_token_id=128001, #setting pad token to eos token ID
        eos_token_id=128001,
        torch_dtype="bfloat16",
        model_type="llama"
    )
    
    print("Script is executing! Intializeing mdoel weights")

    # initialize llama model
    logger.info("initializing model and config")
    model = LlamaForCausalLM(config)
    model_keys = set(model.state_dict().keys())
    print("Keys the Llama model requires", model_keys)
    
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
    print("Missing keys (from the model that do not match the state_dict):", missing_keys) # keys from the state_dict that are not matching the model
    print("Unexpected keys(from the state_dict that do not match the llama model):", unexpected_keys) #keys from the model that do not match the state_dict

    # model.load_state_dict(model_weights, strict=False)

    # # save model locally
    # output_model_dir = Path(cfg.model_dir) / "llama_model"
    # model.save_pretrained(output_model_dir)
    # logger.info(f"Model saved to {output_model_dir}")
    
    # # push model to HF
    # repo_path = f"{cfg.username}/{cfg.repo_name}"
    # logger.info(f"Pushing model to Hugging Face Hub: {repo_path}")
    # model.push_to_hub(repo_path, use_auth_token=HUGGINGFACE_TOKEN)


    # load and save tokenizer

    # tokenizer_path = Path("tokenizers/llama3/original")
    
    # tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    # tokenizer.save_pretrained(output_model_dir)
    # tokenizer.push_to_hub(repo_path, use_auth_token=HUGGINGFACE_TOKEN)

    #print("Tokenizer finished")

    # # Push tokenizer to HF
    # logger.info("Pushing tokenizer to HF")
    # tokenizer_path = Path("./tokenizers/llama3/original")

    # hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # hf_tokenizer.push_to_hub(f"{cfg.username}/{cfg.repo_name}", private=True)
    # logger.info("Tokenizer pushed")
    



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
    print("Script is executing!")
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config  # Remove unsupported attribute

    default_cfg = OmegaConf.structured(UploadArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    upload(cfg)
    print("Script is executing!")
if __name__ == "__main__":
    main()
