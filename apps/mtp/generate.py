# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import argparse

from lm_eval import simple_evaluate

from omegaconf import OmegaConf
import torch
# apps imports from main
from apps.main.eval import (
    EvalArgs,
    EvalHarnessLM,
)
from apps.main.generate import (
    PackedCausalTransformerGenerator,
    load_consolidated_model_and_tokenizer,
)
# app imports from mtp.transformer
from apps.mtp.transformer import LMTransformer, LMMTPArgs

# try with ntp  model
from apps.main.transformer import LMTransformer, LMTransformerArgs

# lingua imports (modules)
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import DistributedArgs, get_global_rank, setup_torch_distributed


# template for an eval folder name : -> zeros, 10 numbers, d -> integer
EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


# takes Eval Args from main.eval (-> config) -> EvalArgs HAS TO BE MODIFIED
def launch_eval(cfg: EvalArgs): # cfg = config
    
    # ensures PyTorch distributed computing is initialized and consolidates if necessary
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


    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True) # ensures dump_dir exits
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False) #saves eval in dir

    consolidate_path = str(consolidate_path)
    torch.distributed.barrier()
    logger.info("Loading model")
    # Loading Model and Tokenizer from consolidated checkpoint
    # imported from MTP Transformer
    
    # For NTP prediction change to args to LMTransformerArgs
    model, tokenizer= load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        model_args_cls=LMMTPArgs,
    )
    logger.info("Model and tokenizer loaded")
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    model.eval() # set model in evaluation mode (disables dropout layers and prepares the model for inference)
    generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)
   
    # Check if prompts are provided
    if cfg.single_prompts:  # Assuming cfg.single_prompts is a list of strings
        prompts = cfg.single_prompts  # Use the list of prompts directly
        logger.info(f"Generating text for prompts: {prompts}")
        
        # Generate outputs for the batch of prompts
        outputs, _, _ = generator.generate(prompts)  
        
        # Log and print outputs for each prompt
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            logger.info(f"Prompt {i + 1}: {prompt}")
            logger.info(f"Generated Output {i + 1}: {output}")
            print(f"Prompt {i + 1}: {prompt}")
            print(f"Generated Output {i + 1}: {output}")
        
    else:
        print("No single prompt provided. Exiting launch_eval...")
        
    print ("Finishing launch_eval...")
    
    del generator
    


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        mode: LMMambaArg

    @dataclass
    class LMMTPArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMMTPArgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    print("Configuration", OmegaConf.to_yaml(cfg))
    launch_eval(cfg)


if __name__ == "__main__":
    main()
