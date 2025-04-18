from dataclasses import dataclass
import os
import logging
from pathlib import Path
import shutil
from huggingface_hub import login, HfApi 

from transformers import PretrainedConfig, AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, AutoModelForCausalLM
from typing import Optional
from omegaconf import OmegaConf
import torch
from apps.main.generate import load_consolidated_model_and_tokenizer
from apps.mtp.transformer import LMTransformer, LMMTPArgs
#from apps.main.transformer import LMTransformer, LMTransformerArgs

from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    DistributedArgs,
    get_global_rank,
    setup_torch_distributed,
)
# import files 
from huggingface.lingua_config import LinguaModelConfig
#from huggingface.lingua_model import LinguaModel
from huggingface.lingua_model import LinguaModelForCausalLM



# registering to AutoClass for local development
AutoConfig.register("lingua_model", LinguaModelConfig)  
# model_type, config class
#AutoModel.register(LinguaModelConfig, LinguaModel) 
# config class, model class
AutoModelForCausalLM.register(LinguaModelConfig, LinguaModelForCausalLM)


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


# here also try just loading with the original code!! -> no need for a config 

# maybe need for clarifying for causal lm?
# AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
    
# upload function 
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
    model, tokenizer = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        #model_args_cls=LMTransformerArgs,
        model_args_cls=LMMTPArgs,
    )
    logger.info("Consolidated file created")

    # ----------------------------
    # load state_dict from pth
    # code can be used for NTP model!!!!
    #file_path = Path("./apps/main/ntp_llama_512/checkpoints/0000015300/consolidated/consolidated.pth")
    file_path = Path("./apps/mtp/llama_512/checkpoints/0000015300/consolidated/consolidated.pth")

    state_dict = torch.load(file_path)
    model_weights = state_dict["model"]
    
    print(model_weights.keys())
    
    for key in list(model_weights.keys()):
        model_weights[f"model.{key}"] = model_weights[key]
        del model_weights[key]


    # ONLY FOR NTP needed
    if "model.output.weight" in model_weights:
        model_weights["model.heads.0.weight"] = model_weights["model.output.weight"]
        del model_weights["model.output.weight"]
    
    # For strict = TRUE needed -> remove unexpected extra heads
    if "model.heads.1.weight" in model_weights:
        del model_weights["model.heads.1.weight"]
    
    if "model.heads.2.weight" in model_weights:
        del model_weights["model.heads.2.weight"]
        
    if "model.heads.3.weight" in model_weights:
        del model_weights["model.heads.3.weight"]
    
    
    # CONFIG
    config = LinguaModelConfig()
    

    
    #INITIALIZING MODELS
    # lingua_model_128 = LinguaModel(config)
    lingua_causal_model = LinguaModelForCausalLM(config)
    
    # BASE MODEL: load weights -> only needed for only using base model
    #lingua_model_128.load_state_dict(model_weights)
    
    # ! rename output weight for CausalLM (keeping both so that I can generate AND the model has model.heads.0.weight)
    model_weights["lm_head.weight"] = model_weights["model.heads.0.weight"]
    del model_weights["model.heads.0.weight"]
    print(model_weights.keys())
    
    # CAUSAL MODEL: load weights
    lingua_causal_model.load_state_dict(model_weights, strict=False)
    
    # tok_embeddings_weight = model_weights["model.tok_embeddings.weight"]
    # heads_0_weight = model_weights["model.heads.0.weight"]

    print("LM Causal Model keys:", list(lingua_causal_model.state_dict().keys()))

    
    # try registering here
    AutoConfig.register("lingua_model", LinguaModelConfig)  
    #AutoModel.register(LinguaModelConfig, LinguaModel)
    AutoModelForCausalLM.register(LinguaModelConfig, LinguaModelForCausalLM)
    
    LinguaModelConfig.register_for_auto_class()
    #LinguaModel.register_for_auto_class("AutoModel")
    LinguaModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    
  
    # # # save CAUSAL model
    save_dir = Path("./lingua_3fh_causal_512_15300")
    save_dir_outside = Path("../lingua_3fh_causal_512_15300")
    
    # create dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_outside.mkdir(parents=True, exist_ok=True)
    
    # save model in created dir
    lingua_causal_model.save_pretrained(save_dir, safe_serialization=False)
    # save directly for using it in eval pipeline
    lingua_causal_model.save_pretrained(save_dir_outside, safe_serialization=False)

    

    # push_to_hub not possible because of weight tie....
    # #lingua_model_128.push_to_hub("llama_128_15300", private=True)
    #lingua_causal_model.push_to_hub("llama_causal_128_15300", private=True, safe_serialization=False)

    shutil.copy("./huggingface/__init__.py", f"{save_dir}/__init__.py")
    shutil.copy("./huggingface/__init__.py", f"{save_dir_outside}/__init__.py")

    ## NOT PUSHING TO HF RIGHT NOW
    # api = HfApi()
    
    # # causal model
    # repo_id = "umlauf/lingua_3fh_causal_512_15300"

    # api.upload_folder(
    #     folder_path="./lingua_3fh_causal_512_15300",
    #     repo_id=repo_id,
    #     repo_type="model",
    # )
    
    # ---------------------
    # TOKENIZER
    logger.info("Pushing tokenizer to HF")
    tokenizer_path = Path("./tokenizers/llama3/original")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    hf_tokenizer.save_pretrained(save_dir)
    hf_tokenizer.save_pretrained(save_dir_outside)
    #hf_tokenizer.push_to_hub(f"lingua_3fh_causal_512_15300", private=True)
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
    