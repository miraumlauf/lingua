import re
import os
from dataclasses import dataclass
import logging
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig, LlamaConfig, LlamaForCausalLM, AutoTokenizer
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

logger = logging.getLogger()

@dataclass
class UploadArgs:
    username: str = ""
    repo_name: str = ""

    name: str = "upload"
    dump_dir: Optional[str] = None
    ckpt_dir: str = ""
    model_dir: str = ""

class LinguaToHFModel(LlamaForCausalLM, PyTorchModelHubMixin):
    """
    Converts a Lingua model to a Llama-style Hugging Face format, matching parameter keys 
    with those expected by a typical Llama implementation.

    Reference: 
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py 
    """

    @classmethod
    def from_lingua(cls, lingua_model):
        # Build a LlamaConfig using dimensions from lingua_model
        config = LlamaConfig(
            hidden_size=lingua_model.dim,
            intermediate_size=lingua_model.layers[0].feed_forward.w1.weight.shape[0],
            num_attention_heads=lingua_model.layers[0].attention.n_heads,
            num_hidden_layers=len(lingua_model.layers),
            vocab_size=lingua_model.heads[0].out_features, #changed because of multiple heads
            rope_theta=10000.0,
            max_position_embeddings=getattr(lingua_model, "max_seqlen", 512),
            use_cache=False
        )

        # Instantiate an HF LlamaForCausalLM
        model = cls(config)

        # Map Lingua state_dict to HF keys
        state_dict = lingua_model.state_dict()
        final_state_dict = {}

        key_map = {
            "tok_embeddings.weight":            "model.embed_tokens.weight",
            "layers.{}.attention.wo.weight":    "model.layers.{}.self_attn.o_proj.weight",
            "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
            "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
            "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
            "layers.{}.attention_norm.weight":  "model.layers.{}.input_layernorm.weight",
            "layers.{}.ffn_norm.weight":        "model.layers.{}.post_attention_layernorm.weight",
            "norm.weight":                      "model.norm.weight",
            "heads.2.weight":                    "lm_head.weight"
        }

        for old_key, value in state_dict.items():
            layer_match = re.search(r"layers\.(\d+)\.", old_key)

            # Remap Q/K/V weights without adding unused biases
            if "attention.wq.weight" in old_key and layer_match:
                new_key = f"model.layers.{layer_match.group(1)}.self_attn.q_proj.weight"
                final_state_dict[new_key] = value
            elif "attention.wk.weight" in old_key and layer_match:
                new_key = f"model.layers.{layer_match.group(1)}.self_attn.k_proj.weight"
                final_state_dict[new_key] = value
            elif "attention.wv.weight" in old_key and layer_match:
                new_key = f"model.layers.{layer_match.group(1)}.self_attn.v_proj.weight"
                final_state_dict[new_key] = value
            else:
                # Handle feed_forward, LN, output via key_map
                if layer_match:
                    # Replace only the layer index, preserving feed_forward part
                    layer_idx = layer_match.group(1)
                    suffix = old_key[len(f"layers.{layer_idx}."):]
                    abstract_key = f"layers.{{}}.{suffix}"
                    if abstract_key in key_map:
                        new_key = key_map[abstract_key].format(layer_idx)
                        final_state_dict[new_key] = value
                else:
                    if old_key in key_map:
                        new_key = key_map[old_key]
                        final_state_dict[new_key] = value

        # Load and show missing/unexpected
        missing, unexpected = model.load_state_dict(final_state_dict, strict=True)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

        return model
    
    
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
    
    return model


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
    model = upload(cfg)
    attributes = dir(model)
    print(attributes)
    print(model.heads)

    # Convert and test
    hf_model = LinguaToHFModel.from_lingua(model)
    hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", subfolder="original")
    # Fix pad_token_id warning by setting it to the eos_token_id
    hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id
    hf_model.config.pad_token_id = hf_tokenizer.pad_token_id

    input_text = "The quick brown"
    inputs = hf_tokenizer(input_text, return_tensors="pt")
    
    outputs = hf_model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        top_p=0.8
    )
        
    
    generated_text = hf_tokenizer.decode(outputs[0])

    print(f"Input text: {input_text}")
    print(f"Generated with HF model: {generated_text}")

if __name__ == "__main__":
    main()
