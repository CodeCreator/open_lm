import argparse
from utils.transformers.hf_model import OpenLMModel
from transformers import GPTNeoXTokenizerFast
from utils.transformers.hf_config import OpenLMConfig
import torch
import json
import yaml
from open_lm.model import Params, create_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--params")
    parser.add_argument("--model-config")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint)

    params_args_dict = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)
    params_args_dict["model"] = args.model_config

    openlm_config = OpenLMConfig(params_args_dict=params_args_dict)
    open_lm = OpenLMModel(openlm_config)
    # hardcoded to NeoX Tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("_orig_mod.", "model."): y for x, y in state_dict.items()}
    open_lm.load_state_dict(state_dict)
    open_lm.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
