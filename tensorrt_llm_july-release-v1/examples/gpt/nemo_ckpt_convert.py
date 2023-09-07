#! /usr/bin/env python3
import argparse
import configparser
import datetime
import logging
import multiprocessing
import shutil
import sys
import tempfile
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, T5Tokenizer
from utils.convert import (cpu_map_location, gpu_map_location,
                           split_and_save_weight)
from utils.nemo import (UnpackedNemoCheckpointDir, extract_layers_with_prefix,
                        nemo_to_gpt_config, unpack_nemo_ckpt)

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

LOGGER = logging.getLogger(__name__)


def rename_key(old_key: str, pp_rank: int, num_layers: int, pp_size: int):
    new_key = old_key

    if "layers." in old_key:
        split_key = old_key.split(".")
        split_key[1] = str(int(split_key[1]) + pp_rank * num_layers // pp_size)
        new_key = ".".join(split_key)

        if "self_attention" in new_key:
            new_key = new_key.replace("self_attention", "attention")
    return new_key


@torch.no_grad()
def convert_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir,
                       args):
    nemo_model_config = unpacked_checkpoints_dir.model_config

    checkpoints_paths = unpacked_checkpoints_dir.get_checkpoints_paths(
        nemo_model_config.get("tensor_model_parallel_size", 1),
        nemo_model_config.get("pipeline_model_parallel_size", 1),
    )

    # if checkpoints files could be found - start preparing output dir
    out_dir = create_out_dir(args)

    map_location_fn = gpu_map_location if args.load_checkpoints_on_gpu else cpu_map_location
    storage_type = str_dtype_to_torch(args.storage_type)

    # load position_embedding from rank 0
    model_00 = torch.load(checkpoints_paths[0][0], map_location=map_location_fn)
    model_00 = model_00.get("state_dict", model_00)

    has_position_embedding = "model.language_model.embedding.position_embeddings.weight" in model_00
    has_lm_head = "model.language_model.output_layer.weight" in model_00

    num_layers = nemo_model_config["num_layers"]
    training_tp_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    training_pp_size = nemo_model_config.get("pipeline_model_parallel_size", 1)
    inference_tp_size = args.tensor_parallelism

    export_config = {
        "apply_layernorm_1p":
        nemo_model_config.get('normalization', '') == "layernorm1p",
        "tp_size":
        training_tp_size,
        "split_gated_activation":
        "swiglu" in nemo_model_config.get('activation', "gelu"),
        "num_attention_heads":
        nemo_model_config["num_attention_heads"],
        "use_attention_nemo_shape":
        True,
        "transpose_weights":
        True,
    }

    # merge_factor: how many TP training nodes are merged into an inference TP node
    # split_factor: in how many parts a TP training node is split
    gcd = np.gcd(training_tp_size, inference_tp_size)
    merge_factor = training_tp_size // gcd
    split_factor = inference_tp_size // gcd

    model_level_weights = defaultdict(list)

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        if tp_idx == 0 and pp_idx == 0:
            if has_position_embedding:
                val = model[
                    "model.language_model.embedding.position_embeddings.weight"]
                # not weight, do not need to transpose
                val = torch_to_numpy(val.to(storage_type).cpu())
                val.tofile(out_dir / "model.wpe.bin")
                model_level_weights["model.wpe.bin"].append(val)
        if pp_idx == 0:
            val = model.get(
                "state_dict",
                model)["model.language_model.embedding.word_embeddings.weight"]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.wte.bin"].append(val)
        if has_lm_head and pp_idx == training_pp_size - 1:
            val = model.get("state_dict",
                            model)["model.language_model.output_layer.weight"]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.lm_head.weight.bin"].append(val)

    for tp_rank in range(training_tp_size // merge_factor):
        for pp_rank in range(training_pp_size):

            models = []
            for k in range(merge_factor):
                rank_weights = checkpoints_paths[tp_rank * merge_factor +
                                                 k][pp_rank]
                model = torch.load(rank_weights, map_location=map_location_fn)
                handle_model_level_weights(model, tp_rank * merge_factor + k,
                                           pp_rank)
                layers = extract_layers_with_prefix(
                    model, "model.language_model.encoder.")
                models.append(layers)

            starmap_args = []
            for key in models[0].keys():
                starmap_args.append((
                    tp_rank,
                    out_dir,
                    split_factor,
                    rename_key(key, pp_rank, num_layers, training_pp_size),
                    [model[key] for model in models],
                    storage_type,
                    None,
                    export_config,
                ))
            starmap_args = tqdm(starmap_args, desc="saving weights")

            if args.processes > 1:
                with multiprocessing.Pool(args.processes) as pool:
                    pool.starmap(split_and_save_weight, starmap_args)
            else:
                # simpler for debug situations
                for starmap_arg in starmap_args:
                    split_and_save_weight(*starmap_arg)

    for key, values in model_level_weights.items():
        model_level_weights[key] = np.concatenate(values, axis=0)
        model_level_weights[key].tofile(out_dir / key)
    vocab_size = model_level_weights["model.wte.bin"].shape[0]

    tokenizer_config = update_tokenizer_paths(nemo_model_config["tokenizer"],
                                              unpacked_checkpoints_dir)
    copy_tokenizer_files(tokenizer_config, out_dir)
    tokenizer = build_tokenizer(tokenizer_config)
    gpt_model_config = nemo_to_gpt_config(nemo_model_config, vocab_size,
                                          tokenizer.eos_token_id,
                                          tokenizer.bos_token_id)

    config = configparser.ConfigParser()
    config["gpt"] = {k: str(v) for k, v in vars(gpt_model_config).items()}
    config["gpt"]["storage_dtype"] = args.storage_type
    config_path = out_dir / "config.ini"
    with config_path.open("w") as config_file:
        config.write(config_file)


def create_out_dir(args):
    out_dir = Path(args.out_dir) / f"{args.tensor_parallelism}-gpu/"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    return out_dir


def update_tokenizer_paths(tokenizer_config: typing.Dict,
                           unpacked_checkpoints_dir):

    def _update_config_entry(key, file_pattern):
        old_path = tokenizer_config[key]
        if old_path is None:
            return
        old_path = Path(old_path)
        new_path = unpacked_checkpoints_dir.get_tokenizer_file_path(
            "tokenizer", key, file_pattern)
        if new_path:
            LOGGER.debug(f"Update tokenizer {key} {old_path} -> {new_path}")
            tokenizer_config[key] = new_path.as_posix()
        elif not old_path.exists():
            LOGGER.warning(
                f"Tokenizer {key}'s path {old_path} does not exists: set it to None"
            )
            tokenizer_config[key] = None

    _update_config_entry("model", "*.model")
    _update_config_entry("vocab_file", "*vocab*")
    _update_config_entry("merge_file", "*merge*.txt")

    return tokenizer_config


def copy_tokenizer_files(config, out_dir):
    basenames = {
        "model": "tokenizer",
        "vocab_file": "vocab",
        "merge_file": "merges",
    }

    for key in basenames.keys():
        if config[key] is None:
            continue
        path = Path(config[key])
        if not path.exists():
            LOGGER.debug(f"Tokenizer {key}: {path} file not found")
            continue

        dst_path = out_dir / f"{basenames[key]}{path.suffix}"
        LOGGER.debug(f"Copy tokenizer {key}: {path}->{dst_path}")
        shutil.copy(path.as_posix(), dst_path.as_posix())


def build_tokenizer(tokenizer_config: typing.Dict):
    if tokenizer_config["library"] == "sentencepiece":
        tokenizer = T5Tokenizer(tokenizer_config["model"], extra_ids=0)
    elif "GPT2" in tokenizer_config["type"]:
        tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"],
                                  tokenizer_config["merge_file"])
    else:
        raise ValueError(
            f'Tokenizer type {tokenizer_config["library"]} not handled')

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": "<s>"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    return tokenizer


def main():
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out-dir',
                        '-o',
                        type=Path,
                        help='path to output directory',
                        required=True)
    parser.add_argument('--in-file',
                        '-i',
                        type=Path,
                        help='path to input checkpoint file',
                        required=True)
    parser.add_argument('--tensor-parallelism',
                        '-tp',
                        type=int,
                        help='Requested tensor parallelism for inference',
                        default=1)
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help=
        "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
        default=4)
    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--load-checkpoints-on-gpu",
                        action="store_true",
                        help="Whether to load model weights to GPU")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    args = parser.parse_args()

    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=log_format)

    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    if not args.in_file.exists():
        LOGGER.error("%s does not exists", args.in_file)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # unpack if needed
        if args.in_file.is_dir():
            nemo_dir = args.in_file
        else:
            start_time = datetime.datetime.now()
            checkpoint_dir_path = temp_dir / "unpacked"
            nemo_dir = unpack_nemo_ckpt(args.in_file, checkpoint_dir_path)
            LOGGER.info("Spent %s (h:m:s) to unpack NeMo archive",
                        datetime.datetime.now() - start_time)

        unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
            nemo_dir, load_checkpoints_to_cpu=not args.load_checkpoints_on_gpu)

        start_time = datetime.datetime.now()
        convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model",
                    datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
