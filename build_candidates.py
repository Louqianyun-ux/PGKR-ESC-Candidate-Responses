# coding=utf-8

import os
import sys
import json
import argparse
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch import Tensor
from transformers.trainer_utils import set_seed
from tqdm import tqdm

from inputters import inputters
from utils.building_utils import build_model


def tensor_to_list(x):
    if x is None:
        return None
    if isinstance(x, Tensor):
        x = x.detach().cpu()
        if x.dim() == 0:
            return int(x.item())
        return x.tolist()
    return x


def decode_clean(tokenizer, ids):
    if ids is None:
        return ""
    if isinstance(ids, Tensor):
        ids = ids.detach().cpu().tolist()
    ids = [x for x in ids if x != -100]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def decode_comet_relations(tokenizer, rel_ids):
    if rel_ids is None:
        return {}

    if isinstance(rel_ids, Tensor):
        rel_ids = rel_ids.detach().cpu().tolist()

    rel_names = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
    comet_text = {}

    for i, ids in enumerate(rel_ids):
        ids = [x for x in ids if x != -100]
        key = rel_names[i] if i < len(rel_names) else f"rel_{i}"
        comet_text[key] = tokenizer.decode(ids, skip_special_tokens=True).strip()

    return comet_text


def get_first_row(x):
    if isinstance(x, Tensor):
        if x.dim() >= 2:
            return x[0]
        return x
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return x


@torch.no_grad()
def generate_k_candidates(
    model,
    tokenizer,
    batch: Dict[str, Any],
    device: torch.device,
    k: int,
    max_length: int,
    min_length: int,
    top_p: float,
    temperature: float,
):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    persona_input_ids = batch["persona_input_ids"].to(device)
    persona_attention_mask = batch["persona_attention_mask"].to(device)
    comet_input_ids = batch["comet_input_ids"].to(device)
    comet_attention_mask = batch["comet_attention_mask"].to(device)

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    batch_size = input_ids.size(0)
    decoder_input_ids = torch.full(
        (batch_size, 1),
        fill_value=bos,
        dtype=torch.long,
        device=device
    )

    generations = []
    seen = set()

    for _ in range(k * 4):
        _, gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            persona_input_ids=persona_input_ids,
            persona_attention_mask=persona_attention_mask,
            comet_input_ids=comet_input_ids,
            comet_attention_mask=comet_attention_mask,
            decoder_input_ids=decoder_input_ids,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
            top_p=top_p,
            temperature=temperature,
            max_length=max_length,
            min_length=min_length,
        )

        text = decode_clean(tokenizer, gen_ids[0])
        if text and text not in seen:
            seen.add(text)
            generations.append(text)

        if len(generations) >= k:
            break

    while len(generations) < k:
        generations.append(generations[-1] if generations else "")

    return generations, decoder_input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default="strat")
    parser.add_argument('--inputter_name', type=str, default="strat")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--load_checkpoint", '-c', type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_decoder_input_length", type=int, default=50)
    parser.add_argument("--max_knowledge_len", type=int, default=None)
    parser.add_argument('--label_num', type=int, default=None)
    parser.add_argument('--only_encode', action='store_true')
    parser.add_argument('--use_all_persona', action='store_true')
    parser.add_argument('--encode_context', action='store_true')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--gen_max_length", type=int, default=40)
    parser.add_argument("--gen_min_length", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/disk_8t_2/lemon/DKPE-master/DATA/muffin_train/candidates.txt",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = {
        'inputter_name': args.inputter_name,
        'config_name': args.config_name,
    }

    tokenizer = build_model(only_toker=True, local_rank=-1, **names)
    inputter = inputters[args.inputter_name](use_all_persona=args.use_all_persona)

    train_dataloader = inputter.train_dataloader(
        toker=tokenizer,
        feature_dataset=inputter.train_dataset,
        batch_size=args.batch_size,
        **names
    )

    _, model = build_model(checkpoint=args.load_checkpoint, local_rank=-1, **names)
    model.to(device)
    model.eval()

    with open(args.output_path, "w", encoding="utf-8") as fout:
        sample_idx = 0
        for batch in tqdm(train_dataloader, desc="Building candidates"):
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            candidates, decoder_input_ids = generate_k_candidates(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                device=device,
                k=args.num_candidates,
                max_length=args.gen_max_length,
                min_length=args.gen_min_length,
                top_p=args.top_p,
                temperature=args.temperature,
            )

            input_ids_1 = get_first_row(batch["input_ids"])
            attention_mask_1 = get_first_row(batch["attention_mask"])
            persona_ids_1 = get_first_row(batch["persona_input_ids"])
            persona_mask_1 = get_first_row(batch["persona_attention_mask"])
            comet_ids_1 = get_first_row(batch["comet_input_ids"])
            comet_mask_1 = get_first_row(batch["comet_attention_mask"])
            decoder_ids_1 = get_first_row(decoder_input_ids)
            labels_1 = get_first_row(batch["labels"]) if "labels" in batch else None
            strat_id_1 = get_first_row(batch["strat_id"]) if "strat_id" in batch else None

            record = {
                "sample_id": sample_idx,
                "context_text": decode_clean(tokenizer, input_ids_1),
                "persona_text": decode_clean(tokenizer, persona_ids_1),
                "comet_text": decode_comet_relations(tokenizer, comet_ids_1),
                "response_text": decode_clean(tokenizer, labels_1),
                "generation": candidates,
                "model_inputs": {
                    "input_ids": tensor_to_list(input_ids_1),
                    "attention_mask": tensor_to_list(attention_mask_1),
                    "persona_input_ids": tensor_to_list(persona_ids_1),
                    "persona_attention_mask": tensor_to_list(persona_mask_1),
                    "comet_input_ids": tensor_to_list(comet_ids_1),
                    "comet_attention_mask": tensor_to_list(comet_mask_1),
                    "decoder_input_ids": tensor_to_list(decoder_ids_1),
                    "strat_id": tensor_to_list(strat_id_1),
                }
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            sample_idx += 1

    print(f"Saved candidates to {args.output_path}")


if __name__ == "__main__":
    main()
