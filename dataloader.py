# coding=utf-8

import json
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MuffinFeature:
    input_ids: List[int]
    attention_mask: List[int]
    persona_input_ids: List[int]
    persona_attention_mask: List[int]
    comet_input_ids: List[List[int]]
    comet_attention_mask: List[List[int]]
    decoder_input_ids: List[int]
    candidate_ids: List[List[int]]
    candidate_labels: List[int]
    strat_id: int = -1


class MuffinDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        feedback_path: str,
        tokenizer,
        max_candidate_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_candidate_len = max_candidate_len

        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(x) for x in f]

        self.feedback = np.load(feedback_path)
        assert len(self.samples) == len(self.feedback), "samples and feedback size mismatch"

        self.features = self._build_features()

    def _encode_candidate(self, text: str) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_candidate_len,
        )

    def _safe_list(self, x):
        return list(x) if x is not None else []

    def _safe_2d_list(self, x):
        if x is None:
            return []

        x = list(x)

        # compatible with both [R, L] and accidentally saved [1, R, L]
        if len(x) > 0 and isinstance(x[0], list):
            if len(x[0]) > 0 and isinstance(x[0][0], list):
                if len(x) == 1:
                    x = x[0]

        return [list(row) for row in x]

    def _safe_scalar_int(self, x, default=-1):
        if x is None:
            return default

        while isinstance(x, list) and len(x) > 0:
            x = x[0]

        try:
            return int(x)
        except Exception:
            return default

    def _build_features(self):
        feats = []
        for item, labels in zip(self.samples, self.feedback):
            model_inputs = item["model_inputs"]

            gold = item["response_text"]
            candidates = item["generation"]
            all_responses = [gold] + candidates

            candidate_ids = [self._encode_candidate(r) for r in all_responses]
            candidate_labels = labels.tolist()

            feats.append(
                MuffinFeature(
                    input_ids=self._safe_list(model_inputs["input_ids"]),
                    attention_mask=self._safe_list(model_inputs["attention_mask"]),
                    persona_input_ids=self._safe_list(model_inputs["persona_input_ids"]),
                    persona_attention_mask=self._safe_list(model_inputs["persona_attention_mask"]),
                    comet_input_ids=self._safe_2d_list(model_inputs["comet_input_ids"]),
                    comet_attention_mask=self._safe_2d_list(model_inputs["comet_attention_mask"]),
                    decoder_input_ids=self._safe_list(model_inputs["decoder_input_ids"]),
                    candidate_ids=candidate_ids,
                    candidate_labels=candidate_labels,
                    strat_id=self._safe_scalar_int(model_inputs.get("strat_id", -1)),
                )
            )
        return feats

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def pad_2d(seqs, pad_id):
    max_len = max(len(x) for x in seqs)
    return [x + [pad_id] * (max_len - len(x)) for x in seqs]


def pad_3d_relations(batch_rel_seqs, pad_id):
    normalized = []

    for sample in batch_rel_seqs:
        sample = list(sample)

        # compatible with both [R, L] and [1, R, L]
        if len(sample) > 0 and isinstance(sample[0], list):
            if len(sample[0]) > 0 and isinstance(sample[0][0], list):
                if len(sample) == 1:
                    sample = sample[0]

        normalized.append(sample)

    num_rel = max(len(x) for x in normalized) if len(normalized) > 0 else 0
    max_len = 1
    for sample in normalized:
        for rel in sample:
            max_len = max(max_len, len(rel))

    padded = []
    for sample in normalized:
        rels = []
        for rel in sample:
            rels.append(list(rel) + [pad_id] * (max_len - len(rel)))
        while len(rels) < num_rel:
            rels.append([pad_id] * max_len)
        padded.append(rels)

    return padded


def collate_muffin(features, pad_token_id: int):
    input_ids = pad_2d([f.input_ids for f in features], pad_token_id)
    attention_mask = pad_2d([f.attention_mask for f in features], 0)
    persona_input_ids = pad_2d([f.persona_input_ids for f in features], pad_token_id)
    persona_attention_mask = pad_2d([f.persona_attention_mask for f in features], 0)
    comet_input_ids = pad_3d_relations([f.comet_input_ids for f in features], pad_token_id)
    comet_attention_mask = pad_3d_relations([f.comet_attention_mask for f in features], 0)
    decoder_input_ids = pad_2d([f.decoder_input_ids for f in features], pad_token_id)

    max_cand_num = max(len(f.candidate_ids) for f in features)
    max_cand_len = max(max(len(c) for c in f.candidate_ids) for f in features)

    candidate_ids = []
    candidate_labels = []

    for f in features:
        cur_ids = []
        for c in f.candidate_ids:
            cur_ids.append(c + [pad_token_id] * (max_cand_len - len(c)))
        while len(cur_ids) < max_cand_num:
            cur_ids.append([pad_token_id] * max_cand_len)

        cur_labels = list(f.candidate_labels)
        while len(cur_labels) < max_cand_num:
            cur_labels.append(-1)

        candidate_ids.append(cur_ids)
        candidate_labels.append(cur_labels)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "persona_input_ids": torch.tensor(persona_input_ids, dtype=torch.long),
        "persona_attention_mask": torch.tensor(persona_attention_mask, dtype=torch.long),
        "comet_input_ids": torch.tensor(comet_input_ids, dtype=torch.long),
        "comet_attention_mask": torch.tensor(comet_attention_mask, dtype=torch.long),
        "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
        "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
        "candidate_labels": torch.tensor(candidate_labels, dtype=torch.long),
        "strat_id": torch.tensor([f.strat_id for f in features], dtype=torch.long),
    }
