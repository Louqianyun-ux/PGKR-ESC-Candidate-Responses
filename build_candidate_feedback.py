# coding=utf-8

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from judge_rules import aggregate_feedback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates_path",
        type=str,
        default="/home/disk_8t_2/lemon/DKPE-master/DATA/muffin_train/candidates.txt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/disk_8t_2/lemon/DKPE-master/DATA/muffin_train/candidate_feedback.npy",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    labels_all = []

    with open(args.candidates_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Building candidate feedback"):
            item = json.loads(line)
            context = item["context_text"]
            gold_response = item["response_text"]
            candidates = item["generation"]

           
            labels = [1]

            for cand in candidates:
                fb = aggregate_feedback(context, cand)
                labels.append(fb["helpful"])

            labels_all.append(labels)

    labels_all = np.array(labels_all, dtype=np.int64)
    np.save(args.output_path, labels_all)
    print(f"Saved feedback to {args.output_path}, shape={labels_all.shape}")


if __name__ == "__main__":
    main()