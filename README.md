# PGKR-ESC: Candidate Response Generation and Weak-Feedback Data

## 1. Overview

This repository provides the candidate response generation scripts, rule-based weak-feedback annotations, and selected implementation modules associated with the PGKR framework for emotional support conversations (ESC).

The repository supports research transparency by documenting the generation of candidate responses and the construction of heuristic feedback labels used in the PGKR experimental pipeline.

## 2. Repository Contents

The repository contains the following files:

| File                          | Description                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `build_candidates.py`         | Script for generating multiple candidate responses using a trained model checkpoint.             |
| `build_candidate_feedback.py` | Script for constructing binary weak-feedback labels from generated candidate responses.          |
| `judge_rules.py`              | Rule-based evaluation functions for empathy, supportive response characteristics, and coherence. |
| `candidate_feedback.npy`      | NumPy array containing the generated binary weak-feedback labels.                                |
| `dataloader.py`               | Additional project data-loading module.                                                          |
| `ranking_loss.py`             | Additional project ranking-loss module.                                                          |

## 3. Candidate Response Generation

The `build_candidates.py` script generates multiple candidate responses for each input sample.

The default configuration generates four candidate responses per sample using stochastic decoding with the following parameters:

* Number of candidates: 4
* Top-p sampling: 0.9
* Temperature: 0.7
* Maximum generation length: 40
* Minimum generation length: 10
* Random seed: 13

These values are script defaults and can be modified using command-line arguments.

Candidate generation requires the corresponding trained model checkpoint, preprocessing components, input data, and original project environment.

The generated outputs are stored in JSON Lines format, with candidate responses recorded under the `generation` field.

## 4. Weak-Feedback Construction

Weak-feedback labels are constructed using the rule-based functions defined in `judge_rules.py`.

The rules consider three response characteristics:

* **Empathy:** Whether the response contains supportive or empathetic expressions and avoids certain harsh patterns.
* **Supportive skill:** Whether the response avoids specified inappropriate or excessively direct advice patterns.
* **Coherence:** Whether the response satisfies basic text-quality and contextual consistency heuristics.

A generated response receives a positive helpfulness label only when all three rule-based checks return positive values.

These labels are automatically generated heuristic annotations, not human evaluation scores.

## 5. Weak-Feedback Dataset

The released `candidate_feedback.npy` file contains a binary NumPy array with the following properties:

* Shape: **(12,271, 5)**
* Data type: `int64`
* Label values: 0 and 1

Each row corresponds to one processed sample.

The five columns are organized as follows:

| Column | Description                                  |
| ------ | -------------------------------------------- |
| 0      | Reference response label, fixed at 1         |
| 1      | Weak-feedback label for candidate response 1 |
| 2      | Weak-feedback label for candidate response 2 |
| 3      | Weak-feedback label for candidate response 3 |
| 4      | Weak-feedback label for candidate response 4 |

For generated candidate responses, a value of 1 indicates that the response satisfies the implemented heuristic criteria, while 0 indicates that at least one criterion is not satisfied.

The row order follows the input order used when constructing the feedback array. The released labels should not be interpreted as independently validated human preference annotations.

## 6. Requirements and Usage

The weak-feedback construction script requires Python, NumPy, tqdm, and the accompanying `judge_rules.py` module.

Given a compatible candidate-response file in the expected JSON Lines format, feedback labels can be generated using:

`python build_candidate_feedback.py --candidates_path ./data/candidates.txt --output_path ./data/candidate_feedback.npy`

Candidate response generation additionally depends on PyTorch, Hugging Face Transformers, the original PGKR project modules, and a compatible trained model checkpoint.

The scripts are not provided as a fully standalone training and inference package.

## 7. Data Availability

The rule-based weak-feedback labels and the accompanying generation and evaluation scripts are publicly available in this repository.

The released feedback array contains 12,271 samples with five binary labels per sample.

The complete generated candidate response texts are not currently included in this public release. Additional preparation and review are required before their public distribution.

Accordingly, this repository provides partial research artifacts rather than a fully self-contained reproduction package.

## 8. Related Resources

The PGKR research framework builds upon resources associated with emotional support conversations and commonsense knowledge.

Relevant public resources include:

* **ESConv:** https://github.com/thu-coai/Emotional-Support-Conversation
* **PAL / PESConv:** https://github.com/chengjl19/PAL
* **COMET:** https://github.com/atcbosselut/comet-commonsense

## 9. Research Notes

The weak-feedback construction procedure uses heuristic rules and does not replace human evaluation.

Reproducing the complete experimental pipeline may require additional model checkpoints, input data, preprocessing components, experimental configurations, and project-specific dependencies.

The availability of these components should be considered when interpreting the reproducibility scope of this repository.

## 10. Citation

This repository accompanies the PGKR research project on emotional support conversation generation.

Formal publication and citation information will be added when available.
