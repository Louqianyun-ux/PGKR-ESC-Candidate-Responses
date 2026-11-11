# PGKR-ESC: Candidate Response Generation and Weak-Feedback Data

## 1. Overview

This repository provides candidate response generation scripts, generated candidate responses, rule-based weak-feedback annotations, and selected implementation modules associated with the PGKR framework for emotional support conversations (ESC).

PGKR integrates persona information, commonsense relational knowledge, and support-strategy modeling for emotional support response generation. Candidate-response ranking with weak feedback is used as an additional response-level alignment mechanism.

The released materials are intended to improve research transparency and support the inspection and partial reproduction of the candidate generation and weak-feedback construction procedures.

## 2. Repository Contents

| File                          | Description                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `build_candidates.py`         | Generates multiple candidate responses using a trained model checkpoint.             |
| `build_candidate_feedback.py` | Constructs binary weak-feedback labels for candidate responses.                      |
| `judge_rules.py`              | Implements heuristic evaluation rules for empathy, supportive skills, and coherence. |
| `candidate_responses.jsonl`   | Released candidate response texts and corresponding weak-feedback labels.            |
| `candidate_feedback.npy`      | NumPy array containing binary weak-feedback labels.                                  |
| `dataloader.py`               | Auxiliary data-loading implementation module.                                        |
| `ranking_loss.py`             | Auxiliary ranking-loss implementation module.                                        |

## 3. Candidate Response Dataset

The released `candidate_responses.jsonl` contains generated candidate responses for 12,271 experimental samples.

Each sample is associated with four candidate response positions, resulting in 49,084 candidate response positions in total.

The dataset preserves the sample ordering and candidate ordering used in the weak-feedback construction procedure.

Each JSON Lines record contains the following fields:

| Field             | Description                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `sample_id`       | Sequential sample identifier, beginning at 0.                                             |
| `generation`      | List of four generated candidate responses.                                               |
| `feedback_labels` | Five binary labels corresponding to the reference response and four generated candidates. |

The first element of `feedback_labels` corresponds to the reference response, while the remaining four elements correspond to the generated candidates in their original order.

The released candidate dataset contains generated response texts rather than the complete original dialogue records.

## 4. Candidate Response Generation

Candidate responses are generated using the `build_candidates.py` script and a compatible trained PGKR model checkpoint.

The script supports stochastic decoding and generates four candidate responses per sample by default.

The default generation configuration is:

| Parameter                 | Default value  |
| ------------------------- | -------------- |
| Number of candidates      | 4              |
| Sampling method           | Top-p sampling |
| Top-p                     | 0.9            |
| Temperature               | 0.7            |
| Maximum generation length | 40             |
| Minimum generation length | 10             |
| Random seed               | 13             |

These are the defaults defined in the released script. Reproduction of the exact experimental generation requires the original model checkpoint, preprocessing environment, input data, and corresponding runtime configuration.

The candidate-generation procedure attempts to produce distinct candidates. If insufficient distinct responses are obtained within the configured attempts, existing candidates may be repeated to fill the required candidate positions.

In the processed release, 14 samples contain repeated candidate response texts. These entries are retained to preserve alignment with the experimental feedback labels.

## 5. Rule-Based Weak-Feedback Construction

Weak-feedback annotations are generated using `build_candidate_feedback.py` and the heuristic evaluation functions implemented in `judge_rules.py`.

The implemented rules consider three response characteristics:

* **Empathy:** Detects predefined empathetic expressions and filters selected harsh or insufficiently empathetic advice patterns.
* **Supportive skills:** Applies heuristic checks intended to discourage certain direct-advice and harsh-response patterns.
* **Coherence:** Applies basic text-quality and contextual overlap checks to identify potentially incoherent responses.

A generated candidate receives a positive helpfulness label only when all three corresponding rule-based checks return positive results.

The resulting labels are automatically constructed heuristic annotations.

**They are not human-annotated preference labels, clinically validated judgments, or human evaluation scores.**

## 6. Weak-Feedback Dataset

The released `candidate_feedback.npy` file contains a NumPy array with the following properties:

| Property          | Value       |
| ----------------- | ----------- |
| Number of samples | 12,271      |
| Array shape       | (12,271, 5) |
| Data type         | int64       |
| Label values      | 0 and 1     |

The five columns are organized as follows:

| Column | Description                                  |
| ------ | -------------------------------------------- |
| 0      | Reference response label, fixed at 1         |
| 1      | Weak-feedback label for candidate response 1 |
| 2      | Weak-feedback label for candidate response 2 |
| 3      | Weak-feedback label for candidate response 3 |
| 4      | Weak-feedback label for candidate response 4 |

For generated candidate responses:

* **1** indicates that all implemented heuristic criteria are satisfied.
* **0** indicates that at least one heuristic criterion is not satisfied.

The reference label is assigned a value of 1 by construction and should not be interpreted as an independently obtained human evaluation.

The order of the samples and candidates is preserved between `candidate_responses.jsonl` and `candidate_feedback.npy`.

## 7. Data Preparation and Release Processing

The original candidate-generation output contains additional information, including dialogue contexts, persona representations, commonsense relation texts, reference responses, and model inputs.

For the public candidate-response release, these additional fields were excluded. The exported records retain only sample identifiers, generated candidates, and corresponding weak-feedback labels.

The data preparation procedure included:

* Checking the consistency of sample identifiers and row ordering.
* Verifying the presence of four candidate responses per sample.
* Confirming alignment with the five-column feedback matrix.
* Detecting empty and duplicate candidate responses.
* Screening for selected patterns of potentially sensitive information.

One phone-like numeric sequence was replaced with the placeholder `[REDACTED_NUMBER]` in the released candidate-response text.

The original feedback labels were constructed before this redaction. Consequently, the corresponding released text is not identical to the original text used for weak-feedback construction.

No sample rows or candidate positions were removed during this redaction step.

Automated structural validation and pattern-based screening do not, by themselves, establish that all possible sensitive information or redistribution restrictions have been eliminated.

## 8. Requirements and Usage

The scripts use Python and require relevant dependencies, including NumPy, PyTorch, Transformers, and tqdm, depending on the component being executed.

The weak-feedback construction script uses `judge_rules.py` and requires the original candidate-generation records containing dialogue contexts and generated responses.

The released `candidate_responses.jsonl` is a reduced data representation and does not contain all fields needed to rerun the original feedback construction procedure.

Researchers can inspect the released NumPy annotations using:

`import numpy as np`

`labels = np.load("candidate_feedback.npy", allow_pickle=False)`

`print(labels.shape)`

The expected output is:

`(12271, 5)`

The released JSON Lines file can be read using standard JSON-processing tools.

## 9. Reproducibility Scope

This repository provides the generated candidate responses, associated heuristic feedback labels, and selected relevant scripts.

It supports examination of the released candidate-response data, analysis of the weak-feedback labels, and inspection of the implemented heuristic rules.

However, the complete original PGKR training and inference environment is not included in this repository.

Exact end-to-end reproduction may require additional components, including the corresponding model checkpoints, source datasets, preprocessing modules, generated commonsense knowledge, and experiment-specific configurations.

Therefore, the repository should not be interpreted as a fully self-contained reproduction package for all PGKR experiments.

## 10. Data Availability

The generated candidate response texts, associated rule-based weak-feedback labels, and relevant generation and evaluation scripts are publicly available in this repository:

https://github.com/Louqianyun-ux/PGKR-ESC-Candidate-Responses

The released candidate-response dataset contains 12,271 samples and four generated candidate response positions per sample.

The corresponding feedback matrix contains five binary labels per sample, including the reference response label and four generated candidate labels.

The complete original model-input records, trained checkpoints, and study-specific generated commonsense relation data are not included in this release.

The original source datasets should be accessed through their respective repositories and are subject to the applicable data-use and redistribution terms.

## 11. Related Resources

The PGKR research framework builds upon publicly available resources for emotional support conversations and commonsense knowledge modeling.

* **ESConv — Emotional Support Conversation:** https://github.com/thu-coai/Emotional-Support-Conversation
* **PAL / PESConv — Persona-annotated Emotional Support Conversation:** https://github.com/chengjl19/PAL
* **COMET — Commonsense Knowledge Generation:** https://github.com/atcbosselut/comet-commonsense

Users should consult the original resource repositories for access conditions, usage restrictions, and appropriate citations.

## 12. Citation

This repository accompanies the PGKR research project on emotional support conversation generation.

Formal bibliographic information will be added when the associated manuscript becomes publicly available.

Until then, researchers using these materials are encouraged to acknowledge this repository and cite the relevant original datasets and tools.

## 13. Notes

The released weak-feedback labels are generated using deterministic heuristic rules applied to the original candidate responses.

They are provided as research artifacts rather than human-verified ground-truth assessments.

The repository includes selected research components and derived data. Users remain responsible for complying with applicable source-data terms and evaluating the suitability of these materials for their own research.
