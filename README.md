# RL for Medical

## Phase 1: Medical Accuracy/ Knowledge Improvement
- Set the `custom_reward_function` to `verl/utils/simpleqa_eval.py` and `compute_score` (For open-ended dataset or remove if working on multiple-choice/ non open-ended dataset)
- Dataset phase 1: https://huggingface.co/datasets/Intelligent-Internet/II-Medical-RL


## Phase 2: Safety Alignment
- Set the `custom_reward_function` to `verl/utils/complexqa_eval.py` and `compute_score`
- Dataset phase 2: https://huggingface.co/datasets/Intelligent-Internet/ChatDoctor-RL


```
bash dapo_script_ii_8b.sh
```