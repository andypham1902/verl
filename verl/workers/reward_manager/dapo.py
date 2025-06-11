# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score, medical
from verl.utils.reward_score.ifeval import ifeval
from verl.utils.reward_score.healthbench import healthbench
from verl.workers.reward_manager import register


def extract_content(solution_str):
    """Remove reasoning content from the solution string.
    Args:
        solution_str (str): The solution string containing reasoning content.
    Returns:
        str: The solution string with reasoning content removed.
    """
    reasoning_tags = ["</think>", "</thinking>"]
    for tag in reasoning_tags:
        if tag in solution_str:
            solution_str = solution_str.split(tag)[1].strip()
    return solution_str

@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        n_resp_per_prompt=1,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.n_resp_per_prompt = n_resp_per_prompt

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(0, len(data), self.n_resp_per_prompt):
            # Batch clean check
            prev_groudtruth = None
            prev_data_source = None
            clean_batch = True
            for j in range(i, i+self.n_resp_per_prompt):
                _data_item = data[j]
                _ground_truth = _data_item.non_tensor_batch["reward_model"]["ground_truth"]
                _data_source = _data_item.non_tensor_batch[self.reward_fn_key]
                if prev_groudtruth and _ground_truth != prev_groudtruth:
                        clean_batch = False
                        print(f"Ground truth mismatch at index {j}: {_ground_truth} != {prev_groudtruth}")
                        break
                if prev_data_source and _data_source != prev_data_source:
                        clean_batch = False
                        print(f"Data source mismatch at index {j}: {_data_source} != {prev_data_source}")
                        break
                prev_groudtruth = _ground_truth
                prev_data_source = _data_source
            
            if not clean_batch:
                print(f"Skipping batch from index {i} to {i + self.n_resp_per_prompt - 1} due to mismatch.")
                continue
            
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = [data[_i].batch["responses"] for _i in range(i, i + self.n_resp_per_prompt)]
            valid_response_length = [data[_i].batch["attention_mask"][prompt_length:].sum() for _i in range(i, i + self.n_resp_per_prompt)]
            valid_response_ids = []
            for _i in range(self.n_resp_per_prompt):
                valid_response_ids.append(response_ids[_i][:valid_response_length[_i]])

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            response_strs = []
            for _i in range(self.n_resp_per_prompt):
                response_str = self.tokenizer.decode(valid_response_ids[_i], skip_special_tokens=True)
                if response_str.endswith(eos_token):
                    response_str = response_str[: -len(eos_token)]
                response_strs.append(extract_content(response_str))

            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            if data_source in ["miriad/miriad-5.8M"]:
                result = self.compute_score(
                    data_source=data_source,
                    solution_strs=response_strs,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    prompt = prompt_str,
                )
            elif data_source in ['hoanganh/Medical-Train', 'TsinghuaC3I/MedXpertQA', 'hoanganh/MedQA-Test']:
                result = [medical.compute_score(response_str, ground_truth) for response_str in response_strs]
            elif data_source in ['google/IFEval']:
                result = [ifeval.compute_score(response_str, ground_truth) for response_str in response_strs]
            elif data_source in ['openai/HealthBench']:
                result = [healthbench.compute_score(response_str, ground_truth) for response_str in response_strs]
            else:
                raise NotImplementedError(f"Reward function is not implemented for {data_source}")
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = [_valid_response_length - expected_len for _valid_response_length in valid_response_length]
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = [min(-_exceed_len / overlong_buffer_len * overlong_penalty_factor, 0) for _exceed_len in exceed_len]
                reward = [r + o for r, o in zip(reward, overlong_reward)]
                # if self.overlong_buffer_cfg.log:
                #     reward_extra_info["overlong_reward"].append(overlong_reward)
                #     reward_extra_info["overlong"].append(overlong_reward < 0)

            for _i in range(i, min(i + self.n_resp_per_prompt, len(reward_tensor))):
                reward_tensor[_i, valid_response_length[_i - i] - 1] = reward[_i - i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
