# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List
import re


def parse_alfworld_action_response(response: str):
    """Return the exact trimmed action body and structural format validity."""
    lowered = response.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None, False
    action_body = response[start_idx + len(start_tag):end_idx].strip()
    has_think = response.find("<think>") != -1 and response.find("</think>") != -1
    has_chinese = re.search(r'[\u4e00-\u9fff]', response) is not None
    return action_body, bool(action_body) and has_think and not has_chinese


def alfworld_action_identity(response: str, projected_action: str,
                             format_valid: bool, environment_valid: bool):
    if not format_valid:
        return None
    if environment_valid:
        return f"valid::{projected_action}"
    raw_action_body, _ = parse_alfworld_action_response(response)
    return f"invalid::{raw_action_body}"

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    # Do not mutate the decoded model responses. BACE needs the exact CoT/action
    # response while the environment consumes only the parsed action body.
    projected_actions = list(actions)
    valids = [0] * len(projected_actions)

    for i in range(len(projected_actions)):
        original_str = projected_actions[i]
        lowered = original_str.lower()
        raw_action_body, format_valid = parse_alfworld_action_response(original_str)

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lowered.find(start_tag)
        end_idx = lowered.find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                projected_actions[i] = lowered[-30:]
                continue

            # Extract just the content between the tags
            extracted_action = raw_action_body.lower()
            
            projected_actions[i] = extracted_action
            valids[i] = int(format_valid)

        except:
            projected_actions[i] = lowered[-30:]

    return projected_actions, valids
