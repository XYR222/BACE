from __future__ import annotations

import numpy as np

import verl.utils.torch_functional as verl_F

from .validator import ReplayCategory, ReplayValidator


class ReplayAdapter:
    def __init__(self, manager, tokenizer, config, compare_action_set: bool = True):
        self.manager = manager
        self.tokenizer = tokenizer
        self.config = config
        self.validator = ReplayValidator(
            compare_action_set=compare_action_set,
            action_is_executable=getattr(manager, "is_action_executable", None),
        )

    def _prompt_token_ids(self, prompt: str) -> tuple[int, ...]:
        rendered = self.tokenizer.apply_chat_template(
            np.array([{"content": prompt, "role": "user"}]),
            add_generation_prompt=True,
            tokenize=False,
            **self.config.data.get("apply_chat_template_kwargs", {}),
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=rendered,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        active = attention_mask[0].bool()
        return tuple(input_ids[0][active].detach().cpu().tolist())

    def replay_and_validate(self, requests):
        try:
            observations, dones, _ = self.manager.replay(requests)
        except Exception as error:
            return [self.validator.from_exception(request, error) for request in requests]

        results = []
        for idx, request in enumerate(requests):
            result = self.validator.validate(
                request=request,
                observation=observations["anchor"][idx],
                action_set=observations["admissible_actions"][idx],
                done=bool(dones[idx]),
            )
            if result.replay_ok:
                rebuilt_prompt_ids = self._prompt_token_ids(observations["text"][idx])
                if rebuilt_prompt_ids != request.original_prompt_token_ids:
                    result = type(result)(
                        **{
                            **result.__dict__,
                            "replay_ok": False,
                            "category": ReplayCategory.PROMPT_IDENTITY_MISMATCH.value,
                            "error_message": "Restored prompt token IDs do not match the natural origin",
                        }
                    )
            results.append(result)
        return results

    def replay_selected_and_validate(self, worker_indices, requests):
        """Replay onto persistent sibling slots used by frontier scheduling."""
        try:
            observations, dones, _ = self.manager.replay_selected(worker_indices, requests)
        except Exception as error:
            return None, None, [
                self.validator.from_exception(request, error) for request in requests
            ]

        results = []
        for idx, request in enumerate(requests):
            result = self.validator.validate(
                request=request,
                observation=observations["anchor"][idx],
                action_set=observations["admissible_actions"][idx],
                done=bool(dones[idx]),
            )
            if result.replay_ok:
                rebuilt_prompt_ids = self._prompt_token_ids(observations["text"][idx])
                if rebuilt_prompt_ids != request.original_prompt_token_ids:
                    result = type(result)(
                        **{
                            **result.__dict__,
                            "replay_ok": False,
                            "category": ReplayCategory.PROMPT_IDENTITY_MISMATCH.value,
                            "error_message": (
                                "Restored prompt token IDs do not match the natural origin"
                            ),
                        }
                    )
            results.append(result)
        return observations, np.asarray(dones, dtype=bool), results

    def validate_transitions(self, requests, observations, rewards, dones, infos):
        return [
            self.validator.validate_transition(
                request=request,
                observation=observations["anchor"][idx],
                reward=float(rewards[idx]),
                done=bool(dones[idx]),
                action_identity=infos[idx].get("action_identity"),
            )
            for idx, request in enumerate(requests)
        ]


# Compatibility for code importing the original Stage-1 environment-specific name.
AlfworldReplayAdapter = ReplayAdapter
