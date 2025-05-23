from .base import BaseDataset
from .sft import SFTDataset
from .dpo import DPODataset
from .rl import RLDataset

def tokenize_messages(tokenizer, messages):

    states, actions, action_mask = [], [], []
    for idx, message in enumerate(messages):

        state = tokenizer.apply_chat_template(
            messages[:idx + 1],
            add_generation_prompt=idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant"
        )[len(states):]

        states.extend(state)
        actions.extend(
            state if message["role"] == "assistant"
            else len(state) * [0]
        )
        action_mask.extend(len(state) * [
            1 if message["role"] == "assistant" else 0
        ])

    return {
        "states": states[:-1],
        "actions": actions[1:],
        "action_mask": action_mask[1:],
        "position_ids": list(range(len(states) - 1))
    }