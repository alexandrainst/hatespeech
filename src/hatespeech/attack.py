"""Loads the A-ttack model."""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.utils.logging as hf_logging
import wget
from transformers import ElectraModel, ElectraTokenizer


def load_attack() -> Tuple[ElectraTokenizer, nn.Module]:
    """Loads the A-ttack model.

    Returns:
        pair of Hugging Face tokenizer and PyTorch Module:
            The A-ttack tokenizer and model.
    """
    # Load the tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(
        "Maltehb/aelaectra-danish-electra-small-cased",
        cache_dir=".cache",
    )

    # Load the model architecture
    hf_logging.set_verbosity_error()
    model = ElectraClassifier(
        pretrained_model_name="Maltehb/aelaectra-danish-electra-small-cased"
    )

    # Set the path to the state dict
    attack_dir = Path("models") / "attack"
    state_dict_path = attack_dir / "state_dict.bin"

    # If the attack directory doesn't exist then create it
    if not attack_dir.exists():
        attack_dir.mkdir()

    # Download the state dict if it doesn't exist
    if not state_dict_path.exists():
        state_dict_url = "https://github.com/ogtal/A-ttack/raw/main/pytorch_model.bin"
        wget.download(state_dict_url, out=str(state_dict_path))

    # Load the state dict
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    return tokenizer, model


class ElectraClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(
            pretrained_model_name,
            cache_dir=".cache",
        )
        self.dense = nn.Linear(
            self.electra.config.hidden_size, self.electra.config.hidden_size
        )
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra.config.hidden_size, self.num_labels)

    def classifier(self, sequence_output):
        x = sequence_output[:, 0, :]
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

    def forward(self, input_ids=None, attention_mask=None):
        discriminator_hidden_states = self.electra(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        return logits


if __name__ == "__main__":
    tok, model = load_attack()
    breakpoint()
