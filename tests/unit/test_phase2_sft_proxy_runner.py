from argparse import Namespace
from pathlib import Path

from scripts.run_phase2_sft_proxy import build_command


def _args(model: str) -> Namespace:
    return Namespace(
        model=model,
        pretrained_model_path=Path("checkpoint.pt"),
        pretraining_seed=42,
        sft_seed=42,
        seq_len=8192,
        batch_size=1,
        grad_accum_steps=16,
        max_steps=10,
        mix_config_path=Path("mix.json"),
        model_tokenizer_path=Path("tokenizer.json"),
    )


def test_tokenizer_sft_uses_common_text_mix_and_model_tokenizer() -> None:
    command = build_command(_args("tokenizer"), Path("output"))
    assert command[command.index("--mix-config-path") + 1] == "mix.json"
    assert command[command.index("--model-tokenizer-path") + 1] == "tokenizer.json"
    assert command[command.index("--seq-len") + 1] == "8192"


def test_hnet_sft_keeps_byte_input() -> None:
    command = build_command(_args("k1g1"), Path("output"))
    assert "--model-tokenizer-path" not in command
    assert command.count("--compression-ratio") == 2
