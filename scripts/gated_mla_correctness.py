"""CUDA correctness checks for H-Net's gated MLA and KDA+MLA models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from hnet.models.config_io import load_hnet_config, save_hnet_config
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.modules.mla import GatedMLA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-config", default="configs/hnet_1stage_tiny_g1.json")
    parser.add_argument(
        "--k1g1-config", default="configs/hnet_1stage_tiny_k1g1.json"
    )
    parser.add_argument(
        "--two-stage-config", default="configs/hnet_2stage_tiny_main_k1g1.json"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overfit-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AssertionError(f"{name} contains non-finite values")


def make_layer() -> GatedMLA:
    return GatedMLA(
        d_model=64,
        num_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=12,
        use_output_gate=True,
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )


def check_backward(layer: GatedMLA) -> float:
    layer.train()
    hidden = torch.randn(
        2, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    loss = layer(hidden).float().square().mean()
    loss.backward()
    assert_finite("gated MLA loss", loss)
    gradients = [
        parameter.grad
        for parameter in layer.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError("gated MLA backward produced no gradients")
    for gradient in gradients:
        assert_finite("gated MLA gradient", gradient)
    return float(loss.detach())


@torch.no_grad()
def check_packed_isolation(layer: GatedMLA) -> float:
    layer.eval()
    first_len, second_len = 31, 37
    hidden = torch.randn(
        first_len + second_len, 64, device="cuda", dtype=torch.bfloat16
    )
    packed_cu = torch.tensor(
        [0, first_len, first_len + second_len], device="cuda", dtype=torch.int32
    )
    packed = layer(hidden, cu_seqlens=packed_cu, max_seqlen=second_len)
    separated = []
    for start, end in ((0, first_len), (first_len, first_len + second_len)):
        local_cu = torch.tensor([0, end - start], device="cuda", dtype=torch.int32)
        separated.append(
            layer(hidden[start:end], cu_seqlens=local_cu, max_seqlen=end - start)
        )
    reference = torch.cat(separated, dim=0)
    error = float((packed - reference).abs().max())
    if not torch.allclose(packed, reference, atol=2e-2, rtol=2e-2):
        raise AssertionError(f"packed document state leakage: max_abs_error={error}")
    return error


@torch.no_grad()
def check_right_padding_invariance(layer: GatedMLA) -> float:
    layer.eval()
    lengths = (31, 37)
    max_length = max(lengths)
    hidden = torch.randn(
        len(lengths), max_length, 64, device="cuda", dtype=torch.bfloat16
    )
    padded = layer(hidden)
    errors = []
    for batch_index, length in enumerate(lengths):
        reference = layer(hidden[batch_index : batch_index + 1, :length])
        errors.append(
            float((padded[batch_index, :length] - reference.squeeze(0)).abs().max())
        )
    error = max(errors)
    if error > 2e-2:
        raise AssertionError(f"right padding changed real tokens: max_abs_error={error}")
    return error


@torch.no_grad()
def check_causality(layer: GatedMLA) -> float:
    layer.eval()
    prefix_length, total_length = 29, 53
    hidden = torch.randn(1, total_length, 64, device="cuda", dtype=torch.bfloat16)
    changed = hidden.clone()
    changed[:, prefix_length:] = torch.randn_like(changed[:, prefix_length:])
    baseline = layer(hidden)[:, :prefix_length]
    mutated = layer(changed)[:, :prefix_length]
    error = float((baseline - mutated).abs().max())
    if error > 2e-2:
        raise AssertionError(f"future tokens changed the prefix: max_abs_error={error}")
    return error


@torch.no_grad()
def check_recurrent_decode(layer: GatedMLA) -> float:
    layer.eval()
    length = 48
    hidden = torch.randn(1, length, 64, device="cuda", dtype=torch.bfloat16)
    full = layer(hidden)
    cache = SimpleNamespace(
        max_batch_size=1,
        max_seqlen=length,
        batch_size_offset=0,
        seqlen_offset=0,
        key_value_memory_dict={0: layer.allocate_inference_cache(1, length)},
    )
    outputs = []
    for index in range(length):
        outputs.append(layer.step(hidden[:, index : index + 1], cache))
        cache.seqlen_offset += 1
    recurrent = torch.cat(outputs, dim=1)
    error = float((full - recurrent).abs().max())
    if not torch.allclose(full, recurrent, atol=2e-2, rtol=2e-2):
        raise AssertionError(f"full/recurrent mismatch: max_abs_error={error}")
    return error


def make_model(config_path: str) -> HNetForCausalLM:
    config = load_hnet_config(config_path)
    model = HNetForCausalLM(config, device="cuda", dtype=torch.bfloat16)
    model.init_weights()
    return model


def patterned_batch() -> tuple[torch.Tensor, torch.Tensor]:
    text = ("今日は良い天気です。Gated MLAの動作を確認します。" * 5).encode("utf-8")
    sequence = torch.tensor(list(text[:129]), device="cuda", dtype=torch.long)
    if sequence.numel() < 129:
        repeats = (129 + sequence.numel() - 1) // sequence.numel()
        sequence = sequence.repeat(repeats)[:129]
    return sequence[:-1].view(2, 64), sequence[1:].view(2, 64)


def training_step(
    model: HNetForCausalLM,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=inputs)
    loss = F.cross_entropy(output.logits.float().reshape(-1, 256), labels.reshape(-1))
    assert_finite("training loss", loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert_finite("model gradient", parameter.grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def check_overfit_and_resume(
    config_path: str, output_dir: Path, name: str, steps: int
) -> dict[str, object]:
    model = make_model(config_path)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    inputs, labels = patterned_batch()
    losses = [training_step(model, optimizer, inputs, labels) for _ in range(steps)]
    if losses[-1] >= losses[0]:
        raise AssertionError(f"short overfit did not reduce CE: {losses[0]} -> {losses[-1]}")

    checkpoint_path = output_dir / f"{name}_checkpoint_resume_probe.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": steps},
        checkpoint_path,
    )
    expected_loss = training_step(model, optimizer, inputs, labels)

    resumed_model = make_model(config_path)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=3e-3)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    resumed_model.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_loss = training_step(resumed_model, resumed_optimizer, inputs, labels)
    if abs(expected_loss - resumed_loss) > 1e-5:
        raise AssertionError(
            f"resume discontinuity: expected={expected_loss}, resumed={resumed_loss}"
        )
    return {
        "initial_ce": losses[0],
        "final_ce": losses[-1],
        "post_checkpoint_ce": expected_loss,
        "resumed_ce": resumed_loss,
        "checkpoint": str(checkpoint_path),
    }


@torch.no_grad()
def check_model_recurrent_decode(config_path: str) -> float:
    model = make_model(config_path).eval()
    length = 48
    inputs = torch.randint(0, 256, (1, length), device="cuda")
    full_logits = model(input_ids=inputs).logits
    cache = model.allocate_inference_cache(1, length, dtype=torch.bfloat16)
    recurrent_logits = torch.cat(
        [model.step(inputs[:, index : index + 1], cache).logits for index in range(length)],
        dim=1,
    )
    error = float((full_logits - recurrent_logits).abs().max())
    if not torch.allclose(full_logits, recurrent_logits, atol=5e-2, rtol=5e-2):
        raise AssertionError(f"model full/recurrent mismatch: max_abs_error={error}")
    return error


@torch.no_grad()
def check_model_packed_isolation(config_path: str) -> float:
    model = make_model(config_path).eval()
    # HNet's packed API uses a fixed length per row, so use equal-sized documents.
    length = 37
    inputs = torch.randint(0, 256, (2, length), device="cuda")
    packed = model(input_ids=inputs).logits
    reference = torch.cat(
        [model(input_ids=inputs[index : index + 1]).logits for index in range(2)],
        dim=0,
    )
    error = float((packed - reference).abs().max())
    if not torch.allclose(packed, reference, atol=5e-2, rtol=5e-2):
        raise AssertionError(f"model packed document leakage: max_abs_error={error}")
    return error


@torch.no_grad()
def check_model_right_padding(config_path: str) -> float:
    model = make_model(config_path).eval()
    lengths = (31, 37)
    max_length = max(lengths)
    inputs = torch.randint(0, 256, (2, max_length), device="cuda")
    mask = torch.arange(max_length, device="cuda").unsqueeze(0) < torch.tensor(
        lengths, device="cuda"
    ).unsqueeze(1)
    padded = model(input_ids=inputs, mask=mask).logits
    errors = []
    for index, length in enumerate(lengths):
        local_mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
        reference = model(
            input_ids=inputs[index : index + 1, :length], mask=local_mask
        ).logits
        errors.append(
            float((padded[index, :length] - reference.squeeze(0)).abs().max())
        )
    error = max(errors)
    if error > 5e-2:
        raise AssertionError(f"model right padding changed real tokens: max_abs_error={error}")
    return error


def check_two_stage(config_path: str) -> dict[str, object]:
    model = make_model(config_path)
    model.train()
    inputs, labels = patterned_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = training_step(model, optimizer, inputs, labels)
    return {
        "loss": loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("gated MLA correctness checks require CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    saved_configs = {}
    for name, config_path in (
        ("g1", args.g1_config),
        ("k1g1", args.k1g1_config),
        ("two_stage_k1g1", args.two_stage_config),
    ):
        saved_configs[name] = str(
            save_hnet_config(load_hnet_config(config_path), output_dir / f"{name}_config.json")
        )

    layer = make_layer()
    results = {
        "status": "passed",
        "seed": args.seed,
        "configs": saved_configs,
        "backward_loss": check_backward(layer),
        "packed_isolation_max_abs_error": check_packed_isolation(layer),
        "right_padding_prefix_max_abs_error": check_right_padding_invariance(layer),
        "causal_prefix_max_abs_error": check_causality(layer),
        "layer_recurrent_decode_max_abs_error": check_recurrent_decode(layer),
        "g1_model_recurrent_decode_max_abs_error": check_model_recurrent_decode(
            args.g1_config
        ),
        "k1g1_model_recurrent_decode_max_abs_error": check_model_recurrent_decode(
            args.k1g1_config
        ),
        "g1_model_packed_isolation_max_abs_error": check_model_packed_isolation(
            args.g1_config
        ),
        "k1g1_model_packed_isolation_max_abs_error": check_model_packed_isolation(
            args.k1g1_config
        ),
        "g1_model_right_padding_max_abs_error": check_model_right_padding(
            args.g1_config
        ),
        "k1g1_model_right_padding_max_abs_error": check_model_right_padding(
            args.k1g1_config
        ),
        "g1_overfit_resume": check_overfit_and_resume(
            args.g1_config, output_dir, "g1", args.overfit_steps
        ),
        "k1g1_overfit_resume": check_overfit_and_resume(
            args.k1g1_config, output_dir, "k1g1", args.overfit_steps
        ),
        "two_stage_forward_backward": check_two_stage(args.two_stage_config),
    }
    results["elapsed_seconds"] = time.perf_counter() - started_at
    results["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)

    environment = {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git_output("status", "--porcelain")),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "flash_attn": importlib.metadata.version("flash-attn"),
        "fla_core": importlib.metadata.version("fla-core"),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "correctness_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"environment": environment, "results": results}, indent=2))


if __name__ == "__main__":
    main()
