"""CUDA correctness checks for H-Net's Kimi Delta Attention adapter."""

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
from hnet.modules.kda import KimiDeltaAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-stage-config", default="configs/hnet_1stage_tiny_k3t1.json"
    )
    parser.add_argument(
        "--two-stage-config", default="configs/hnet_2stage_tiny_main_k3t1.json"
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


def make_layer(*, k3_gate: bool) -> KimiDeltaAttention:
    return KimiDeltaAttention(
        d_model=64,
        num_heads=4,
        head_dim=16,
        short_conv_kernel_size=4,
        use_full_rank_gate=k3_gate,
        gate_lower_bound=-5.0 if k3_gate else None,
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )


@torch.no_grad()
def check_packed_isolation(layer: KimiDeltaAttention) -> float:
    layer.eval()
    first_len, second_len = 31, 37
    hidden = torch.randn(
        1, first_len + second_len, 64, device="cuda", dtype=torch.bfloat16
    )
    packed_cu = torch.tensor(
        [0, first_len, first_len + second_len], device="cuda", dtype=torch.int32
    )
    packed = layer(hidden, cu_seqlens=packed_cu)
    separated = []
    for start, end in ((0, first_len), (first_len, first_len + second_len)):
        local_cu = torch.tensor([0, end - start], device="cuda", dtype=torch.int32)
        separated.append(layer(hidden[:, start:end], cu_seqlens=local_cu))
    reference = torch.cat(separated, dim=1)
    error = float((packed - reference).abs().max())
    if not torch.allclose(packed, reference, atol=2e-2, rtol=2e-2):
        raise AssertionError(f"packed document state leakage: max_abs_error={error}")
    return error


@torch.no_grad()
def check_padded_isolation(layer: KimiDeltaAttention) -> float:
    layer.eval()
    lengths = (31, 37)
    hidden = torch.randn(2, max(lengths), 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.arange(max(lengths), device="cuda").unsqueeze(0) < torch.tensor(
        lengths, device="cuda"
    ).unsqueeze(1)
    padded = layer(hidden, attention_mask=mask)
    references = []
    for batch_index, length in enumerate(lengths):
        local_cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        references.append(layer(hidden[batch_index : batch_index + 1, :length], cu_seqlens=local_cu))
    error = max(
        float((padded[index, :length] - references[index].squeeze(0)).abs().max())
        for index, length in enumerate(lengths)
    )
    if error > 2e-2:
        raise AssertionError(f"padded batch state leakage: max_abs_error={error}")
    for batch_index, length in enumerate(lengths):
        if not bool((padded[batch_index, length:] == 0).all()):
            raise AssertionError("padded KDA output must remain zero")
    return error


@torch.no_grad()
def check_recurrent_decode(layer: KimiDeltaAttention) -> float:
    layer.eval()
    length = 48
    hidden = torch.randn(1, length, 64, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    full = layer(hidden, cu_seqlens=cu_seqlens)
    cache = SimpleNamespace(
        key_value_memory_dict={0: layer.allocate_inference_cache(1, length)}
    )
    recurrent = torch.cat(
        [layer.step(hidden[:, index : index + 1], cache) for index in range(length)],
        dim=1,
    )
    error = float((full - recurrent).abs().max())
    if not torch.allclose(full, recurrent, atol=5e-2, rtol=5e-2):
        raise AssertionError(f"full/recurrent mismatch: max_abs_error={error}")
    return error


def check_backward(layer: KimiDeltaAttention) -> float:
    layer.train()
    hidden = torch.randn(
        2, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    output = layer(hidden, attention_mask=torch.ones(2, 64, device="cuda", dtype=torch.bool))
    loss = output.float().square().mean()
    loss.backward()
    assert_finite("KDA loss", loss)
    gradients = [parameter.grad for parameter in layer.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("KDA backward produced no gradients")
    for gradient in gradients:
        assert_finite("KDA gradient", gradient)
    return float(loss.detach())


def make_model(config_path: str) -> HNetForCausalLM:
    config = load_hnet_config(config_path)
    for stage, (d_model, num_heads, head_dim) in enumerate(
        zip(config.d_model, config.kda_cfg.num_heads, config.kda_cfg.head_dim)
    ):
        if d_model != num_heads * head_dim:
            raise AssertionError(
                f"stage {stage}: d_model={d_model} != num_heads*head_dim={num_heads * head_dim}"
            )
    model = HNetForCausalLM(config, device="cuda", dtype=torch.bfloat16)
    model.init_weights()
    return model


def patterned_batch() -> tuple[torch.Tensor, torch.Tensor]:
    text = ("今日は良い天気です。KDAの動作を確認します。" * 5).encode("utf-8")
    sequence = torch.tensor(list(text[:129]), device="cuda", dtype=torch.long)
    if sequence.numel() < 129:
        repeats = (129 + sequence.numel() - 1) // sequence.numel()
        sequence = sequence.repeat(repeats)[:129]
    inputs = sequence[:-1].view(2, 64)
    labels = sequence[1:].view(2, 64)
    return inputs, labels


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
    config_path: str, output_dir: Path, steps: int
) -> dict[str, object]:
    model = make_model(config_path)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    inputs, labels = patterned_batch()
    losses = [training_step(model, optimizer, inputs, labels) for _ in range(steps)]
    if losses[-1] >= losses[0]:
        raise AssertionError(f"short overfit did not reduce CE: {losses[0]} -> {losses[-1]}")

    checkpoint_path = output_dir / "checkpoint_resume_probe.pt"
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
        raise RuntimeError("KDA correctness checks require CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    config = load_hnet_config(args.one_stage_config)
    saved_config = save_hnet_config(config, output_dir / "model_config.json")
    linear_layer = make_layer(k3_gate=False)
    k3_layer = make_layer(k3_gate=True)
    results = {
        "status": "passed",
        "seed": args.seed,
        "one_stage_config": str(saved_config),
        "linear_gate_backward_loss": check_backward(linear_layer),
        "k3_gate_backward_loss": check_backward(k3_layer),
        "packed_isolation_max_abs_error": check_packed_isolation(k3_layer),
        "padded_isolation_max_abs_error": check_padded_isolation(k3_layer),
        "recurrent_decode_max_abs_error": check_recurrent_decode(k3_layer),
        "one_stage_overfit_resume": check_overfit_and_resume(
            args.one_stage_config, output_dir, args.overfit_steps
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
