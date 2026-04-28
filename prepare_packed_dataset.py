import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

import hnet.training.dataset_template as dataset_template
from hnet.training import DatasetSource
from hnet.training.data import DefaultRecordFormatter
from hnet.utils.tokenizers import ByteTokenizer


TEMPLATE_CHOICES = sorted(
    name for name in dir(dataset_template) if name.startswith("SOURCES_")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare packed byte-token dataset as dataset-wise multi-shards "
            "(data-XXXXX.bin + data-XXXXX.idx + manifest.json + mix_manifest.json)"
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Hugging Face dataset name. Repeat to specify multiple datasets.",
    )
    parser.add_argument(
        "--dataset-template",
        type=str,
        choices=TEMPLATE_CHOICES,
        help="Named dataset template from hnet.training.dataset_template.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for packed dataset files.",
    )
    parser.add_argument(
        "--max-shard-tokens",
        type=int,
        default=100_000_000,
        help="Maximum byte-tokens per shard file.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of parallel dataset-source packing processes.",
    )
    parser.add_argument(
        "--no-add-bos",
        action="store_true",
        help="Do not prepend BOS token.",
    )
    parser.add_argument(
        "--no-add-eos",
        action="store_true",
        help="Do not append EOS token.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N records per source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed written to mix_manifest for downstream mixing reproducibility.",
    )
    return parser.parse_args()


def resolve_sources(args: argparse.Namespace) -> list[DatasetSource]:
    if args.datasets:
        return [DatasetSource(name=name) for name in args.datasets]
    if args.dataset_template:
        return list(getattr(dataset_template, args.dataset_template))
    return [
        DatasetSource(name="if001/bunpo_phi4_ctx"),
        DatasetSource(name="if001/bunpo_phi4"),
    ]


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "dataset"


def _source_alias(source: DatasetSource, index: int) -> str:
    base = _safe_component(source.name.replace("/", "__"))
    if source.config_name:
        base += "__" + _safe_component(source.config_name)
    base += f"__{source.split}"
    return f"{index:02d}_{base}"


def iter_source_records(source: DatasetSource):
    dataset = load_dataset(
        source.name,
        source.config_name,
        split=source.split,
        streaming=True,
    )
    if source.skip_examples > 0:
        dataset = dataset.skip(source.skip_examples)
    if source.take_examples > 0:
        dataset = dataset.take(source.take_examples)
    return dataset


def _flush_shard(
    source_dir: Path,
    shard_index: int,
    shard_chunks: list[np.ndarray],
    shard_offsets: list[int],
) -> dict[str, object]:
    shard_name = f"data-{shard_index:05d}"
    bin_path = source_dir / f"{shard_name}.bin"
    idx_path = source_dir / f"{shard_name}.idx"

    token_count = int(sum(int(chunk.shape[0]) for chunk in shard_chunks))
    with bin_path.open("wb") as f:
        for chunk in shard_chunks:
            f.write(chunk.tobytes())
    np.asarray(shard_offsets, dtype=np.uint64).tofile(idx_path)

    return {
        "name": shard_name,
        "bin_file": bin_path.name,
        "idx_file": idx_path.name,
        "token_count": token_count,
        "doc_count": max(0, len(shard_offsets) - 1),
    }


def pack_single_source(
    source_dict: dict[str, object],
    source_dir: str,
    add_bos: bool,
    add_eos: bool,
    max_shard_tokens: int,
    progress_every: int,
) -> dict[str, object]:
    source = DatasetSource(**source_dict)
    output_dir = Path(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteTokenizer()
    formatter = DefaultRecordFormatter()

    total_records_seen = 0
    total_records_used = 0
    total_tokens = 0

    shard_index = 0
    shard_token_count = 0
    shard_chunks: list[np.ndarray] = []
    shard_offsets: list[int] = [0]
    shard_meta: list[dict[str, object]] = []

    for record in iter_source_records(source):
        total_records_seen += 1
        text = formatter.format_record(record)
        if not text:
            if progress_every > 0 and total_records_seen % progress_every == 0:
                print(
                    f"progress source={source.name} seen={total_records_seen} used={total_records_used} tokens={total_tokens}"
                )
            continue

        encoded = tokenizer.encode(
            [text],
            add_bos=add_bos,
            add_eos=add_eos,
        )[0]["input_ids"]
        token_count = int(encoded.shape[0])

        if shard_token_count > 0 and shard_token_count + token_count > max_shard_tokens:
            shard_meta.append(
                _flush_shard(
                    source_dir=output_dir,
                    shard_index=shard_index,
                    shard_chunks=shard_chunks,
                    shard_offsets=shard_offsets,
                )
            )
            shard_index += 1
            shard_token_count = 0
            shard_chunks = []
            shard_offsets = [0]

        shard_chunks.append(encoded)
        shard_token_count += token_count
        shard_offsets.append(shard_offsets[-1] + token_count)

        total_tokens += token_count
        total_records_used += 1

        if progress_every > 0 and total_records_seen % progress_every == 0:
            print(
                f"progress source={source.name} seen={total_records_seen} used={total_records_used} tokens={total_tokens}"
            )

    if shard_chunks:
        shard_meta.append(
            _flush_shard(
                source_dir=output_dir,
                shard_index=shard_index,
                shard_chunks=shard_chunks,
                shard_offsets=shard_offsets,
            )
        )

    manifest = {
        "format": "hnet_packed_source_v2",
        "dtype": "uint8",
        "tokenizer": "ByteTokenizer",
        "add_bos": add_bos,
        "add_eos": add_eos,
        "source": asdict(source),
        "total_records_seen": total_records_seen,
        "total_records_used": total_records_used,
        "total_tokens": total_tokens,
        "shard_count": len(shard_meta),
        "shards": shard_meta,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "source": asdict(source),
        "source_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "total_records_seen": total_records_seen,
        "total_records_used": total_records_used,
        "total_tokens": total_tokens,
        "shard_count": len(shard_meta),
    }


def main() -> None:
    args = parse_args()
    if args.max_shard_tokens <= 0:
        raise ValueError("--max-shard-tokens must be > 0")
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be > 0")

    sources = resolve_sources(args)
    add_bos = not args.no_add_bos
    add_eos = not args.no_add_eos

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_root = output_dir / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    print("packing_mode=dataset_wise_multi_shard")
    print(f"num_sources={len(sources)}")
    print(f"max_shard_tokens={args.max_shard_tokens}")
    print(f"num_proc={args.num_proc}")
    print(f"output_dir={output_dir}")

    tasks: list[tuple[DatasetSource, Path]] = []
    for idx, source in enumerate(sources, start=1):
        alias = _source_alias(source, idx)
        source_dir = datasets_root / alias
        tasks.append((source, source_dir))

    results: list[dict[str, object]] = []
    if args.num_proc == 1:
        for source, source_dir in tasks:
            print(f"packing source={source.name} -> {source_dir}")
            result = pack_single_source(
                source_dict=asdict(source),
                source_dir=str(source_dir),
                add_bos=add_bos,
                add_eos=add_eos,
                max_shard_tokens=args.max_shard_tokens,
                progress_every=args.progress_every,
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
            futures = [
                executor.submit(
                    pack_single_source,
                    source_dict=asdict(source),
                    source_dir=str(source_dir),
                    add_bos=add_bos,
                    add_eos=add_eos,
                    max_shard_tokens=args.max_shard_tokens,
                    progress_every=args.progress_every,
                )
                for source, source_dir in tasks
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda x: str(x["source_dir"]))
    total_records_seen = int(sum(int(r["total_records_seen"]) for r in results))
    total_records_used = int(sum(int(r["total_records_used"]) for r in results))
    total_tokens = int(sum(int(r["total_tokens"]) for r in results))
    total_shards = int(sum(int(r["shard_count"]) for r in results))

    dataset_entries = []
    for item in results:
        source_dict = item["source"]
        assert isinstance(source_dict, dict)
        take_examples = int(source_dict.get("take_examples", -1))
        weight = float(take_examples if take_examples > 0 else 1.0)
        dataset_entries.append(
            {
                "name": str(source_dict.get("name", "")),
                "config_name": source_dict.get("config_name"),
                "split": str(source_dict.get("split", "train")),
                "take_examples": take_examples,
                "skip_examples": int(source_dict.get("skip_examples", 0)),
                "weight": weight,
                "source_dir": str(Path(item["source_dir"]).relative_to(output_dir)),
                "manifest": str(
                    Path(item["manifest_path"]).relative_to(output_dir)
                ),
                "total_tokens": int(item["total_tokens"]),
            }
        )

    weight_sum = sum(float(entry["weight"]) for entry in dataset_entries)
    if weight_sum > 0:
        for entry in dataset_entries:
            entry["weight"] = float(entry["weight"]) / weight_sum

    mix_manifest = {
        "format": "hnet_packed_mix_v1",
        "seed": args.seed,
        "dtype": "uint8",
        "tokenizer": "ByteTokenizer",
        "add_bos": add_bos,
        "add_eos": add_eos,
        "total_records_seen": total_records_seen,
        "total_records_used": total_records_used,
        "total_tokens": total_tokens,
        "total_shards": total_shards,
        "datasets": dataset_entries,
    }
    mix_manifest_path = output_dir / "mix_manifest.json"
    mix_manifest_path.write_text(
        json.dumps(mix_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== PACKING COMPLETE ===")
    print(f"mix_manifest={mix_manifest_path}")
    print(f"total_records_seen={total_records_seen}")
    print(f"total_records_used={total_records_used}")
    print(f"total_tokens={total_tokens}")
    print(f"total_shards={total_shards}")


if __name__ == "__main__":
    main()
