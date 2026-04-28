import argparse
import json
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
        description="Prepare packed byte-token dataset (data.bin / data.idx / metadata.json)"
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


def main() -> None:
    args = parse_args()
    sources = resolve_sources(args)
    add_bos = not args.no_add_bos
    add_eos = not args.no_add_eos

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.bin"
    idx_path = output_dir / "data.idx"
    metadata_path = output_dir / "metadata.json"

    tokenizer = ByteTokenizer()
    formatter = DefaultRecordFormatter()

    print("packing_mode=byte_tokenizer")
    print(f"sources={len(sources)}")
    print(f"add_bos={add_bos} add_eos={add_eos}")
    print(f"output_dir={output_dir}")

    total_records_seen = 0
    total_records_used = 0
    total_tokens = 0
    doc_offsets: list[int] = [0]

    with data_path.open("wb") as data_file:
        for source_index, source in enumerate(sources, start=1):
            print(f"[{source_index}/{len(sources)}] source={asdict(source)}")
            source_seen = 0
            source_used = 0
            source_tokens = 0

            for record in iter_source_records(source):
                source_seen += 1
                total_records_seen += 1

                text = formatter.format_record(record)
                if not text:
                    continue

                encoded = tokenizer.encode(
                    [text],
                    add_bos=add_bos,
                    add_eos=add_eos,
                )[0]["input_ids"]
                data_file.write(encoded.tobytes())

                token_count = int(encoded.shape[0])
                source_tokens += token_count
                total_tokens += token_count
                source_used += 1
                total_records_used += 1
                doc_offsets.append(total_tokens)

                if args.progress_every > 0 and source_seen % args.progress_every == 0:
                    print(
                        f"progress source={source.name} records_seen={source_seen} "
                        f"records_used={source_used} total_tokens={source_tokens}"
                    )

            print(
                f"result source={source.name} records_seen={source_seen} "
                f"records_used={source_used} total_tokens={source_tokens}"
            )

    np.asarray(doc_offsets, dtype=np.uint64).tofile(idx_path)

    metadata = {
        "format": "hnet_packed_byte_v1",
        "dtype": "uint8",
        "tokenizer": "ByteTokenizer",
        "add_bos": add_bos,
        "add_eos": add_eos,
        "total_records_seen": total_records_seen,
        "total_records_used": total_records_used,
        "total_tokens": total_tokens,
        "doc_count": max(0, len(doc_offsets) - 1),
        "sources": [asdict(source) for source in sources],
        "data_file": data_path.name,
        "index_file": idx_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== PACKING COMPLETE ===")
    print(f"data_file={data_path}")
    print(f"index_file={idx_path}")
    print(f"metadata_file={metadata_path}")
    print(f"total_records_seen={total_records_seen}")
    print(f"total_records_used={total_records_used}")
    print(f"total_tokens={total_tokens}")


if __name__ == "__main__":
    main()
