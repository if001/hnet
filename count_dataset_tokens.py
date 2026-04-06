import argparse
from dataclasses import asdict

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
        description="Count byte-level tokens for datasets/templates using H-Net tokenizer"
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
        "--add-bos",
        action="store_true",
        help="Include BOS token in token counting.",
    )
    parser.add_argument(
        "--add-eos",
        action="store_true",
        help="Include EOS token in token counting.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N records per source (default: 50000)",
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


def count_source_tokens(
    source: DatasetSource,
    tokenizer: ByteTokenizer,
    formatter: DefaultRecordFormatter,
    add_bos: bool,
    add_eos: bool,
    progress_every: int,
) -> tuple[int, int, int]:
    records_seen = 0
    records_used = 0
    total_tokens = 0

    for record in iter_source_records(source):
        records_seen += 1
        text = formatter.format_record(record)
        if not text:
            if progress_every > 0 and records_seen % progress_every == 0:
                print(
                    f"progress source={source.name} records_seen={records_seen} records_used={records_used} tokens={total_tokens}"
                )
            continue

        encoded = tokenizer.encode([text], add_bos=add_bos, add_eos=add_eos)[0][
            "input_ids"
        ]
        total_tokens += int(encoded.shape[0])
        records_used += 1

        if progress_every > 0 and records_seen % progress_every == 0:
            print(
                f"progress source={source.name} records_seen={records_seen} records_used={records_used} tokens={total_tokens}"
            )

    return records_seen, records_used, total_tokens


def main() -> None:
    args = parse_args()
    sources = resolve_sources(args)

    formatter = DefaultRecordFormatter()
    tokenizer = ByteTokenizer()

    print("counting_mode=byte_tokenizer")
    print(f"add_bos={args.add_bos} add_eos={args.add_eos}")
    print(f"num_sources={len(sources)}")

    grand_records_seen = 0
    grand_records_used = 0
    grand_tokens = 0

    for index, source in enumerate(sources, start=1):
        print(f"\n[{index}/{len(sources)}] source={asdict(source)}")
        records_seen, records_used, tokens = count_source_tokens(
            source=source,
            tokenizer=tokenizer,
            formatter=formatter,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            progress_every=args.progress_every,
        )

        grand_records_seen += records_seen
        grand_records_used += records_used
        grand_tokens += tokens

        print(
            f"result source={source.name} records_seen={records_seen} records_used={records_used} tokens={tokens}"
        )

    print("\n=== TOTAL ===")
    print(f"total_records_seen={grand_records_seen}")
    print(f"total_records_used={grand_records_used}")
    print(f"total_tokens={grand_tokens}")


if __name__ == "__main__":
    main()
