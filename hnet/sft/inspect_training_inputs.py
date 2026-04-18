from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as:
#   python hnet/sft/inspect_training_inputs.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hnet.sft.trainer import StreamingSFTByteDataset
from hnet.utils.tokenizers import ByteTokenizer


def decode_tokens_for_display(token_ids: list[int], tokenizer: ByteTokenizer) -> str:
    chunks: list[str] = []
    buf = bytearray()

    def flush_buf() -> None:
        if not buf:
            return
        chunks.append(buf.decode("utf-8", errors="replace"))
        buf.clear()

    for token_id in token_ids:
        if token_id == tokenizer.bos_idx:
            flush_buf()
            chunks.append("<BOS>")
            continue
        if token_id == tokenizer.eos_idx:
            flush_buf()
            chunks.append("<EOS>")
            continue
        buf.append(int(token_id))

    flush_buf()
    return "".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect decoded training inputs used by SFT pipeline"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of training samples (chunks) to print",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chat-tokenizer-path",
        type=str,
        default="Qwen/Qwen3-0.6B-Instruct",
        help="Tokenizer path used for apply_chat_template",
    )
    parser.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Print raw token ids as well",
    )
    return parser.parse_args()


"""
sftのinput_ids, labelsの中身を確認する
python hnet/sft/inspect_training_inputs.py --num-samples 3 --seq-len 512
"""


def main() -> None:
    args = parse_args()
    byte_tokenizer = ByteTokenizer()

    dataset = StreamingSFTByteDataset(
        seq_len=args.seq_len,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        chat_tokenizer_path=args.chat_tokenizer_path,
    )

    iterator = iter(dataset)
    for idx in range(1, args.num_samples + 1):
        sample = next(iterator)
        input_ids = sample["input_ids"].tolist()
        labels = sample["labels"].tolist()

        input_text = decode_tokens_for_display(input_ids, byte_tokenizer)
        label_text = decode_tokens_for_display(labels, byte_tokenizer)

        print(f"\n[{idx}/{args.num_samples}]")
        print("input_text:")
        print(input_text)
        print("label_text:")
        print(label_text)

        if args.show_token_ids:
            print("input_ids:")
            print(input_ids)
            print("labels:")
            print(labels)


if __name__ == "__main__":
    main()
