from tokenizers import Tokenizer, models, pre_tokenizers, processors

from hnet.sft.data import StreamingSFTByteDataset


def test_sft_dataset_can_encode_with_baseline_tokenizer(tmp_path) -> None:
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={"<unk>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    dataset = StreamingSFTByteDataset(
        seq_len=3,
        packing=True,
        shuffle_buffer_size=1,
        seed=42,
        chat_tokenizer_path="unused",
        model_tokenizer_path=str(tokenizer_path),
    )
    dataset._iter_texts = lambda: iter(["hello world"])  # type: ignore[method-assign]

    samples = list(dataset)

    assert len(samples) == 1
    assert samples[0]["input_ids"].tolist() == [1, 3, 4]
    assert samples[0]["labels"].tolist() == [3, 4, 2]
