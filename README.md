# H-Net
H-Netの学習コード

Colab上で実行する想定の日本語データセット学習用

## Install
Colab での最小セットアップ例です。

```sh
!pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
!pip install -v mamba-ssm[causal-conv1d] --no-build-isolation
!pip install 'triton>=3.6.0' \
'flash_attn==2.8.0.post2' \
'datasets>=3.0.0' \
einops optree regex omegaconf \
'rich>=13.9.0' \
'matplotlib>=3.9.0'
```

## Train
`--dataset` を指定した場合はそれを優先し、未指定時は `--dataset-template` を使えます。
`--max-steps` を省略した場合は 1epoch で終了します（streaming / packed の両方）。

```sh
!python train.py \
--dataset-template SOURCES_JA9_EN0_CODE1 \
--save-every 1000 \
--output-dir artifacts/hnet_1stage_100m
```

個別 dataset を直接渡す例:

```sh
!python train.py \
--dataset if001/bunpo_phi4_ctx \
--dataset if001/bunpo_phi4 \
--max-steps 2000 \
--save-every 1000 \
--validation-every 100 \
--validation-max-batches 20 \
--output-dir artifacts/hnet_1stage_100m
```

### packed dataset（推奨）
`dataset-template` から事前に `mix_manifest.json` と datasetごとの shard (`data-XXXXX.bin/.idx`) を作成し、学習時は mmap で読み込みます。

```sh
python prepare_packed_dataset.py \
--dataset-template SOURCES_JA8_EN1_CODE1_10 \
--max-shard-tokens 50000000 \
--num-proc 4 \
--index-seq-len 1024 \
--output-dir artifacts/packed/j8e1c1_10
```

```sh
python train.py \
--packed-data-dir artifacts/packed/j8e1c1_10 \
--model-config-path configs/hnet_2stage_200m.json \
--validation-split-ratio 0.05 \
--batch-size 64 \
--grad-accum-steps 4 \
--save-every 4000 \
--output-dir artifacts/hnet_2stage_200m_packed
```

`--seq-len` は packed 生成時の `--index-seq-len` と一致させてください（不一致時は実行時に動的indexへフォールバック）。

検証用 packed データを別途使う場合:

```sh
python train.py \
--packed-data-dir artifacts/packed/train \
--packed-validation-data-dir artifacts/packed/val \
--model-config-path configs/hnet_2stage_200m.json \
--batch-size 64 \
--grad-accum-steps 4 \
--output-dir artifacts/hnet_2stage_200m_packed
```

`prepare_packed_dataset.py` の出力:
- `mix_manifest.json`（全datasetのmix定義）
- `datasets/<source_alias>/manifest.json`
- `datasets/<source_alias>/data-00000.bin`, `data-00000.idx`, ...
- `indices/seq_<N>/sample_index.npy`, `shuffle_index_seed_<seed>.npy`（Megatron方式のsample-level index）

主な出力ファイル:
- `artifacts/.../checkpoint_step_XXXXXX.pt`
- `artifacts/.../model_config.json`
- `artifacts/.../training_metrics.csv`

### 2-stage
chunkingとmainの層で異なるlrを使うために、lr-multiplierを設定する.
2-stageではchukingを２つの層で行う。それぞれの目標圧縮率を指定するためcompression-ratioを設定する

``` sh
!python train.py \
--model-config-path 'configs/hnet_2stage_200m.json' \
--lr-multiplier 3.0 \
--lr-multiplier 1.0 \
--lr-multiplier 1.0 \
--compression-ratio 3 \
--compression-ratio 3 \
--dataset-template 'SOURCES_JA8_EN1_CODE1_10' \
--seq-len 1024 \
--batch-size 64 \
--grad-accum-steps 4 \
--learning-rate 4.0e-4 \
--save-every 4000 \
--max-steps 16000 \
--log-every 2000 \
--validation-every 500 \
--validation-max-batches 20 \
--validation-split-ratio 0.05 \
--shuffle-buffer-size=4096 \
--output-dir "artifacts/hnet_2stage_200m_j8_e1_c1_10"
```

### UTF-8 byte boundary constraint
stage0 の chunk 境界が UTF-8 continuation byte (`0x80..0xBF`) の途中に立ちづらくする soft constraint を有効化できます。
hard mask ではなく、boundary を取りづらくする prior と補助損失です。

```sh
python train.py \
--model-config-path configs/hnet_2stage_200m.json \
--dataset-template SOURCES_JA8_EN1_CODE1_10 \
--compression-ratio 4 \
--compression-ratio 5 \
--byte-boundary-constraint utf8-soft \
--byte-boundary-constraint-bias 0.05 \
--byte-boundary-constraint-weight 0.02 \
--output-dir artifacts/hnet_2stage_200m_utf8_constraint
```

主な引数:
- `--byte-boundary-constraint utf8-soft` で制約を有効化
- `--byte-boundary-constraint-bias` は stage0 routing の boundary 確率を continuation byte 上でどれだけ下げるか
- `--byte-boundary-constraint-weight` は continuation byte 上の boundary 確率に対する補助損失の重み


## Metrics
学習ログは `training_metrics.csv` に保存されます。あとから PNG に描画できます。

```sh
!python plot_training_log.py \
--csv-path artifacts/hnet_1stage_100m/training_metrics.csv \
--output-path artifacts/hnet_1stage_100m/training_metrics.png
```

``` sh
!python plot_validation_log.py \
--csv-path artifacts/hnet_1stage_100m/validation_metrics.csv \
--output-path artifacts/hnet_1stage_100m/validation_metrics.png
```

### loss
- ce_loss は訓練バッチ上の next-byte cross-entropy
- ratio_loss は圧縮率が target ratio に近づくよう促す補助損失
- byte_boundary_loss は UTF-8 continuation byte の途中で chunk を始めにくくする補助損失
- total_loss はその合計

### val
- val_ce は検証データ上の next-byte cross-entropy
- val_bpb はbits-per-byte で、byte-level モデルの主品質指標
どちらも低いほど良いです。

### 2-stage
2-stage の chunking 指標では、
- L1/L0 は入力→stage1 の圧縮率
- L2/L1 はstage1→stage2 の圧縮率
- L2/L0 は入力→stage2 までの総圧縮率

- selected_fraction_s0 は stage1 で保持された割合
- selected_fraction_s1 は stage2 で保持された割合
小さいほど強圧縮を表す

- target_gap_s0, target_gap_s1 は target ratio との差で、これまでの整理では正なら under-compress（保持しすぎ）、負なら over-compress（削りすぎ）と解釈
- stage0_mid_utf8_boundary_fraction は stage0 で continuation byte 上に boundary が立った割合で、小さいほど自然な UTF-8 境界に寄る

## Inference
学習済み checkpoint と保存された `model_config.json` を指定します。

```sh
!python generate.py \
--model-path artifacts/hnet_1stage_100m/checkpoint_step_000020.pt \
--config-path artifacts/hnet_1stage_100m/model_config.json
```

## chunking

``` sh
python inspect_chunking.py \
    --model-path outputs/checkpoint_step_200.pt \
    --config-path outputs/model_config.json \
    --prompt "今日の天気は" \
    --prompt "機械学習の基本は"
```

## token count

``` sh
python count_dataset_tokens.py \
--dataset-template SOURCES_JA8_EN1_CODE1 \
--add-bos --add-eos
```

## resume

``` sh
python train.py \
    --model-config-path artifacts/base/model_config.json \
    --resume-from-checkpoint artifacts/base/checkpoint_step_008000.pt \
    --rope-type yarn \
    --rope-factor 4.0 \
    --rope-original-max-position-embeddings 4096 \
    --lr-multiplier 3.0 \
    --lr-multiplier 1.0 \
    --lr-multiplier 1.0 \
    --compression-ratio 3 \
    --compression-ratio 3 \
    --dataset-template 'SOURCES_JA8_EN1_CODE1_10' \
    --seq-len 16384 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --learning-rate 4.0e-4 \
    --save-every 4000 \
    --max-steps 16000 \
    --log-every 2000 \
    --validation-every 1000 \
    --validation-max-batches 20 \
    --validation-split-ratio 0.05 \
    --shuffle-buffer-size=16384 \
    --output-dir "artifacts/hnet_2stage_200m_j8_e1_c1_10"
```

## sft
packingされるので事前にstepsをカウント
``` sh
python estimate_sft_epoch_steps.py \
    --context-len 512 \
    --batch-size 2 \
    --grad-accum-steps 8
```

``` sh
python -m hnet.sft.train \
    --model-config-path artifacts/hnet_2stage_200m/model_config.json \
    --pretrained-model-path artifacts/hnet_2stage_200m/checkpoint_step_8000.pt \
    --output-dir artifacts/hnet_2stage_200m_sft \
    --seq-len 512 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --max-steps 1000
```

``` sh
python generate_sft.py \
    --model-path /path/to/sft_final_model.pt \
    --config-path /path/to/model_config.json
```

## eval

open_ai互換のサーバーを動かす
``` sh
python scripts/hnet_openai_server.py \
    --model-path /path/to/checkpoint_step_xxxxxx.pt \
    --config-path /path/to/model_config.json \
    --model-name hnet-local \
    --host 0.0.0.0 \
    --port 8000 \
    --max-tokens 256 \
    --temperature 0.0 \
    --top-p 1.0 > /content/hnet_server.log 2>&1 &

## check
tail -n 100 /content/hnet_server.log
curl -I -X GET http://127.0.0.1:8000/health
```

その後はllm-jp-evalを動作

``` sh
%cd /content
!git clone https://github.com/llm-jp/llm-jp-eval.git

%cd /content/llm-jp-eval
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv sync

## データセットの準備(初回のみ)
%cd /content/hnet/llm-jp-eval
!uv run scripts/preprocess_dataset.py -d all
```

gsm8k_jaについては、以下

``` sh
python eval_gsm8k_ja_openai.py \
  --base-url http://localhost:8000/v1 \
  --model hnet-local \
  --out results/gsm8k_ja.jsonl \
  --concurrency 4 \
  --max-tokens 512
```

---

# H-Net Original


<table width="100%">
  <tr>
    <td><img src="assets/english.gif" alt="English" width="100%"></td>
    <td><img src="assets/code.gif" alt="Code" width="100%"></td>
  </tr>
  <tr>
    <td><img src="assets/chinese.gif" alt="Chinese" width="100%"></td>
    <td><img src="assets/korean.gif" alt="Korean" width="100%"></td>
  </tr>
</table>

> **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling**\
> Sukjun Hwang, Brandon Wang, Albert Gu\
> Paper: https://arxiv.org/abs/2507.07955

## About
![H-Net](assets/arch.png "H-Net Architecture")

This repository contains code of the H-Net architecture. Most of the code lies in `hnet/`, which has the following structure:

```
configs/
hnet/
├── models/            # Directory for H-Net
|   ├── config_hnet.py     (defines the config for the H-Net)
|   ├── hnet.py            (h-net as a (B, L, D) -> (B, L, D) sequence model)
│   └── mixer_seq.py       (wrapper to turn h-net into a language model)
└── modules/           # Directory of model components
    ├── dc.py              (modeling code for the dynamic chunking mechanism)
    └── isotropic.py       (code for isotropic, i.e. non-hierarchical components)
generate.py        # Script for inference/generation
```

## Installation

### Requirements:
- PyTorch >= 2.5.1

Clone the repository and install package.
``` sh
git clone https://github.com/goombalab/hnet
cd hnet
pip install -e .
```


We strongly recommend building **mamba_ssm** package from [**the latest source**](https://github.com/state-spaces/mamba) as follows:
``` sh
git clone https://github.com/state-spaces/mamba
cd mamba
pip install .
```

## Pretrained Models

Pretrained models are uploaded to
[Hugging Face](https://huggingface.co/cartesia-ai): `hnet_1stage_L`, `hnet_2stage_L`,
`hnet_1stage_XL`, `hnet_2stage_XL`.
We trained our models on the 100B-Token subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). <em>Large</em> and <em>XL</em> are compute-matched to GPT-3 <em>Large</em> and <em>XL</em>, respectively.

We also provide model weights for Chinese and Code, each trained using the 46B-Token subset of [FineWeb-Edu Chinese V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) and [Pile Github](https://huggingface.co/datasets/EleutherAI/pile): `hnet_2stage_XL_chinese`, `hnet_2stage_XL_code`.

You can find specifics of these models at [configs](configs), and more details from the paper.


## Text Generation

We provide [generate.py](generate.py) for text generation that you can use with the pretrained checkpoints.

### Examples
``` sh
python generate.py --model-path [MODEL_CKPT] --config-path [CONFIG]
python generate.py --model-path hnet_2stage_XL.pt --config-path configs/hnet_2stage_XL.json --max-tokens 1024 --temperature 1.0 --top-p 1.0
```


## Citation

If you use this codebase, or otherwise find our work valuable, please cite H-Net:

```
@article{hnet,
  title={Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
  author={Hwang, Sukjun and Wang, Brandon and Gu, Albert},
  journal={arXiv preprint arXiv:2507.07955},
  year={2025}
}
```
