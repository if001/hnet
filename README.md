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
`--max-steps` を省略すると、設定したデータセットを使い切るまで学習します。

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
