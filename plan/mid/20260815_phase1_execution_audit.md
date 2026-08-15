# Phase 1実行監査メモ

作成日: 2026-08-15

## 再利用する固定資産

- train: `/content/drive/MyDrive/hnet_agent/datasets/screening/ja8_en1_code1_0p25b_ctx2k`
  - 269,759,141 bytes
  - tree SHA256: `bfe38d88f92ae40cbc9eac4d24e2f743d424e5593eea50d533b9a6aac070e25c`
- validation: `/content/drive/MyDrive/hnet_agent/datasets/validation/ja8_en1_code1_12m_ctx2k`
  - 11,185,274 bytes
  - tree SHA256: `6bc22349d0bc842c0b8fee209fa97b257d5b711eeddf26dbb1d328c014efaf59`
- A100 40GB、PyTorch 2.10.0+cu130、Triton 3.6.0。
- 旧runは`hnet_agent_kda_diff`に残し、今回の新規成果物は
  `/content/drive/MyDrive/hnet_agent_200m_main`だけへ保存する。

## 既存証拠

- T26: seed 42/43、220 steps、4 checkpointsあり。
- K1T1: seed 42/43、220 steps、4 checkpointsあり。
- M3T1: seed 42、220 steps、4 checkpointsあり。
- K1G1: seed 42の55-step calibrationのみ。`comp=3.0/2.5`, `ratio_weight=0.08`。

## 新規実行matrix

- T26: seed 44
- K1T1: seed 44
- M3T1: seed 43/44
- K1G1: seed 42/43/44

各runは220 steps、checkpoint 55/110/165/220、UTF-8 hard制約、同一raw document列。
既存runを含めて各候補3 seedを揃える。

## 環境上の発見

Colab PrepareセルはKDA用`fla-core`を導入しないため追加installが必要だった。
またPyTorch 2.10 CUDA 13 wheelのNVRTC libraryは
`/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib`にあるが、Colab既定の
`LD_LIBRARY_PATH=/usr/lib64-nvidia`から見えない。KDA/Gated MLA実行時は前者を
`LD_LIBRARY_PATH`へprependする。

最初のK1G1 attemptはこのpath不足で学習前に失敗した。診断logは削除せず
`runs/phase1_calibration/k1g1_failures/`へ保存した。
