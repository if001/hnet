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

2026-08-15 21:40 JSTにColab A100の逐次queueとして開始した。queue完了後は
`scripts/summarize_phase1_selection.py`で既存runと新規runを統合し、K1G1の3 seedへ
Learned、Fixed、Random、Morph、shifted-left/rightの共通probeを実行する。

## Phase 2対抗候補の事前選定規則

Phase 2結果を見る前に次の規則を固定する。

1. 各候補3 seedの最終BPB平均とpopulation SDを集計する。
2. 実用上の同等幅を`max(0.001 BPB, T26/K1T1/K1G1のpopulation SD)`とする。
3. K1G1がK1T1を同等幅より大きく上回る場合だけK1G1を対抗候補とする。
4. それ以外は、既存2 seedでBPBと分散が安定しているK1T1を対抗候補とする。
5. M3T1は計画どおり、明確な長文効率またはagent proxy優位がないためPhase 2へ
   自動昇格させない。

partial集約時点ではK1T1が2 seed平均1.6768567、range 0.0002668、T26が
2 seed平均1.6789823、range 0.0051653、K1G1 seed42が1.6781823だった。
K1G1 seed43/44完了前のため、この値だけでは候補を確定しない。

## 環境上の発見

Colab PrepareセルはKDA用`fla-core`を導入しないため追加installが必要だった。
またPyTorch 2.10 CUDA 13 wheelのNVRTC libraryは
`/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib`にあるが、Colab既定の
`LD_LIBRARY_PATH=/usr/lib64-nvidia`から見えない。KDA/Gated MLA実行時は前者を
`LD_LIBRARY_PATH`へprependする。

最初のK1G1 attemptはこのpath不足で学習前に失敗した。診断logは削除せず
`runs/phase1_calibration/k1g1_failures/`へ保存した。
