# KDA量・配置位置／seed要因分離 screening 実行memo

更新日: 2026-08-23

## Phase 0

- 実装commit: `3ef7be1`
- Colab: A100-SXM4-40GB、PyTorch 2.10.0+cu130
- focused tests: 40 passed
- 対象6構成のmodel構築とparameter accounting: 成功

### Parameter accounting

| 構成 | K/G | total parameters | main-network parameters |
| --- | ---: | ---: | ---: |
| K1G1 | 13/13 | 217,033,864 | 181,249,160 |
| K14-front | 14/12 | 218,757,552 | 182,972,848 |
| K14-middle | 14/12 | 218,757,552 | 182,972,848 |
| K14-late | 14/12 | 218,757,552 | 182,972,848 |
| K15-split | 15/11 | 220,481,240 | 184,696,536 |
| K16-even | 16/10 | 222,204,928 | 186,420,224 |

K14の3配置はparameter数が完全に一致するため位置だけを比較できる。KDA量を増やす比較では、
K1G1比でK14は+0.79%、K15は+1.59%、K16は+2.38%のtotal parameter増加を伴う。

Drive artifact:
`/content/drive/MyDrive/hnet_agent_200m_main/manifests/kda_factorized_screening/phase0_parameter_accounting.json`

### 同一seed triple再現性audit

K1G1とK14-middleを同じ `(init=42, data=42, runtime=42)`、Phase 1と同じ
batch/sequence/gradient accumulation条件で各2回、1 optimizer step実行した。

- step 0 model hash: 両構成で反復間一致
- shuffle index hash: 両構成で反復間一致
- 先頭32 sample `(shard_id, chunk_offset)`: 両構成で反復間一致
- step 1 model: bitwise一致しない

| 構成 | changed fraction | mean absolute weight diff | max absolute diff | CE loss absolute diff |
| --- | ---: | ---: | ---: | ---: |
| K1G1 | 0.005314 | 9.08e-9 | 6.10e-5 | 1.73e-5 |
| K14-middle | 0.005087 | 1.07e-8 | 6.10e-5 | 1.09e-6 |

初期weightとdata orderは厳密に再現できている。一方、同一GPU/software/seed tripleでもCUDA演算後の
checkpointはbitwise一致せず、ごく小さい数値差が生じた。Phase 1以降ではこの幅を同一triple反復noiseの
実測下限として扱い、完全一致を要求しない。architecture差の判断は境界metricの実用差とrecord実例を
用い、1e-5程度のloss差を構成差として解釈しない。

Drive artifact:
`/content/drive/MyDrive/hnet_agent_200m_main/manifests/kda_factorized_screening/phase0_reproducibility_audit.json`

## Phase 1

K1G1 + 新規5構成を `(42,42,42)`、55 step、LR horizon 220で順次実行開始した。
各runはfamily denseをstep 10/20/30/40/50で取得し、step 55 checkpointとstep 0 stateをDriveへ保存する。

Status:
`/content/drive/MyDrive/hnet_agent_200m_main/manifests/kda_factorized_screening/phase1_training_status.json`
