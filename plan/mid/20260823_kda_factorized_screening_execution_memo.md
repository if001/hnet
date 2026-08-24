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

### Phase 1完了・artifact監査

6構成はすべてreturn code 0で終了した。各runについて次を確認した。

- step 55 checkpoint: 1個
- family dense chunk/raw: step 10/20/30/40/50の5時点
- training/validation metrics、dense summary、seed manifest
- architectureごとのstep 0 state
- 全構成で同一shuffle index hash

再起動後のColabでもfocused tests 40件が通過した。

Drive artifact:
`/content/drive/MyDrive/hnet_agent_200m_main/manifests/kda_factorized_screening/phase1_artifact_audit.json`

### Phase 1主要結果

step 55、central、stage 1の主要値を示す。dense P/C/Fは5時点のnative time-average
precision/coverage/fracture-record occupancyである。

| 構成 | category P/C/F | family P/C | integrity | landmark C | 文節 P/C | dense P/C/F |
| --- | --- | --- | ---: | ---: | --- | --- |
| K1G1 | .414/.235/.226 | .944/.686 | .917 | .750 | .111/.130 | .479/.485/.125 |
| K14-front | .417/.214/.133 | .708/.487 | .958 | .583 | .120/.130 | .446/.515/.125 |
| K14-middle | .431/.201/.110 | .772/.515 | .958 | .583 | .087/.087 | .503/.552/.133 |
| K14-late | .435/.214/.139 | .869/.651 | .958 | .750 | .080/.087 | .493/.515/.217 |
| K15-split | .468/.282/.234 | .750/.454 | .833 | .583 | .393/.478 | .519/.388/.125 |
| K16-even | .504/.274/.197 | .750/.537 | .958 | .667 | .345/.435 | .650/.461/.258 |

K15-splitとK16-evenはK1G1比で文節precision/coverageをそれぞれ
`+.282/+.348`、`+.234/+.304`改善した。category全体も両者で改善した。K16-evenは助動詞、
助詞、複合語にも強い一方、dense fracture occupancyが`.125 -> .258`へ悪化した。K15-splitは
dense fractureを`.125`に維持したが、dense coverageは低下した。

両構成ともcentral family precision/coverageを大きく失った。ただしprofile別では、K15-splitの
family P/C差はlow `+.022/-.014`、high `.000/+.150`、native `+.101/+.028`、K16-evenは
low `+.055/-.117`、high `-.042/+.078`、native `-.069/.000`であり、centralの退行が全profileへ
一貫していない。profile依存trade-offとseed依存性を分けるためPhase 2で確認する。

K14群では、K14-frontの文節precision `+.009`以外は文節改善がなく、1境界程度の差を昇格理由に
しない条件により除外する。K14-middleはlate dense P/C `.667/.848`、late fracture `.104`と良好だが、
文節改善という本計画の主条件を満たさない。K14-lateはfamily/landmarkを最も維持したが、文節とdense
fractureが退行した。

### Phase 2昇格

- anchor: K1G1
- 候補: K15-split、K16-even
- 条件: init-43、init-44、data-43、data-44を各構成で実施し、baselineはPhase 1を再利用する。
- data条件ではPhase 1で保存したarchitecture別step 0 stateをロードし、hash一致を必須とする。

集計JSON:
`/content/drive/MyDrive/hnet_agent_200m_main/reports/kda_factorized_screening/phase1_new6/phase1_analysis.json`
