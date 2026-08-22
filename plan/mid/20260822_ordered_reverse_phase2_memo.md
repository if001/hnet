# K1G1/K3G1 ordered-reverse混合 Phase 2 暫定集計

作成日: 2026-08-22

## 実行状態

- K1-first / K3-first を seed 42、step 220まで学習した。
- step 55からの再開ではoptimizer、data position、LR horizon=220を維持した。
- checkpoint 55/110/165/220、family dense 10 step間隔（22時点）、各checkpointの112文
  low/central/high/native評価を確認した。
- raw artifactと集計JSONはDriveの `hnet_agent_200m_main` 以下に保存した。

## central profileの主要結果

| model | step | category P/C/F | family P/C | integrity | landmark C | 文節 P/C |
| --- | ---: | --- | --- | ---: | ---: | --- |
| K1G1 | 220 | .541/.342/.149 | .889/.562 | .917 | .708 | .065/.087 |
| K3G1 | 220 | .602/.278/.241 | .042/.021 | .875 | .042 | .636/.609 |
| K1-first | 220 | .437/.282/.305 | .625/.396 | .958 | .375 | .219/.304 |
| K3-first | 220 | .392/.239/.329 | .611/.375 | .958 | .458 | .161/.217 |

混合構成はstep 55では親の相補的特徴を一部両立したが、その優位はstep 110以降に弱くなった。
step 220では、K1G1より文節を改善している一方、category precision、family precision/coverage、
fractureでK1G1に退行し、K3G1の文節にも届かない。このため、seed 42単独では長時間候補とする
Pareto改善は確認できない。

## 22時点dense評価

| model | time P/C | fracture occupancy | late P/C | late fracture |
| --- | --- | ---: | --- | ---: |
| K1G1 | .598/.675 | .098 | .636/.636 | .083 |
| K3G1 | .478/.492 | .335 | .412/.424 | .208 |
| K1-first | .559/.428 | .426 | .500/.424 | .542 |
| K3-first | .627/.445 | .326 | .622/.348 | .292 |

同じK=16/G=10でもdense軌跡ではK3-firstがK1-firstよりprecisionとfractureで良い。一方、
step 220単点ではK1-firstがcategory/family/文節の多くでK3-firstを上回る。したがって、配置順差は
観測されたが、優位方向は評価時点・評価軸にまたがって一貫しておらず、seed 42だけで層位置効果とは
結論できない。

## Phase 3判定

計画の「意味のあるPareto改善または配置順差が残った場合」に照らし、後者を満たすものとして
K1-firstとK3-firstの両方をseed 43/44へ進める。片方だけでは順序効果とseed効果を分離できない。

Phase 3では次を判定する。

1. denseのtime-averaged precisionとfractureでK3-first優位が再現するか。
2. step 220のcategory/family/文節でK1-first優位が再現するか。
3. seed間で方向が反転する場合は層位置効果とせず、混合探索を停止する。
4. 順序差が再現しても親K1G1に対するPareto改善がなければ、長期本線はK1G1のままとする。

