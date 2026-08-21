# 言語境界family評価 Phase M2 dense trajectoryレポート

作成日: 2026-08-21

## 1. 実行概要

- 対象: K1G1、K3T1
- seed: 42
- 学習: 220 step、同一packed train/validation data
- probe: 24文、6 family、各4文
- 観測: 10 stepごと、計22時点、native chunking
- checkpoint: step 55、110、165、220
- byte boundary constraint: `utf8-hard`
- K1G1 outer compression target: 2.5
- K3T1 outer compression target: 3.0
- K1G1 run commit: `816096c`（後処理は`9df2bd5`で復旧）
- K3T1 run commit: `9df2bd5`

両runともtraining metrics 220行、checkpoint 4個、chunk JSON 22個、dense raw JSON 22個、
集計JSONを確認した。作業領域とDrive archiveはそれぞれ82ファイルで、相対pathとsizeが一致し、
training metrics、summary、raw JSONのSHA-256も一致した。

本評価は学習中の実際のnative分割を追う。K1G1とK3T1ではouter compression targetが異なるため、
coverageの絶対値だけを同一境界予算の比較とは解釈しない。precision、fracture、category差、時間変化を
併せて判断する。またseed 42だけの結果であり、architecture固有の不変な性質とは断定しない。

## 2. Stage 1のdense trajectory

main networkへ渡る最終chunk境界であるstage 1を主比較とする。lateはstep 210/220の平均である。

| model | time precision | time coverage | unexplained occupancy | fracture occupancy | late precision | late coverage | late unexplained | late fracture | transition rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K1G1 | **0.598** | **0.675** | 0.553 | **0.098** | **0.636** | **0.636** | 0.500 | **0.083** | 0.341 |
| K3T1 | 0.508 | 0.417 | **0.390** | 0.311 | 0.233 | 0.152 | **0.417** | 0.396 | 0.375 |

K3T1は説明困難境界を一つでも含むrecordの滞在率では低いが、保護語彙内のfracture滞在率はK1G1の
約3.2倍である。単にunexplained occupancyだけを見るとK3T1を過大評価するため、lexeme fractureを
分離したことが重要だった。K1G1はprecision、coverage、late window、fractureで優位であり、
220 stepまで維持されるanchorとして強い。

stage 0は両モデルともstage 1より説明困難な細分割が多かった。K1G1/K3T1の順にtime precisionは
0.444/0.385、fracture occupancyは0.472/0.591だった。stage 0だけでモデルを選ぶのではなく、
階層化後にmain networkへ渡るstage 1を主に見るべきである。

## 3. Category別の相補性

| category | model | time precision | time coverage | fracture occupancy | late precision | late coverage | late fracture |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compound | K1G1 | 0.340 | 0.290 | 0.557 | 0.000 | 0.000 | 0.500 |
| compound | K3T1 | **0.802** | **0.636** | **0.284** | **0.800** | **0.500** | **0.250** |
| context control | K1G1 | **0.620** | **0.909** | **0.011** | **0.571** | **1.000** | **0.000** |
| context control | K3T1 | 0.315 | 0.341 | 0.534 | 0.000 | 0.000 | 0.875 |
| inflection | K1G1 | **0.641** | **0.777** | **0.006** | **0.708** | **0.810** | **0.000** |
| inflection | K3T1 | 0.459 | 0.348 | 0.261 | 0.105 | 0.048 | 0.313 |

K3T1の複合語優位は時間平均だけでなくlate windowにも残り、K1G1との相補性は再現した。一方、
K3T1は活用と同一surfaceの文脈controlで終盤に崩れた。K1G1は活用・controlに強いが、複合語では
内部landmarkを選ばず、`自然言語処理`全体を一chunkにする側へ移った。

family別にも同じ傾向がある。K3T1の`natural-language-compounds`はtime precision/coverageが
0.765/0.636、lateが0.800/0.500で、K1G1の0.325/0.290、late 0/0を上回った。対してK1G1は
`change-suru`、`split-suru`、`write`、`split-context-control`でlate coverage 0.80以上を維持した。

## 4. 学習中の変化

stage 1のfamily macroを抜粋する。precisionが未定義の「境界なし」familyは0として平均した。

| step | K1 precision | K1 coverage | K1 integrity | K1 landmark | K3 precision | K3 coverage | K3 integrity | K3 landmark |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 30 | 0.451 | 0.783 | 0.958 | 0.667 | 0.694 | 0.550 | 0.750 | 0.708 |
| 50 | 0.480 | 0.756 | 0.833 | 0.792 | **0.889** | **0.782** | 0.792 | **0.875** |
| 70 | 0.756 | 0.740 | **0.917** | **0.958** | 0.807 | 0.740 | 0.667 | 0.833 |
| 90 | 0.810 | **0.856** | **0.917** | **0.833** | **0.913** | 0.674 | 0.875 | 0.750 |
| 110 | **0.603** | **0.889** | **0.917** | **0.833** | 0.487 | 0.465 | 0.500 | 0.542 |
| 150 | 0.547 | **0.850** | **0.917** | **0.833** | **0.667** | 0.319 | 0.667 | 0.333 |
| 190 | **0.629** | **0.706** | **0.917** | **0.833** | 0.300 | 0.117 | 0.583 | 0.042 |
| 220 | **0.598** | **0.706** | **0.917** | **0.833** | 0.300 | 0.117 | 0.583 | 0.042 |

K3T1はstep 40--90で高いprecision/coverageを示したが、step 100前後から低下し、step 180以降は
landmark coverageが0.042まで落ちた。したがってstep 55だけならK3T1を過大評価し、step 220だけなら
初期・中期の強みを見落とす。今回の10-step観測により、「分割が揺れる」だけでなく、どのcategoryの
許容領域へ滞在し、いつ別の領域へ移ったかを区別できた。

K1G1も固定ではない。step 30のpathological record rateは0.375だったが、step 190--220では0に低下し、
landmark coverageはstep 70以降ほぼ0.833を維持した。変化量そのものではなく、終盤に説明可能領域へ
滞在しているかを見るという評価方針が支持される。

## 5. 分割例

`<0xFE>`はBOS表示なので、言語的評価から除外する。

### K1G1

- step 100: `...|ごとに|分割|す|る。`
- step 220: `...|段落|ごとに|分割|す|る。`
- step 220: `...|自動で分割|し|ている。`
- step 220: `...|聞|いて笑|っている。`
- step 220: `自然言語処理|の基礎|を学ぶ。`

K1G1はサ変語幹と活用語尾の境界を複数文脈で維持した。複合語は終盤に全体をまとめるため、
複合語内部の再利用可能landmarkという今回の注釈に対するcoverageは低い。

### K3T1

- step 50: `...|ごとに分割|する。`
- step 50: `...|笑|っている。`
- step 50: `自然|言語処理|の基礎|を学ぶ|。`
- step 220: `...|落ごとに分|割す|る。`
- step 220: `...|自動で分|割し|ている。`
- step 220: `自然|言語処理の基|礎を|学ぶ。`

K3T1のstep 50は説明可能な活用・複合語境界が多い。しかし終盤には`分|割す`相当の語彙内切断や
`基|礎`の切断が現れ、活用・controlのfamily指標低下と整合する。一方、`自然|言語...`の複合語境界は
残っており、compound categoryの相補性も実例で確認できる。

## 6. 判断

1. **長期学習の第一anchorはK1G1を維持する。** seed 42 dense trajectoryでは、K1G1が複合語以外の
   family、late precision/coverage、lexeme integrityで明確に優位だった。
2. **K3T1を単独の同格anchorには昇格させない。** step 40--90の良さは持続せず、終盤に活用・controlの
   coverageとintegrityが低下した。ただしseed 42だけなので、architecture一般の失敗とは断定しない。
3. **category相補性の条件は満たした。** K3T1のcompound優位は時間平均・late・実例で再現したため、
   Phase M3の小規模なordered/reverse混合比較へ進む根拠はある。
4. 混合比較ではK3系成分を全面採用せず、K1G1成分との同一成分数・配置順だけを変える2構成をまず
   seed 42、55 stepで比較する。compound改善と同時に、K1G1の活用・controlを壊さないことをgateにする。
5. 混合55 stepでも10-step dense coreを残す。K3T1がstep 55付近だけ良く見える現象があるため、
   最終点だけで選ばない。差が小さい、または活用・controlが悪化するなら混合探索を打ち切る。

## 7. Artifact

- K1G1 archive: Drive `runs/dense_family_v1/r6_dense_family_v1_k1g1_s42_step220_816096c/`
- K3T1 archive: Drive `runs/dense_family_v1/r6_dense_family_v1_k3t1_s42_step220_9df2bd5/`
- combined comparison: Drive `reports/linguistic_boundary_selection/dense_family_v1/dense_comparison.json`
- 各archive内: `training_metrics.csv`、`validation_chunks/`、`dense_raw/`、`dense_summary/`、4 checkpoint

