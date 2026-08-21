# 言語境界family評価 Phase M1・疎trajectoryレポート

作成日: 2026-08-21

## 1. 実行概要

- 対象: T26、K1G1、K3G1、K3T1
- seed: 42、43、44
- checkpoint: step 55、110、165、220
- probe: 24文、6 family、各4文
- 条件: stage 1、`central`を主比較。native/low/highもrawへ保存
- byte boundary constraint: `utf8-hard`
- 実行数: 48 checkpoint、各24 records。全件成功
- code commit: `0337682`（評価時本体は`2d000f7`、parser修正のみ追補）

活用4 family、自然言語複合語1 family、同じ`分割する`を異なる文脈へ置くcontrol 1 familyを
評価した。familyごとに共通landmarkを注釈し、coverageとconsistencyを分離した。

## 2. Step 220

3 seed、6 familyのmacro平均を示す。境界が一つも選ばれずprecisionが未定義のfamilyは0として
平均し、「何も選ばない一貫性」を高く評価しない。

| model | family precision | family coverage | lexeme integrity | landmark coverage | landmark consistency | context consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T26 | 0.448 | 0.332 | 0.806 | 0.417 | 0.889 | 0.917 |
| K1G1 | **0.708** | **0.506** | 0.861 | **0.611** | 0.861 | 0.750 |
| K3G1 | 0.339 | 0.240 | 0.833 | 0.306 | **0.917** | **1.000** |
| K3T1 | 0.619 | 0.380 | **0.903** | 0.458 | 0.847 | 0.833 |

K1G1はfamily precision、coverage、landmark coverageが最大だった。K3T1はlexeme integrityが
最大で、説明困難境界を避ける側が強い。K3G1のconsistencyは高いがlandmark coverageが低く、
一貫して説明可能landmarkを選ぶというより、一貫して境界を置かないfamilyの寄与を含む。

## 3. 疎trajectory

step 55/110/165/220の全観測と、late window（165/220）を3 seedで平均した。

| model | time precision | time coverage | unexplained occupancy | fracture occupancy | late precision | late coverage | late unexplained | late fracture | transition rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T26 | 0.705 | 0.369 | 0.191 | 0.163 | 0.604 | 0.313 | 0.243 | 0.194 | 0.190 |
| K1G1 | 0.796 | **0.518** | 0.170 | 0.122 | 0.767 | **0.490** | 0.194 | 0.132 | 0.255 |
| K3G1 | 0.488 | 0.275 | 0.330 | 0.132 | 0.388 | 0.232 | 0.389 | 0.160 | 0.199 |
| K3T1 | **0.834** | 0.475 | **0.125** | **0.066** | **0.782** | 0.404 | **0.153** | **0.090** | 0.389 |

K3T1はprecisionと有害occupancyで最良だが、pattern transitionも最大である。K1G1はcoverageが
最良で、K3T1よりtransitionが低い。4 checkpointだけでは、K3T1の高いtransitionが許容分割間の
有益な移動か、不安定性かを判定できない。

family macroのstep推移は次のとおりだった（precision / coverage / integrity / landmark coverage）。

| model | step 55 | step 110 | step 165 | step 220 |
| --- | --- | --- | --- | --- |
| T26 | .880/.527/.931/.611 | .539/.384/.806/.458 | .487/.361/.806/.444 | .448/.332/.806/.417 |
| K1G1 | .816/.577/.903/.653 | .749/.552/.875/.625 | .770/.551/.875/.639 | .708/.506/.861/.611 |
| K3G1 | .574/.411/.917/.542 | .339/.254/.875/.319 | .326/.244/.847/.306 | .339/.240/.833/.306 |
| K3T1 | .964/.700/.958/.792 | .576/.442/.958/.472 | .662/.479/.917/.583 | .619/.380/.903/.458 |

K1G1は他候補よりstep間の低下が小さい。K3T1は非単調に変化し、最終1点だけでも初期1点だけでも
特徴を代表できない。

## 4. Seed差

step 220のlandmark coverageをseed 42/43/44の順に示す。

| model | seed 42 | seed 43 | seed 44 |
| --- | ---: | ---: | ---: |
| T26 | 0.667 | 0.208 | 0.375 |
| K1G1 | 0.833 | 0.542 | 0.458 |
| K3G1 | 0.042 | 0.833 | 0.042 |
| K3T1 | 0.167 | 0.375 | 0.833 |

全候補にseed差があり、特にK3G1とK3T1で大きい。K1G1も完全には安定していないが、3 seedすべてで
0.45以上を維持した唯一の候補だった。このため、step 220平均だけで1構成へ確定しない。

seed 42、step 220の`分割する／した／している／される`では、T26とK1G1は4語形すべてで
`分割|...`を選んだ。K3G1とK3T1は4語形すべてでそのlandmarkを選ばなかった。ただしK3T1はseed 44で
全体landmark coverageが高く、これはarchitecture固有の固定的特徴ではない。

## 5. 暫定判断

1. **K1G1を第一anchorとする。** family coverageとlandmark coverageが最大で、seed最小値も
   他候補より高い。
2. **K3T1を第二anchorとして残す。** lexeme integrity、precision、低いunexplained/fracture
   occupancyが強く、K1G1と相補的である。
3. **K3G1は現段階でdense対象にしない。** 既存88文ではcoverageの強みがあったが、今回の
   再利用可能landmarkでは平均coverageが低く、seed 42/44でほぼ選択しなかった。
4. **T26はbaselineとして保持するがdense再学習対象にはしない。** control consistencyは高いが、
   family coverageとlate有害occupancyでK1G1/K3T1に劣る。

K1G1とK3T1について10 step間隔のdense coreを観測し、許容分割occupancy、fracture occupancy、
遷移先を確認する。その結果でも相補性が再現する場合だけ、ordered/reverseの混合構成へ進む。

## 6. Artifact

- raw: Drive `evals/linguistic_boundary_family_v1/`
- summary: Drive `reports/linguistic_boundary_selection/family_v1/`
- `linguistic_boundary_families.csv`
- `linguistic_boundary_family_landmarks.csv`
- `linguistic_boundary_trajectory.csv`
- `family_v1_analysis.json`
- `linguistic_boundary_blind_gallery.md`
