# 112文統合category・family評価／dense再実験レポート

作成日: 2026-08-21

## 1. 修正した実験範囲

category評価を11 category x 8文の88文probeと定義し、24文・6 familyのfamily probeと統合した。
family probe内の`inflection`、`compound`、`context_control`を88文category評価の代用にはしていない。

実験は二段階で行った。

1. 既存のT26、K1G1、K3G1、K3T1について、3 seed、step 55/110/165/220、
   low/central/high/nativeの112文結果を統合集計した。
2. 統合Paretoに残ったK1G1、K3G1、K3T1をseed 42で220 step再学習し、family 24文を
   10 stepごとにnative観測した。さらに4 checkpointすべてで112文をlow/central/high/native評価した。

88文rawとfamily rawは48組すべてでmodel path、config、seed、checkpoint、compression profile、
`utf8-hard`設定が一致したため、同じ決定的推論を再実行せず112 recordsへ統合した。K3G1 denseだけを
新規実行し、既に同条件で完了していたK1G1/K3T1 denseは再利用した。

## 2. 4候補の112文統合スクリーニング

step 220、central、stage 1の3 seed平均である。categoryは88文のmicro、familyは6 familyのmacro。
family precisionが未定義の「境界なし」familyは0として平均した。

| model | category precision | category coverage | category fracture | family precision | family coverage | integrity | landmark coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K1G1 | 0.451 | **0.271** | 0.255 | **0.708** | **0.506** | 0.861 | **0.611** |
| K3G1 | **0.490** | 0.261 | **0.218** | 0.339 | 0.240 | 0.833 | 0.306 |
| K3T1 | 0.453 | 0.254 | 0.265 | 0.619 | 0.380 | **0.903** | 0.458 |
| T26 | 0.439 | 0.239 | 0.257 | 0.448 | 0.332 | 0.806 | 0.417 |

厳密Pareto frontはK1G1、K3G1、K3T1となった。T26は表示した全軸でK1G1に支配された。
K3G1はcategory precision/fracture、K1G1はcategory coverageとfamily、K3T1はfamily integrityに
固有の強みがあり、family結果だけでK3G1を除外した以前の判断は撤回した。

## 3. 10-step dense core

stage 1、native、family 24文のstep 10--220時間集約である。lateはstep 210/220。

| model | time precision | time coverage | unexplained occupancy | fracture occupancy | late precision | late coverage | late fracture | transition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K1G1 | **0.598** | **0.675** | 0.553 | **0.098** | **0.636** | **0.636** | **0.083** | 0.341 |
| K3G1 | 0.478 | 0.492 | 0.547 | 0.335 | 0.412 | 0.424 | 0.208 | 0.349 |
| K3T1 | 0.508 | 0.417 | **0.390** | 0.311 | 0.233 | 0.152 | 0.396 | 0.375 |

K1G1はtime/late precision・coverageとfractureで最良だった。K3G1はK3T1よりlateで良く、
K3T1は説明困難境界を含むrecordの滞在率こそ低いが、終盤のlexeme fractureが大きい。
変化率だけではなく、説明可能領域とfracture領域のどちらに滞在したかを分離する必要性が再確認された。

## 4. Dense checkpointの112文評価

### 4.1 Step 220 central

| model | category precision | category coverage | category fracture | family precision | family coverage | integrity | landmark coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K1G1 | 0.541 | **0.342** | **0.149** | **0.889** | **0.562** | 0.917 | **0.708** |
| K3G1 | **0.602** | 0.278 | 0.241 | 0.042 | 0.021 | 0.875 | 0.042 |
| K3T1 | 0.492 | 0.252 | 0.317 | 0.222 | 0.088 | **0.958** | 0.125 |

K3G1のcategory precisionは最良だが、固定境界予算ではfamily landmarkをほとんど選ばなかった。
K1G1はcategory coverage/fractureとfamily全体が良く、単一の強いcategoryに依存しない。
K3T1の高integrityは、境界を置かないことで保護語彙を壊さない寄与を含むため、coverageと必ず併記する。

### 4.2 Step 220 native

| model | category precision | category coverage | category fracture | family precision | family coverage | integrity | landmark coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K1G1 | 0.447 | **0.359** | **0.213** | **0.598** | **0.706** | **0.917** | **0.833** |
| K3G1 | **0.456** | 0.355 | 0.214 | 0.445 | 0.428 | 0.792 | 0.500 |
| K3T1 | 0.316 | 0.158 | 0.470 | 0.300 | 0.117 | 0.583 | 0.042 |

nativeのcategoryではK1G1とK3G1が近いが、familyではK1G1が明確に上回った。K3T1は
category/familyの双方でstep 220までに低下した。

### 4.3 境界予算への頑健性

step 220のprecision/coverageを示す。

| model | low category | low family | central category | central family | high category | high family |
| --- | --- | --- | --- | --- | --- | --- |
| K1G1 | .452/.423 | **.900/.719** | .541/.342 | **.889/.562** | .547/.274 | **.833/.372** |
| K3G1 | **.509/.350** | .067/.042 | **.602/.278** | .042/.021 | **.608/.205** | .167/.021 |
| K3T1 | .419/.385 | .229/.163 | .492/.252 | .222/.088 | .345/.124 | .167/.042 |

K1G1だけがlow/central/highの全profileで高いfamily precision/coverageを維持した。K3G1は
category precisionが一貫して高い一方、familyの説明可能landmarkは強制profileで消える。

## 5. 学習中の変化

centralのstep 55から220への変化を示す（category precision/coverage、family precision/coverage）。

| model | step 55 | step 110 | step 165 | step 220 |
| --- | --- | --- | --- | --- |
| K1G1 | .402/.226, .944/.550 | .503/.329, .906/.754 | .511/.299, .889/.529 | .541/.342, .889/.562 |
| K3G1 | .510/.316, .278/.075 | .556/.316, .233/.075 | .604/.274, .042/.021 | .602/.278, .042/.021 |
| K3T1 | .492/.256, .967/.761 | .436/.248, .310/.158 | .517/.261, .278/.108 | .492/.252, .222/.088 |

K3T1はstep 55だけならfamily最良に見えるが、step 110で急減した。K3G1はcategoryが改善する一方、
familyはstep 165までにほぼ境界なしへ移った。K1G1はcategoryが改善し、familyも全期間で高水準だった。
55/220の一点比較ではなくdense trajectoryを見る判断は有効だった。

## 6. Category別の相補性

step 220 centralでK3G1は文節precision/coverage 0.636/0.609、K1G1は0.065/0.087だった。
一方K1G1は助動詞0.778/0.438、助詞0.769/0.476、structured 0.762/0.356など、複数categoryで
高いprecision/coverageを示した。K3G1は文節と全体category precision、K1G1はfamilyと広いcategoryに
強く、相補性は残る。ただしK3G1単独を長期anchorにする根拠ではなく、配置順だけを変える小規模な
混合比較を行う根拠と解釈する。

固有名詞は全候補で弱く、K1G1/K3G1/K3T1のprecision/coverageはそれぞれ
0.286/0.125、0.444/0.250、0.333/0.062だった。長期学習候補でも監視を継続する必要がある。

## 7. 判断

1. **第一anchorはK1G1。** familyの再利用可能境界、native、固定profile、late windowのすべてで
   最も一貫していた。
2. **第二の単独比較候補はK3G1。** familyでは弱いが、88文category precision、文節、native categoryで
   K1G1と異なる強みがあり、K3T1よりlate familyも良かった。
3. **K3T1は長期単独anchorから外す。** step 55の良さが持続せず、step 220 nativeのfractureと
   coverage低下が大きい。ただし初期挙動の対照artifactは保持する。
4. Phase M3の条件であるcategory相補性はK1G1/K3G1間で満たされた。次は両者の同一成分数を保ち、
   配置順だけを変えたordered/reverse 2構成をseed 42、55 step、10-step dense付きで比較する。
5. 混合でK1G1のfamily・助動詞・助詞・structuredを壊す場合、または文節改善が再現しない場合は
   混合探索を打ち切り、K1G1を長期本線とする。

## 8. Artifact

- 4候補112文raw: Drive `evals/linguistic_boundary_combined112_v1/`
- 4候補統合集計: Drive `reports/linguistic_boundary_selection/combined112_v1/`
- K3G1 dense archive: Drive `runs/dense_family_v1/r6_dense_family_v1_k3g1_s42_step220_0e89e00/`
- 3構成dense full raw: Drive `evals/dense_full112_v1/`
- 3構成dense full集計: Drive `reports/linguistic_boundary_selection/dense_full112_v1/`
- 実行manifest: Drive `manifests/dense_full112_v1_status.json`
- analysis: `combined112_analysis.json`、`dense_full112_analysis.json`

## 9. ここまでに分かったこと

### 9.1 構成ごとの強み

K1G1はfamily probe、landmark、助動詞、助詞、複合語、structured categoryで強く、
low/central/highの境界予算を変えても比較的安定していた。fractureが少なく、学習中の
family指標も他の構成より維持された。K3G1はcategory precisionと文節で強い一方、学習後半の
強制profileでfamily landmarkが選ばれにくくなった。native categoryではK1G1と近いため、
これは単に境界数が少ないことでは説明できない。

K3T1はstep 55のfamilyが強いが、step 110以後に低下し、強みが維持されなかった。T26は
今回の境界評価軸でK1G1に対する独立した優位点を示さなかった。そのため、最初の混合実験は
K1G1とK3G1に絞る。

### 9.2 強みの発生理由についての解釈

K1G1とK3G1はKDAとGated MLAという同じ種類のmain-network layerを使うが、配置頻度が異なる。
K1G1は両者を交互に配置し、K3G1はKDAを3層続けてからGated MLAを配置する。観測結果と
整合する仮説は次の通りである。

- K1G1の頻繁なKDA/Gated MLAの切り替えは、異なる語形や文脈で再利用できる活用・助詞・
  複合語境界を、outer-stageのboundary predictorが安定して上位に順位付けする学習と関連している。
- K3G1の連続KDA区間は文節のようなやや長いまとまりに適応する学習と関連するが、familyで
  共通する形態的landmarkを固定境界予算の上位に維持しにくい。
- K3G1でcategoryが改善する一方でfamilyが低下し、K3T1の初期優位も消えたことから、強みは
  固定された層の能力だけでなく、main networkとencoder/decoder・boundary predictorの共適応によって
  形成される。

ただし、以上は結果と整合する仮説であり、層ごとの表現や勾配を直接測定した因果説明ではない。
境界はouter stageが出力するため、特定のmain-network layerが直接文節や助詞を検出したとは結論しない。

### 9.3 混合構成について分かった範囲

K1G1のfamily・複数category・低fractureと、K3G1の文節・category precisionに相補性があるため、
K1G1を基盤にK3G1型の連続KDA区間を部分的に入れる方向は次の検証対象になる。一方、
現在の観測だけでは、前半と後半のどちらに連続KDA区間を置くべきかは分からない。

次の実験では26層、KDA 16層、Gated MLA 10層を共通にし、`K1G1 x 7 -> K3G1 x 3`と
`K3G1 x 3 -> K1G1 x 7`の配置順だけを反転する。両者の差は層位置の影響、両者に共通する
親構成からの変化はK/G比率を中間化した影響の手がかりとする。実験前に分かったのはこの
検証可能な構成仮説までであり、最適な配置は未確定である。

混合の成功は単一のaggregate scoreではなく、K1G1のfamily coverage、landmark、助動詞、助詞、
structured、fractureを維持しながら、K3G1の文節precision/coverageに近づくかというPareto改善で
判定する。
