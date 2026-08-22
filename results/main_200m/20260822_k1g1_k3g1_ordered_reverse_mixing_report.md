# K1G1/K3G1 ordered-reverse混合main network実験レポート

作成日: 2026-08-22

## 結論

K1G1型区間とK3G1型区間の順序は、境界の性質に影響する手がかりを示した。ただし、その効果は
seed、学習時点、評価categoryの間で交互作用し、どちらか一方の順序が総合的に優れる結果には
ならなかった。

- K1-firstは文節precisionで強く、step 220では3 seedすべてでK3-firstを上回った。
- K3-firstは22時点dense平均で良い傾向があり、3 seed平均のprecisionは `.633` 対 `.571`、
  fracture record occupancyは `.158` 対 `.220` だった。
- ただしdenseのK3-first優位はseed 43で反転し、step 220のfamily優位もseed 42/43とseed 44で
  反転した。
- 両混合構成とも、親K1G1の強みを保ちながら文節を改善する、再現性のあるPareto解には
  ならなかった。

したがって、順序差を層位置の一般的効果とは結論せず、本ordered/reverse混合を長時間実験の本線には
採用しない。現計画の停止条件に従い、長時間候補はK1G1を維持する。

## 構成と実行条件

| ID | main-network layout | K/G数 |
| --- | --- | --- |
| K1-first | `K1G1` x 7 → `K3G1` x 3 | K=16/G=10 |
| K3-first | `K3G1` x 3 → `K1G1` x 7 | K=16/G=10 |

両構成は総層数26、総parameter数222,204,928、main-network parameter数186,420,224で一致する。
seed 42/43/44を220 stepまで学習し、family 24文を10 step間隔でnative評価した。category 88文と
family 24文を統合した112文はstep 55/110/165/220でlow/central/high/native評価した。

seed 42の55-step pilotにはLR horizonの不一致があったため破棄し、horizon=220で再学習したartifact
だけを使用した。seed 42のstep 55→220再開ではoptimizerとdata positionの復元を監査した。

## step 220 central評価

### seed別結果

`category P/C/F` のFはlexeme fracture offset rateであり、低い方がよい。

| seed | model | category P/C/F | family P/C | integrity | landmark C | 文節 P/C |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 42 | K1-first | .437/.282/.305 | .625/.396 | .958 | .375 | .219/.304 |
| 42 | K3-first | .392/.239/.329 | .611/.375 | .958 | .458 | .161/.217 |
| 43 | K1-first | .435/.231/.290 | .500/.396 | .833 | .500 | .280/.304 |
| 43 | K3-first | .389/.188/.301 | .422/.312 | .833 | .375 | .250/.304 |
| 44 | K1-first | .542/.278/.192 | .556/.346 | .917 | .417 | .385/.435 |
| 44 | K3-first | .541/.342/.101 | .906/.782 | .958 | .917 | .133/.174 |

K1-firstの文節precisionは3 seedすべてで高く、差はK3-first minus K1-firstで
`-.057/-.030/-.251` だった。これは順序によるcategory選好の比較的強い手がかりである。
一方、category coverage、fracture、family、landmarkはseed 44で方向が変わるため、K1-firstの
総合優位とはいえない。

### 3 seed平均と母標準偏差

| model | category P/C/F | family P/C | landmark C | 文節 P/C |
| --- | --- | --- | ---: | --- |
| K1-first | .471±.050 / .264±.023 / .262±.050 | .560±.051 / .379±.024 | .431±.052 | .294±.068 / .348±.061 |
| K3-first | .441±.071 / .256±.064 / .244±.101 | .646±.199 / .490±.208 | .583±.238 | .182±.050 / .232±.054 |

K3-firstのfamily平均は高いが、標準偏差が大きく、seed 44の `.906/.782` に強く依存する。
一方、K1-firstの文節優位は3 seedで方向が一致している。

## dense 22時点評価

fractureは「その評価時点でfractureを含むfamily recordの割合」を全checkpointで平均した値で、
低い方がよい。

| seed | model | time P/C | fracture | late P/C | late fracture | transition |
| ---: | --- | --- | ---: | --- | ---: | ---: |
| 42 | K1-first | .559/.428 | .426 | .500/.424 | .542 | .220 |
| 42 | K3-first | .627/.445 | .326 | .622/.348 | .292 | .226 |
| 43 | K1-first | .688/.387 | .047 | .786/.333 | .042 | .300 |
| 43 | K3-first | .641/.371 | .055 | .800/.364 | .042 | .300 |
| 44 | K1-first | .467/.453 | .186 | .433/.394 | .208 | .310 |
| 44 | K3-first | .630/.528 | .093 | .696/.485 | .042 | .365 |

K3-first minus K1-firstのtime-averaged precision差は `+.068/-.047/+.163`、fracture差は
`-.100/+.008/-.093` だった。平均ではK3-firstが良いが、seed 43で両方の方向が反転するため、
計画で定めた「seed間で方向が反転する場合は層位置効果と結論しない」に該当する。

## 学習時点とcategoryの相互作用

- step 55ではK3-firstのlandmark coverageが3/3 seedで高く、文節precision/coverageはK1-firstが
  3/3 seedで高かった。
- step 110でも文節precisionはK1-firstが3/3 seedで高かった。
- step 165では文節precisionの方向がseed 42だけ反転した。
- step 220では再びK1-firstの文節precisionが3/3 seedで高かった。
- 助動詞、助詞、structuredはseedによって優位構成が変わった。複合語はstep 220でK1-firstが
  3/3 seedでprecision優位だったが、fractureの絶対値とseed差が大きい。

この結果は、層順序が特定の言語的境界に選好を与える可能性を支持する一方、その選好が全categoryや
全時点へ一様に波及しないことを示す。小規模学習では、単純に「前半K3」または「前半K1」を選べば
全指標が改善するとはいえない。

## 仮説判定

1. **親の強みの両立:** 不成立。混合はK1G1より文節を改善した時点があるが、K1G1のcategory、
   family、低fractureを一貫して維持しなかった。
2. **層位置の寄与:** 限定的な手がかりあり。K1-firstの文節、K3-firstの初期landmarkには再現性が
   あるが、denseとfamilyの方向反転により一般的な層位置効果とは結論しない。
3. **位置よりK/G比率:** ordered/reverseが完全に同じではないため位置を無視できない。ただし本実験は
   K/G比率だけの因果効果も証明しない。
4. **長時間候補:** 混合2構成は見送り、K1G1を維持する。

## Artifact

- 学習run: Drive `runs/ordered_reverse_mixing/phase2`, `phase3`
- full probe: Drive `evals/ordered_reverse_mixing/phase1_full112_matched`, `phase2_full112`,
  `phase3_full112`
- 3 seed集計: Drive
  `reports/ordered_reverse_mixing/phase3/phase3_ordered_reverse_reproducibility.json`
- 監査: Drive `manifests/ordered_reverse_mixing/phase2_artifact_audit.json`,
  `phase3_artifact_audit.json`, `phase3_full112_status.json`

これらの指標は、220 step時点の文章生成品質や長期main-network性能を直接測るものではない。
言語的に説明可能な分割を長時間候補選定のproxyとして比較した結果である。
