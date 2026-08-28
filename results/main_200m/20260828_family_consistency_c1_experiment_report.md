# Family consistency loss（C1）実験レポート

作成日: 2026-08-28

## 1. 目的

本実験の目的は、約200M規模で長期間学習するH-Netのベースモデル探索に向けて、似た語族・活用・表記の間で
境界判断を揃える補助損失が、文章の分割品質を改善できるか確認することである。

H-Netではmain networkだけでなく、outer stageのEncoder・Decoder・boundary routerも学習される。そのため、同じ語幹や
複合語境界でも文脈や学習stepによって異なる位置で分割されることがある。適切な揺れは動的チャンクの特徴だが、
再利用したい語彙境界まで無関係に変化すると、後続networkは同じ言語要素を一貫した単位として扱いにくくなる可能性がある。

そこでC1では、対応関係が明確な二文のlandmarkにおける境界確率を近づけた。ただし、小規模学習のlossや生成文を
最終性能とはみなさず、Full112 probeにおける分割の形成軌跡を主判定にした。

## 2. C1とは何か

### 2.1 Family pairとlandmark

Family pairは、共通する言語要素を持つ二つの文章である。例えば活用、助動詞、助詞、複合語、structured text、
identifierのようなカテゴリごとにペアを作り、二文で対応する位置をlandmarkとして指定した。

C1はstage 0 boundary routerが出す二つのlandmarkの境界確率を`p_left`、`p_right`として、次のMSEを最小化する。

`L_C1 = (p_left - p_right)^2`

MSEは小さいほど二文の判断が一貫していることを表す。ただし、両方の確率を同時に下げてもMSEは小さくなる。
したがってMSEだけでは成功と判定せず、landmark確率、Full112のcoverage、integrity、実際の分割を同時に確認した。

### 2.2 補助データ

- train: 24 pair
- dev: 6 pair
- test: 6 pair
- カテゴリ: 活用、助動詞、助詞、複合語、structured text、identifier
- Full112と文章全体が一致する例は除外

補助pairはLM batchとは別iteratorで読み、LMデータの内容・順序・累積raw bytesを変えない。sham条件でも同じpairを
同じ順にforwardし、C1の重みだけを0にした。これにより、追加forwardの有無をC1の効果へ混ぜていない。

## 3. 比較条件

main networkはT26（Transformer 26層）に固定した。model初期値、LMデータ順、training runtime、補助pair順のseedは
すべて42に固定した。

| 条件 | C1係数 | 用途 |
| --- | ---: | --- |
| sham | 0 | 同じ補助forwardを行う対照 |
| low-0.25 | 0.25 | 最終的にstep 220まで追跡した弱用量 |
| low-1 | 1 | 55-step用量pilot |
| medium-5 | 5 | 55-step用量pilot |

係数1と5ではC1 MSEが下がる一方、family/landmark coverageの低下が早期に現れた。係数0.25はstep 50で一度
coverageを回復したため、境界が再び変化する可能性を考慮してstep 100、さらにstep 220まで延長した。

## 4. 評価指標

主評価はFull112のnative条件、stage 1で行った。指標は次の意味を持つ。

| 指標 | 大きい／小さい場合の意味 |
| --- | --- |
| category precision | 大きいほど、選択された境界を言語カテゴリで説明しやすい |
| category coverage | 大きいほど、許容される言語カテゴリ境界を広く拾う |
| fracture record occupancy | 小さいほど、語彙内部の説明困難な分断を含む文章が少ない |
| family precision | 大きいほど、family文で選択された境界を説明しやすい |
| family coverage | 大きいほど、family文の許容境界を広く拾う |
| landmark coverage | 大きいほど、再利用したいfamily landmarkを実際に境界として選ぶ |
| family integrity | 大きいほど、保護したいfamily語彙内部を壊さない |
| 文節 precision | 大きいほど、選択境界が文節境界と一致しやすい |
| 文節 coverage | 大きいほど、文節境界を広く拾う |

単一stepではなく10 step間隔の22時点を保持し、特にterminal window（step 180--220）を持続性判定に使った。
また、step 110、165、220ではlow-compression、central、high-compression、nativeの4 profileを評価した。

## 5. C1そのものの効果

### 5.1 独立dev/test、step 220

| split | 条件 | landmark確率差MSE ↓ | 平均landmark確率 |
| --- | --- | ---: | ---: |
| dev | sham | 0.001816 | 0.6947 |
| dev | low-0.25 | 0.000389 | 0.6787 |
| test | sham | 0.009219 | 0.7119 |
| test | low-0.25 | 0.006236 | 0.7160 |

low-0.25はdev MSEを約79%、test MSEを約32%下げた。testの平均landmark確率は維持され、devの低下も約0.016
に留まったため、補助データ上の改善は単純な全確率collapseだけでは説明できない。C1が未見pairにも一定の一貫性を
与えたことは確認できる。

## 6. Full112の分割品質

### 6.1 terminal window（step 180--220）

| 指標 | sham | low-0.25 | 差 |
| --- | ---: | ---: | ---: |
| category precision ↑ | 0.3681 | 0.4378 | +0.0697 |
| category coverage ↑ | 0.2530 | 0.2889 | +0.0359 |
| fracture occupancy ↓ | 0.3341 | 0.3068 | -0.0273 |
| family precision ↑ | 0.6826 | 0.6397 | -0.0429 |
| family coverage ↑ | 0.5212 | 0.4424 | -0.0788 |
| landmark coverage ↑ | 0.6750 | 0.5583 | -0.1167 |
| family integrity ↑ | 0.8333 | 0.7917 | -0.0417 |
| 文節 precision ↑ | 0.0772 | 0.2257 | +0.1485 |
| 文節 coverage ↑ | 0.0870 | 0.3478 | +0.2609 |

C1はcategory precision/coverage、fracture、文節precision/coverageを同時に改善した。一方で、直接改善したかった
family coverageとlandmark coverageは低下した。特にlandmark coverageはterminal 5時点中3時点で、shamから
0.10を超えて低下した。これは計画で定めたrelaxed許容幅の継続違反に当たる。

### 6.2 profile評価

step 110、165、220の平均では、centralとhigh-compressionで退行が比較的小さく、category precision、fracture、
文節指標は概ね改善した。しかし、nativeではfamily coverageとlandmark coverageがともに約0.111低下した。
low-compressionでは平均landmark coverageを維持したものの、step 220のfamily integrityがshamより0.292低かった。

このため、C1の効果は境界budgetに依存する。あるbudgetでは良い境界を選びやすくしても、nativeの学習済みhard境界では
family landmarkを保持する保証になっていない。

## 7. 解釈

C1は「二つの確率を同じにする」目的であり、その位置を積極的に境界にする目的ではない。したがって次の二つを
区別できない。

1. 両方の文章で再利用可能なlandmarkを高確率にする。
2. 両方の文章でlandmarkを低確率にして判断を揃える。

補助dev/testでは平均確率をほぼ維持したが、Full112の異なるfamilyへ一般化したとき、category・文節境界を増やす代わりに
family landmarkを落とすtrade-offが残った。少数の補助pairに対する確率一致だけでは、未知のfamilyで「どこを残し、
どこを壊さないか」を十分に指定できていない。

## 8. 判定と次の実験

C1には境界形成を変える実効果があり、独立pairの一貫性、category、fracture、文節には改善が見られた。しかし、
terminalのfamily/landmark coverageが非相殺制約を満たさないため、C1単独は不採用とする。K1G1/K3G1への展開や
P1との組合せも、この段階では行わない。

次は計画どおりC2 integrity marginをC1と混ぜずに独立検証する。C2では、保護span内部の境界確率を外側の有効な
landmarkより低くするmarginを明示する。これにより、「family間で確率を揃える」だけでなく、「内部fractureを抑えながら
必要なlandmarkを残す」という方向を直接与えられるか確認する。C1+C2は、C2単独が通過した場合だけ検討する。

## 9. Artifact

- 学習run: Drive `runs/family_consistency_c1_v1/`
- 独立dev/test: Drive `evals/family_consistency_c1_v1/`
- 22時点集計: Drive `analysis/family_consistency_c1_v1_t26_step220/`
- profile評価: Drive `evals/family_consistency_c1_v1/profiles/`
- profile集計: Drive `analysis/family_consistency_c1_v1_t26_profiles/`

