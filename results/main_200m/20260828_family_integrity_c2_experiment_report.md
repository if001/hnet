# Family integrity margin（C2）実験レポート

作成日: 2026-08-28

## 1. 目的

本実験の目的は、約200M規模で長期間学習するH-Netのベースモデル探索に向けて、言語的に一単位として扱いたい
spanの内部を分断せず、その外側にある再利用可能な境界を残す補助損失が、文章の分割品質を改善できるか確認することである。

H-Netの境界は学習中に変化する。そのため、あるstepだけの値や小規模学習の生成文ではなく、Full112 probeで境界品質が
どのように形成・維持されるかを主に評価した。C2以外の要因を混ぜないため、main networkはT26、モデル初期値、LMデータ順、
runtime乱数、補助データ順を固定した。

## 2. C2とは何か

### 2.1 landmarkとprotected span

landmarkは、複合語と接辞の間など、後続networkが再利用しやすいと考える境界候補である。protected spanは、語彙内部など、
一つのchunkとして保持したい範囲である。

C2はlandmarkの境界確率を`p_landmark`、protected span内部の境界確率を`p_internal`、必要な確率差をmarginとして、
次のlossを最小化する。

`L_C2 = ReLU(margin + p_internal - p_landmark)`

つまり、landmarkの確率が内部境界よりmargin以上高ければlossは0になる。C1が二文の同じ位置の確率を揃える目的だったのに
対し、C2は「残したい外側境界」と「抑えたい内部境界」の向きを明示する。

独立dev/testでは次の三指標を使う。

| 指標 | 意味 |
| --- | --- |
| margin loss ↓ | 小さいほど、landmarkが内部境界より指定marginだけ高い条件を満たしやすい |
| satisfaction rate ↑ | 大きいほど、margin条件を満たした比較の割合が多い |
| probability gap ↑ | `p_landmark - p_internal`。大きいほど、残したい境界と抑えたい境界を明確に区別する |

### 2.2 補助データ

- train: 24 pair
- dev: 6 pair
- test: 6 pair
- カテゴリ: 活用、助動詞、助詞、複合語、structured text、identifier
- Full112と文章全体が一致する例は除外

補助pairはLM batchとは別iteratorで供給した。対照のshamも同じpairを同じ順番でforwardし、C2係数だけを0にした。
したがって、LMデータの内容・順序・累積raw bytesと、補助forwardの有無をC2の効果へ混ぜていない。

## 3. 比較条件

main networkはTransformer 26層のT26に固定した。marginは0.15とした。

| 条件 | C2係数 | 学習範囲 | 用途 |
| --- | ---: | ---: | --- |
| sham | 0 | step 220 | 同じ補助forwardを行う対照 |
| low-0.002 | 0.002 | step 55 | 弱用量pilot |
| medium-0.01 | 0.01 | step 220 | pilotで選び、軌跡を追跡した用量 |

low-0.002はstep 55でFull112のfractureとintegrityが悪化したため延長しなかった。medium-0.01は補助目的を明確に改善し、
早期のFull112退行もlowより小さかったため、shamとともにstep 220まで追跡した。両条件はstep 55、100、110、165、220の
checkpointを保存し、nativeでは10 step間隔の22時点を評価した。

## 4. Full112の評価指標

Full112は、日本語88文のcategory probeと、活用・複合語などの対応関係を持つ24文のfamily probeからなる。
主評価は学習済みrouterがそのまま選ぶnative条件、stage 1で行った。

| 指標 | 大きい／小さい場合の意味 |
| --- | --- |
| category precision ↑ | 選択された境界のうち、言語カテゴリで説明できる境界の割合。大きいほど不自然な選択が少ない |
| category coverage ↑ | 許容される言語カテゴリ境界のうち、実際に選べた割合。大きいほど説明可能な境界を広く拾う |
| fracture record occupancy ↓ | 語彙内部の説明困難な分断を一つ以上含む文章の割合。小さいほどよい |
| family precision ↑ | family文で選択された境界のうち、許容境界である割合 |
| family coverage ↑ | family文で定義した許容境界のうち、実際に選べた割合 |
| landmark coverage ↑ | 再利用したいfamily landmarkを実際に境界として選べた割合 |
| family integrity ↑ | 保護したいfamily語彙内部を分断しなかった文章の割合 |
| 文節 precision ↑ | 選択された境界のうち、文節境界と一致する割合 |
| 文節 coverage ↑ | 定義した文節境界のうち、実際に選べた割合 |

precisionだけを上げるには境界数を極端に減らす方法があり、coverageだけを上げるには境界を過剰に増やす方法がある。
そのためprecision・coverage・fracture・integrityを組み合わせ、さらにstep軌跡と異なる圧縮条件を確認した。

## 5. C2目的そのものへの効果

### 5.1 独立dev/test、step 220

| split | 条件 | margin loss ↓ | satisfaction ↑ | landmark確率 | 内部境界確率 | probability gap ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev | sham | 0.1084 | 0.2917 | 0.6263 | 0.5639 | 0.0624 |
| dev | medium-0.01 | 0.0592 | 0.4792 | 0.6654 | 0.5326 | 0.1328 |
| test | sham | 0.1765 | 0.2000 | 0.5907 | 0.6000 | -0.0094 |
| test | medium-0.01 | 0.0888 | 0.2708 | 0.6667 | 0.5576 | 0.1090 |

medium-0.01は未見のdev/testでmargin lossを下げ、satisfactionとgapを上げた。特にtestでは、shamで内部境界の方が
わずかに高かったgapが正に転じた。C2はtrain例の暗記だけでなく、「landmarkを内部境界より高くする」という目的を
未見pairにも一般化した。

## 6. Full112 nativeの分割品質

### 6.1 terminal window（step 180--220）

| 指標 | sham | medium-0.01 | 差 |
| --- | ---: | ---: | ---: |
| category precision ↑ | 0.4016 | 0.4714 | +0.0697 |
| category coverage ↑ | 0.2547 | 0.3487 | +0.0940 |
| fracture occupancy ↓ | 0.2568 | 0.2773 | +0.0205 |
| family precision ↑ | 0.6936 | 0.6395 | -0.0541 |
| family coverage ↑ | 0.4667 | 0.3879 | -0.0788 |
| landmark coverage ↑ | 0.5917 | 0.5333 | -0.0583 |
| family integrity ↑ | 0.8583 | 0.8500 | -0.0083 |
| 文節 precision ↑ | 0.1537 | 0.2178 | +0.0641 |
| 文節 coverage ↑ | 0.2261 | 0.3478 | +0.1217 |

C2はcategory precision/coverageと文節precision/coverageを大きく改善し、integrityもほぼ維持した。一方、family
precision/coverageとlandmark coverageは低下した。fractureの悪化は+0.0205で許容幅+0.05以内だが、step 200、210、220では
family coverageまたはlandmark coverageがshamより0.10を超えて低くなった。これは単一stepの偶然ではなく、学習後半に
続いた退行である。

### 6.2 より広い軌跡（step 110--220）

step 110--220の平均でも、category precisionは0.4145から0.4596へ、文節coverageは0.2065から0.2391へ改善した。
fractureは0.2727から0.2670、integrityは0.8785から0.8924で悪化していない。一方、family coverageは0.5152から
0.4192、landmark coverageは0.6285から0.5660へ低下した。したがって、C2の特徴はterminal windowだけの局所的な現象ではない。

## 7. 圧縮profile評価

step 110、165、220で、境界budgetを変えた4条件を評価した。low-compressionは境界を多く残し、high-compressionは
少ない境界だけを残す。centralは中間の固定budget、nativeはモデル自身のhard boundaryである。固定budgetを見ることで、
単なる境界数の違いと、境界の順位付け自体の違いを分けて考えられる。

次表は3時点平均のmedium-0.01 minus shamである。

| profile | category P | category C | fracture | family P | family C | landmark C | integrity | 文節 P | 文節 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-compression | +0.0516 | +0.0356 | -0.0492 | -0.1817 | -0.1010 | -0.1250 | -0.1667 | +0.0703 | +0.1304 |
| central | +0.0160 | +0.0043 | -0.0227 | -0.0528 | -0.0101 | -0.0139 | -0.0278 | -0.0049 | +0.0000 |
| high-compression | +0.0125 | +0.0014 | -0.0303 | -0.0333 | +0.0303 | +0.0417 | -0.0556 | -0.0344 | -0.0145 |
| native | +0.0479 | +0.0100 | -0.0455 | +0.0452 | -0.1212 | -0.0972 | +0.0694 | +0.0234 | +0.0290 |

高圧縮ではfamily/landmark coverageも改善し、centralでは差が小さい。対して、境界を多く残すlow-compressionではfamily
precision/coverage、landmark、integrityがそろって低下し、nativeではfamily precisionとintegrityが上がる一方でcoverageが
下がった。これはC2が「良い境界をすべて増やす」のではなく、限られた境界確率の順位と配分を変えたことを示す。

非相殺制約では、low-compressionの12比較中8件、centralの12比較中3件、high-compressionの12比較中1件、nativeの
12比較中5件が許容幅を超えた。特にnative step 220ではfamily coverageが-0.1515、landmark coverageが-0.1667であり、
採用判断に必要な持続性を満たさない。

## 8. 解釈

C2は、指定したprotected pairに対しては狙いどおりlandmarkを上げ、内部境界を下げた。またcategory・文節の説明可能な
分割も増やした。しかし、少数のprotected spanで学んだ相対marginは、Full112に含まれる別のfamilyで許容landmarkを
広く残すことを保証しない。

重要なのは、family integrityが概ね維持または改善してもfamily coverageが下がり得る点である。これは語彙内部の病的な
分断を減らすことと、再利用可能な外側境界を十分に選ぶことが別の問題だからである。C2は前者と局所的なlandmark順位を
改善したが、nativeのglobalな境界配分では一部のfamily landmarkが選択圏外へ移ったと考えられる。

したがって、「C2目的が下がった」だけでは採用できない。独立補助評価は目的関数が実際に働いたことを示し、Full112は
その働き方が望む一般化になったかを検査する。今回、この二つは異なる結論を示した。

## 9. 判定と次の段階

C2 medium-0.01は、独立dev/test、category、文節、fractureに明確な改善を示した。一方、native終盤のfamily coverageと
landmark coverage、およびlow-compressionのfamily指標が非相殺制約を継続して超えたため、C2単独は不採用とする。

計画ではC1とC2が単独通過した場合だけC1+C2を試すため、両者を組み合わせない。K1G1/K3G1への展開も行わない。
次は境界lossをさらに調整するのではなく、独立要因P3のencoder/decoder-only warmupへ進む。P3ではmain networkを
一時的にfreezeしてouter stageを20 step先行学習し、joint training開始後の境界形成速度とterminal品質を対照条件と比較する。

## 10. Artifact

- 補助データ: Drive `datasets/family_consistency/c2_pilot_v1.json`
- 学習run: Drive `runs/family_integrity_c2_v1/`
- 独立dev/test: Drive `evals/family_integrity_c2_v1/`
- native 22時点集計: Drive `analysis/family_integrity_c2_v1_t26_step220/`
- profile評価: Drive `evals/family_integrity_c2_v1/profiles/`
- profile集計: Drive `analysis/family_integrity_c2_v1_t26_profiles/`
