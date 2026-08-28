# Encoder/Decoder先行warmup（P3）実験レポート

作成日: 2026-08-28

## 1. 目的

本実験の目的は、約200M規模で長期間学習するH-Netのベースモデル探索に向けて、main networkを一時的に固定し、
Encoder・Decoder・boundary routerなどのouter stageだけを先に学習する方法が、言語的に説明可能な分割の形成を改善するか
確認することである。

「最初に学んだ境界が固定される」とは仮定しない。warmup終了後もouter stageはmain networkとともに学習されるため、
境界は現在のデータに合わせて変化し続ける。本実験では、warmupによって境界形成が早まるか、改善が学習終盤にも残るか、
単に20 step多く学習した効果ではないかを分けて評価した。

## 2. Warmupの定義

warmup中は約184.1M parameterの最内層main networkとembedding/LM headを固定し、約35.5M parameterのouter stageを更新した。

更新対象は各stageのEncoder、Decoder、boundary router、dechunk/residual projection、stage間dimension projectionである。
boundary routerは各byte位置をchunk境界にする確率を出す機構で、Encoder/Decoderはchunk化の前後で表現を変換する。

20 step後には、optimizer state、学習データcursor、runtime乱数状態を引き継がず、warmup済みweightだけをjoint trainingへ渡した。
したがってjoint trainingは全条件で同じデータ先頭、同じデータ順、同じruntime seedから開始する。

weight差分監査では、warmupで更新対象98 parameter tensorが変化し、固定対象159 tensorは初期weightと完全一致した。

## 3. 比較条件

main networkはTransformer 26層のT26に固定し、context長は2,048 bytes、三つのseedは42に固定した。

| 条件 | warmup | joint training | 総optimizer update | 比較目的 |
| --- | ---: | ---: | ---: | --- |
| W0 | なし | 220 | 220 | 通常学習の対照 |
| W1 | outerのみ20 | 220 | 240 | 同じjoint学習量でpreconditioning効果を見る |
| W2 | outerのみ20 | 200 | 220 | W0と総update数を揃えるcompute-control |

W1とW0の同一joint step比較は、同じmain network更新量に対するwarmup済みweightの効果を表す。ただしW1は総計算が20 step多い。
W2とW0の総update比較は追加計算を揃えるが、W2のmain network更新量は20 step少ない。この二つを併記することで、
warmupの効果と追加学習量の効果を混同しない。

## 4. 評価方法

Full112は、日本語88文のcategory probeと、活用・複合語などの対応関係を持つ24文のfamily probeからなる。
native条件は学習済みrouterがそのまま選ぶ境界、profile条件は境界数を固定して確率上位を選ぶ評価である。

| 指標 | 大きい／小さい場合の意味 |
| --- | --- |
| category precision ↑ | 選択境界のうち言語categoryで説明できる割合。大きいほど不自然な境界が少ない |
| category coverage ↑ | 許容category境界のうち実際に拾えた割合。大きいほど説明可能な境界を広く拾う |
| fracture record occupancy ↓ | 語彙内部の説明困難な分断を含む文章の割合。小さいほどよい |
| family precision ↑ | family文の選択境界のうち許容境界である割合 |
| family coverage ↑ | family文の許容境界のうち実際に拾えた割合 |
| landmark coverage ↑ | 再利用したいfamily境界を実際に拾えた割合 |
| family integrity ↑ | 保護対象のfamily語彙内部を分断しなかった文章の割合 |
| 文節 precision ↑ | 選択境界のうち文節境界と一致する割合 |
| 文節 coverage ↑ | 定義した文節境界のうち実際に拾えた割合 |

nativeは10 step間隔で評価した。W0/W1はstep 10--220の22時点、W2はstep 10--200の20時点である。
profileはW0/W1のjoint step 110、165、220と、W2の110、165、200でlow/central/high/nativeを評価した。

## 5. Warmup直後

次表は通常joint 20 stepと、outer-only warmup 20 stepのnative分割である。

| 指標 | W0 joint 20 | outer warmup 20 |
| --- | ---: | ---: |
| category precision ↑ | 0.3571 | 0.3548 |
| category coverage ↑ | 0.1068 | 0.1880 |
| fracture ↓ | 0.1591 | 0.1591 |
| family precision ↑ | 0.3125 | 0.1935 |
| family coverage ↑ | 0.1515 | 0.1818 |
| landmark coverage ↑ | 0.1667 | 0.0417 |
| family integrity ↑ | 0.8333 | 0.8333 |
| 文節 precision ↑ | 0.0833 | 0.3448 |
| 文節 coverage ↑ | 0.0435 | 0.4348 |

outer warmupだけでもcategory・文節境界は形成された。特に文節coverageは大きく上がった。一方、family precisionと
landmark coverageは低く、main networkを固定すれば一様に良い境界基盤ができるわけではない。

## 6. W1: 同じjoint学習量での終盤比較

terminal window（joint step 180--220）の平均を比較する。

| 指標 | W0 | W1 | 差 |
| --- | ---: | ---: | ---: |
| category precision ↑ | 0.4110 | 0.4267 | +0.0157 |
| category coverage ↑ | 0.2735 | 0.3581 | +0.0846 |
| fracture ↓ | 0.3068 | 0.2841 | -0.0227 |
| family precision ↑ | 0.7567 | 0.5389 | -0.2179 |
| family coverage ↑ | 0.5273 | 0.5212 | -0.0061 |
| landmark coverage ↑ | 0.6250 | 0.6167 | -0.0083 |
| family integrity ↑ | 0.9583 | 0.9250 | -0.0333 |
| 文節 precision ↑ | 0.1967 | 0.1938 | -0.0029 |
| 文節 coverage ↑ | 0.2870 | 0.2609 | -0.0261 |

W1はcategory coverageを高め、fractureも減らした。family coverage、landmark、integrityの終盤平均は概ね維持したが、
family precisionは0.218低下した。つまり、許容family境界を拾う数はほぼ同じでも、それ以外の説明しにくい境界がfamily文で
増えている。これは単純な「形成の早期化」ではなく、境界配分がcategory coverage重視の別軌跡へ移ったことを表す。

さらにfamily integrityはstep 210と220でW0より0.083低く、relaxed許容幅0.10以内ではあるがprimary許容幅0.05を連続して超えた。

## 7. W2: 同じ総update数での終盤比較

W0の総update 200--220と、W2のwarmup込み総update 200--220、すなわちjoint step 180--200を比較する。

| 指標 | W0 | W2 | 差 |
| --- | ---: | ---: | ---: |
| category precision ↑ | 0.4027 | 0.4057 | +0.0030 |
| category coverage ↑ | 0.2721 | 0.4145 | +0.1425 |
| fracture ↓ | 0.3144 | 0.4053 | +0.0909 |
| family precision ↑ | 0.7648 | 0.5506 | -0.2142 |
| family coverage ↑ | 0.5253 | 0.5657 | +0.0404 |
| landmark coverage ↑ | 0.6250 | 0.6528 | +0.0278 |
| family integrity ↑ | 0.9583 | 0.8889 | -0.0694 |
| 文節 precision ↑ | 0.1830 | 0.1502 | -0.0328 |
| 文節 coverage ↑ | 0.2609 | 0.2174 | -0.0435 |

総update数を揃えてもcategory coverage、family coverage、landmarkは上がった。しかしfractureは0.091増え、family precisionは
0.214、integrityは0.069低下した。coverageの増加が、説明困難な語彙内部境界の増加を伴っているため、他指標で相殺できない。

## 8. 圧縮profile

W1は同じjoint step 110/165/220の3時点平均、W2は同じ総update数の終点でW0との差を見た。

### 8.1 W1 minus W0

| profile | category P | category C | fracture | family P | family C | landmark C | integrity | 文節 P | 文節 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-compression | +0.0205 | +0.0328 | -0.0833 | -0.1712 | -0.0505 | -0.0278 | +0.0417 | +0.0262 | 0.0000 |
| central | -0.0209 | +0.0484 | +0.0455 | -0.1107 | +0.0808 | +0.1111 | -0.0278 | -0.0163 | 0.0000 |
| high-compression | -0.0240 | +0.0584 | +0.0720 | -0.2353 | +0.0909 | +0.0833 | 0.0000 | +0.0164 | +0.0435 |
| native | +0.0281 | +0.1083 | +0.0038 | -0.2156 | -0.0202 | -0.0833 | 0.0000 | +0.0419 | +0.0870 |

W1はすべてのprofileでcategory coverageを上げたが、family precisionもすべてで0.11--0.24下げた。高圧縮では
family/landmark coverageが上がる一方、fractureが増える。低圧縮ではfractureは減るがfamily coverageが下がる。
したがって、特定の境界budgetだけで解消するtrade-offではない。

### 8.2 W2 minus W0、総update 220時点

| profile | category P | category C | fracture | family P | family C | landmark C | integrity | 文節 P | 文節 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-compression | +0.0368 | +0.0470 | +0.0114 | +0.0251 | -0.1515 | -0.0833 | -0.0417 | +0.0084 | -0.0870 |
| central | -0.0325 | +0.0684 | +0.1250 | -0.0811 | +0.0606 | +0.0833 | -0.1250 | -0.0268 | 0.0000 |
| high-compression | -0.0235 | +0.0726 | +0.1250 | -0.2941 | +0.0909 | +0.1250 | -0.0833 | +0.0051 | +0.0435 |
| native | +0.0196 | +0.1538 | +0.0795 | -0.2102 | +0.0303 | 0.0000 | -0.0417 | -0.0050 | 0.0000 |

central/high-compressionのfractureは許容幅を超え、centralのintegrityも0.125低下した。W2はcoverageを広げるが、
同じ総計算量ではmain networkのjoint更新を20 step減らした影響もあり、品質制約を維持できない。

## 9. 解釈

outer stageはmain networkを固定しても、短時間で文節やcategory境界を学習できる。したがってEncoder/Decoder/routerの高い
学習率は境界形成を実際に加速する。しかし、形成されるのは「後でそのまま使える中立な境界基盤」ではない。
outer stageだけが現在の表現とLM lossに適応すると、category coverageを広げる方向が強くなり、family文では余分な境界が増えた。

joint trainingを220 step行ってもこの特徴は完全には消えず、W1のfamily precision低下として残った。一方、W2ではmain networkの
更新量が少ないため、outer stageとmain networkの協調がさらに不足し、fractureとintegrityが悪化したと考えられる。

この結果はwarmup一般を否定するものではなく、今回の「outer全体を20 step、通常LM lossで先行更新する」仕様が強すぎることを示す。
ただし現在は要因探索段階であり、warmup長や更新範囲をFull112に合わせて細かく調整すると評価probeへの過適合になるため、
このworkstream内で追加探索は行わない。

## 10. 判定

P3 warmupはcategory coverageを持続的に改善したが、W1ではfamily precisionが大きく低下し、複数profileでfractureまたは
family/integrity制約に違反した。W2も同じ総update数でfractureとintegrityを維持できない。よって今回の仕様は不採用とする。

K1G1/K3G1やP1/P2との組合せには展開しない。P1--P3の独立実験で採用条件を満たす要因が残らなかったため、ロードマップの
次段階はP4「FFN-MoE」のT26 pilotである。MoEではmain mixerを変えずにFFN容量と専門化だけを変え、境界形成への影響を分離する。

## 11. Artifact

- 学習run: Drive `runs/outer_warmup_p3_v1/`
- 共通初期weight・warmup済みweight: Drive `artifacts/outer_warmup_p3_v1/`
- native軌跡: Drive `analysis/outer_warmup_p3_v1_native/`
- warmup単独比較: Drive `analysis/outer_warmup_p3_v1_warmup_only/`
- profile評価: Drive `evals/outer_warmup_p3_v1/profiles/`
- profile集計: Drive `analysis/outer_warmup_p3_v1_profiles/`
