# 共通対照・軌跡評価protocol

作成日: 2026-08-26

## 1. 固定条件

短contextのP1--P5では、対象要因以外を次のように固定する。

- anchors: T26、K1G1、K3G1
- `model_init_seed=42`, `data_order_seed=42`, `train_runtime_seed=42`
- context length: 2048 bytes
- max step / LR horizon: 220 / 220
- dataset manifest、raw batch列、batch、gradient accumulation、optimizer、LR、compression ratio、ratio lossを一致
- byte boundary constraint: `utf8-hard`
- probe: category 88文 + family 24文のfull112 v1
- 主評価: native、stage 1
- 診断: stage 0、low/central/high profile、compression、selected boundary count

要因がparameter数またはactive computeを変える場合は、同一parameterと同一computeを同時には満たせない。
その場合はprimary comparisonを事前指定し、総parameter、active parameter、推定FLOPs、peak memoryを必ず併記する。

## 2. 評価時点

- native dense: step 10, 20, ..., 220
- early gate: step 10--100
- sustained evaluation: step 110--220
- milestone profiles: step 110, 165, 220 x low/central/high/native
- 分割gallery: step 20, 50, 100, 150, 200, 220

windowは既存結果と比較可能にするため次を使う。

- initial: 10--50
- medium: 60--110
- late: 120--170
- terminal: 180--220

## 3. 主指標

stage 1の次の9値を原子的に保持し、precisionとcoverageを合成しない。

1. category precision: 大きいほど、選択境界を言語categoryで説明しやすい。
2. category coverage: 大きいほど、許容category境界を広く拾う。
3. category fracture record occupancy: 小さいほど、語彙内部の説明困難な分断を含む文が少ない。
4. family precision
5. family coverage
6. landmark coverage
7. family integrity: 大きいほど、family内の保護語彙を壊さない。
8. 文節precision
9. 文節coverage

補助としてcategory別P/C、boundary数、chunk長分布、compression、boundary probability margin、
segmentation signature、transition、validation BPB、CE/ratio/auxiliary lossを保存する。

## 4. 軌跡の集計

単一stepまたは全step単純平均を主結論にしない。各構成・条件について次を作る。

1. step x 9指標のCSVと構成別表。
2. 各windowのmean、median、P/Cのq20、fractureのq80、MAD、range。
3. thresholdへの初回到達と3時点連続stable-attainment step。
4. terminal 5時点の傾き、改善step割合、最大drawdown、回復の有無。
5. category Cを横軸、P/F/integrityを縦軸とした時間順trajectory。
6. category coverage差 `.03`以内のmatched-state比較。`.02/.05`でも感度分析する。
7. record単位のtransition率。ただし順位には含めない。

同一step paired差はfixed-training-dose、matched-stateは似たcoverageでの品質差を表す。二つを混同しない。

## 5. 非相殺制約

各variantは最初に同じanchorのcontrolと比較する。primary許容幅は次とし、relaxed幅も併記する。

| 指標 | primary | relaxed |
| --- | ---: | ---: |
| category fracture | control `+.05`以内 | `+.10`以内 |
| family integrity | control `-.05`以内 | `-.10`以内 |
| family coverage | control `-.05`以内 | `-.10`以内 |
| landmark coverage | control `-.05`以内 | `-.10`以内 |

全22時点とterminal 5時点について違反占有率を出す。重大退行をprecision/coverageの順位で相殺しない。

### step 100継続条件

- NaN、OOMの反復、compression collapseがない。
- primary制約のうち2つ以上をstep 60--100の全時点で継続違反しない。
- 少なくとも1つの目的指標に、単一peakでない改善方向がある。
- category別または分割galleryで、明らかな病的短断片の増加がない。

### step 220採用条件

- terminalでrelaxed制約を継続違反しない。
- 改善がterminal 5時点の3時点以上、またはmatched-stateで同方向に残る。
- 改善と制約維持が異なる時点にしか存在する状態ではない。
- 3 anchorsの少なくとも2つで同方向、または特定anchorでのみ効く理由がarchitecture interactionとして説明できる。

## 6. seedの扱い

探索段階で全候補のseed gridは行わない。

1. screening: `(i42,d42,r42)`。
2. winner確認: `(i43,d42,r42)`でweight初期値だけ変更。
3. data確認: 同じstep 0 weightを使い`(i42,d43,r42)`でdata orderだけ変更。
4. runtime seed: 同一triple反復幅が無視できない場合だけ変更。
5. 最終候補では必要に応じて`(i43,d43,r42)`を追加してinteractionを確認する。

run-level seed平均とfactorized seed系列を混ぜない。step 0 state hash、shuffle hash、先頭batch ID列、
checkpoint RNG stateをmanifestへ保存する。

## 7. 追加すべき診断

これまでの結果を踏まえ、次を新たに共通保存する。

- 境界の出生・消滅・再出現率: 同じprobe位置の境界がいつ形成され、何step持続したか。
- category間trade-off matrix: あるcategoryの改善と同時に悪化するcategoryを確認する。
- boundary probability margin: hard選択の僅差による見かけの揺れと表現自体の変化を区別する。
- stage 0→stage 1伝播: stage 0の変更がstage 1評価の変化に先行するかを診断する。
- chunk count / effective main tokens: 同じraw bytesでもmain networkが処理したchunk数が異なる点を記録する。
- terminalだけでなくbest-so-farからのdrawdown: 一度形成された品質を維持できるかを見る。

## 8. artifact監査

各runでconfig、commit、dataset/probe hash、3 seed、step 0 state、LR、累積raw bytes、累積main chunks、
checkpoint、raw evaluation数をstatus JSONへ保存する。失敗runを0値として集計せず、欠測として明示する。
