# full112時間安定性 main-network再screening計画

作成日: 2026-08-24

## 1. 背景と修正点

`results/main_200m/20260824_kda_dosage_position_seed_factorized_screening_report.md`のPhase 1表は、
category、family、文節をstep 55・centralの単点で示し、`dense P/C/F`だけをfamily 24文・native・
step 10/20/30/40/50から集計していた。

このため、次は確認できていない。

- category 88文と文節categoryが学習中の複数stepで高いか。
- step 55の差が一時的な揺れか、main-networkに伴う持続的なbiasか。
- K1G1に対する改善・退行が同じstepでpairedに現れるか。

分割位置そのものは学習中に変化してよい。評価対象は「同じ境界を固定して選ぶこと」ではなく、各stepで
言語的に説明可能な分割品質を維持することである。本計画ではfull112を複数stepで測り直し、単点順位を
時間安定性に基づく判断へ更新する。

## 2. 目的

1. category 88文 + family 24文のfull112について、step 10--50の時間軌跡を取得する。
2. 高い平均だけでなく、中央値、下位分位点、late window、退行頻度を比較する。
3. K1G1との同一step paired差により、全候補が同じ学習時点で改善するかを確認する。
4. 境界位置のtransitionと、precision/coverage/fractureの品質安定性を分けて扱う。
5. 既存レポートのK15/K16 specialist判断とK1G1 anchor判断を再検証する。

本実験も55 stepのproxyであり、長期main-network性能や生成品質は証明しない。

## 3. 対象構成

同一プロトコルで次の11構成を再学習する。過去artifactは中間stepのfull112を持たないため、endpointだけを
混ぜず全構成を再実行する。

### KDA量・位置候補

- K1G1
- K14-front
- K14-middle
- K14-late
- K15-split
- K16-even

### 既存control

- K3G1
- K3T1
- T26
- K1-first
- K3-first

速度は順位に使用しない。T26も同じ境界proxyだけで比較する。

## 4. 固定条件

- `model_init_seed=42`
- `data_order_seed=42`
- `train_runtime_seed=42`
- max step 55、LR horizon 220
- packed train/validation datasetは前実験と同一
- sequence length、batch、gradient accumulation、optimizer、LR、ratio lossを前実験と同一
- compression targetと`utf8-hard`を構成ごとの既存matched条件と同一
- stage 1を主判定、stage 0を診断用とする
- full112 v1の文章・annotationを変更しない

## 5. 取得する評価

### 5.1 時間軌跡

full112をnative条件でstep 10/20/30/40/50に評価する。各stepで次を保存する。

- category全体 precision/coverage/fracture
- 11 category別 precision/coverage/fracture
- 文節、助動詞、助詞、複合語、structured、固有名詞等
- family precision/coverage/integrity
- landmark coverage/consistency、context consistency
- unexplained/fracture record occupancy
- recordごとの選択境界signature

### 5.2 Profile頑健性

各再学習runのstep 55 checkpointをlow/central/high/nativeでfull112評価する。時間安定性の主評価はnative、
profile頑健性はendpointの別軸として扱い、両者を一つの平均へ混ぜない。

## 6. 時間安定性の集計

precision/coverageは各step内でmicro集計した後、step間を等重みで扱う。selected境界数を全stepでpoolして
一時点を過度に重くしない。

各指標について次を出す。

- 5時点の値とtime mean
- median、MADまたは標準偏差
- precision/coverageの20%点と最小値
- fractureの80%点と最大値
- late mean/median: step 30/40/50
- segmentation transition rate
- K1G1との同一step paired差
- paired勝率: 差が正のstep割合。fractureは差が負の割合
- practical regression occupancy:
  - precision/coverage/landmarkがK1G1より`-0.05`以下となるstep割合
  - fractureがK1G1より`+0.10`以上となるstep割合

transition rateが高いだけでは失格にしない。異なるが説明可能な分割へ移る場合があるため、品質指標が安定
しているかを優先する。

## 7. 候補判定

固定aggregate scoreは作らず、時間方向のPareto条件で判断する。

候補昇格には少なくとも次を要求する。

1. 文節precisionまたはcoverageのpaired差が5時点中3時点以上、late 3時点中2時点以上で正。
2. 改善のmedian paired差が1--2境界だけでなく、実用差の目安`.05`以上。
3. category全体でもprecisionまたはcoverageが同方向で、単一categoryだけに依存しない。
4. family coverage、landmark coverage、助動詞、助詞、複合語、structuredの複数軸で、`-.05`以下の
   regression occupancyが継続的に高くない。
5. fracture `+.10`以上の退行がlate windowで継続しない。
6. step 55のlow/central/high/nativeの少なくとも2 profileで同方向を確認する。

平均は高いが下位分位点やlate windowが弱い構成は「不安定なspecialist」とする。平均はやや低くても、
下位分位点とpaired勝率が高い構成を長時間候補として優先する。

## 8. 段階

### Phase 0: runner対応

1. dense runnerのprobeを24文固定から、任意の非空・text一意probeへ一般化する。
2. manifestへprobe件数とhashを保存する。
3. full112 temporal用run prefixを使い、既存dense artifactと名前を区別する。
4. T26を同runnerのmatched targetへ追加する。
5. focused unit testとColab import/GPU smoke testを行う。

### Phase 1: 11構成baseline trajectory

11構成を`(42,42,42)`で55 step学習し、full112 nativeを5時点保存する。全run終了後にstep 55の4 profile
評価を行い、時間安定性表、trajectory図、blind galleryを作る。

### Phase 2: 必要時だけseed trajectory

Phase 1で時間安定性条件を満たした上位1--2候補とK1G1だけを対象とする。

- init差 `(43,42,42)`
- data差 `(42,43,42)`。architecture別step 0 stateをロードする

最初はbaselineと1差分のpaired trajectoryで方向を確認する。順位反転または
僅差の場合だけinit-44/data-44を追加する。既存Phase 2のstep 55結果はendpoint補助情報として用いるが、
時間軌跡の代用にはしない。

## 9. 停止条件

- Phase 1でK1G1より時間安定した有用Pareto軸を持つ候補がなければseed trajectoryを追加しない。
- 一時的なpeakだけの候補は延長しない。
- 文節改善とfamily/landmark維持が別stepにしか現れない候補はcombined候補としない。
- 11構成で差が小さい場合、追加layoutを増やさずK1G1をanchorとして維持する。

## 10. 成果物

| artifact | 保存先 |
| --- | --- |
| 本計画 | `plan/mid/20260824_full112_temporal_stability_rescreening_plan.md` |
| runner/tests | `scripts/`, `tests/unit/` |
| checkpoints/raw trajectory/evals | Drive `/content/drive/MyDrive/hnet_agent_200m_main/` |
| 中間memo | `plan/mid/` |
| 更新report | `results/main_200m/` |

## 11. 実行規則

- branchは`200m_main`だけを使用する。
- コードはローカルで修正・test・commit/pushし、Colabでpullする。
- GPU実行とDriveアクセスはColab経由だけで行う。
- 再起動時はclone、Python 3.12環境、CUDA依存関係を再構築する。
- 長い処理はforeground cellで実行し、約10分ごとに状態を確認する。
- 既存artifactを上書きしない。
