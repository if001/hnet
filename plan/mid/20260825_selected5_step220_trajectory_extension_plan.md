# 選択5構成 step 220 境界形成軌跡延長実験計画

作成日: 2026-08-25

## 0. 対象名の確認

依頼に記載された`K15-middle`に対応する既存config、runner定義、過去artifactは存在しない。一方、直前までの
比較対象には`K14-middle`（K14G12、単一K3区間を中盤に配置）がある。本計画では文脈上の表記揺れと判断し、
`K15-middle`を`K14-middle`として扱う。

新規のK15-middle構成を意図していた場合は、K/G総数と26層内の配置を定義するまで実行を開始しない。

## 1. 背景

55 stepのfull112実験では、境界指標が単調に形成されず、構成ごとに形成順序が異なる可能性が観測された。

- 同一step paired比較は、同じraw batch列、累積入力byte数、optimizer update数における品質を比較する。
- ただし同じstepでも、main networkが処理する実効chunk数や境界形成の成熟度は構成間で同じとは限らない。
- step 10--50平均は初期の未形成状態を含み、step 30--50のlate windowでは候補順位が変わった。
- 55 stepだけでは、一時的なpeak、形成順序、持続的なarchitecture bias、長期の揺れを分離できない。

そこで候補を5構成へ絞り、既存のexact step 55 stateからstep 220まで延長する。目的は生成品質や最終的な
main-network性能の証明ではなく、境界品質の形成・崩壊・回復・安定化の軌跡を観測することである。

## 2. 対象構成と現在わかっている特徴

### K1G1: anchor

- step 10ではcategory、family、文節の評価対象境界を選択していない。
- 文節coverageはstep 30で`.435`へ上昇した後、step 50で`.087`へ低下した。
- family P/C、landmarkは後半に伸び、step 50で`.719/.697/.792`だった。
- late integrityは`.944`で5候補中の保全基準になる。
- category Pと文節P/Cが低下する一方でfamilyが後から形成される軌跡が観測された。

### K15-split: 文節先行・late総合候補

- step 20で文節P/Cが`.393/.478`へ形成されたが、family P/Cとlandmarkは0だった。
- step 30でfamily P/Cが`.739/.515`へ形成された。
- step 40で文節Cが`.783`へ上昇する一方、integrityが`.708`へ一時低下した。
- step 50ではfamily Pとintegrityが`.864/.958`へ回復した。
- lateではK1G1比でcategory F `-.148`、文節P/C `+.243/+.334`、family C `-.061`、
  landmark差`.000`、integrity `-.069`だった。
- 「familyを学ばないspecialist」ではなく、文節が先行してfamilyが後から形成される候補である可能性がある。

### K14-middle: 早期多軸形成・均衡候補

- step 20でcategory、family、文節の全軸が形成された。
- step 30でcategory Cとfamily Cが上がる一方、category F `.420`、integrity `.667`へ悪化した。
- step 50ではcategory F `.125`、family C/landmark/integrity `.909/.875/.958`へ回復した。
- 文節P/Cはstep 20の`.387/.522`をpeakとしてstep 50の`.097/.130`へ低下した。
- 5時点平均ではcategory P/C/F、family C、landmarkのバランスがよいが、late integrity平均はK1G1より低い。
- fracture・integrityの悪化が一時的な調整か、以後も反復するかが未確定である。

### K14-late: coverage指向候補

- late category P/C/Fは`.428/.338/.367`で、category Cは候補中でも高い。
- late family P/C、landmarkは`.517/.747/.861`でcoverage側が強い。
- late 文節P/Cは`.211/.319`だった。
- 一方、late integrityは`.681`でK1G1 `.944`より低く、category fractureもK1G1より高い。
- 高coverageを維持したままfracture・integrityが回復するか、構造的trade-offとして残るかを確認する。

### K3-first: precision指向候補

- late category P/C/Fは`.463/.298/.299`でcategory Pが高い。
- late family P/C、landmarkは`.712/.616/.847`、文節P/Cは`.261/.319`だった。
- family Pと文節precisionが強い一方、late integrityは`.653`でK1G1より大きく低い。
- precision優位を維持しながらintegrityが後から回復するかを確認する。

## 3. 目的と仮説

### 主目的

1. step 60--220で9指標がどの順序で形成・崩壊・回復するかを構成別に記録する。
2. 55 step以前の特徴が一時的な位相差か、持続するarchitecture biasかを区別する。
3. K1G1に対するfracture・integrity・family coverageの許容退行制約を満たす候補を確認する。
4. 同一step比較とcoverageを横軸としたmatched-state診断を分けて行う。

### 構成別仮説

- H1 K15-split: 文節先行後にfamilyが追いつき、文節優位をある程度維持する。
- H2 K14-middle: fracture・integrityの悪化と回復を反復するか、後半に安定する。
- H3 K14-late: coverage優位は残るが、fracture・integrity trade-offも持続する。
- H4 K3-first: precision優位は残るが、integrity低下も持続する。
- H5 K1G1: family・integrityを維持する一方、文節coverageは低い状態へ落ち着く。

これらは検証対象であり、計画時点の結論ではない。

## 4. 固定条件と継続方法

- branch: `200m_main`
- source run: full112 temporal Phase 1の各step 55 run
- resume: exact `checkpoint_step_000055.pt`からoptimizer、scheduler、step、data cursorを含めて継続
- `model_init_seed=42`
- `data_order_seed=42`
- `train_runtime_seed=42`
- max step: 220
- LR horizon: 220（既存runと同一）
- dataset、sequence length、batch、gradient accumulation、optimizer、LR、ratio lossを既存runと同一
- compression target、`utf8-hard`を各構成の既存条件と同一
- full112 v1の文章・annotationを変更しない
- stage 1を主判定、stage 0を診断用とする
- 速度は候補順位に使用しない

再開前に次を監査する。

1. source runがstep 55で正常終了し、5時点のfull112 rawを持つ。
2. seed三因子、probe hash、dataset manifest hash、model config hashが既存監査と一致する。
3. checkpointにoptimizer・scheduler・data stateがあり、resume後の最初のstepが56になる。
4. source artifactを変更せず、新しいarchiveへ保存する。

## 5. 評価時点と取得指標

### 5.1 Native時間軌跡

既存step 10/20/30/40/50に続き、step 60から220まで10 step間隔でfull112 nativeを評価する。最終的に
22時点を一つの軌跡として扱う。

各時点でstage 1について次の9指標を主表へ保存する。

1. category precision
2. category coverage
3. category fracture record occupancy
4. family precision
5. family coverage
6. landmark coverage
7. family integrity
8. 文節precision
9. 文節coverage

precision未定義と`.000`を区別する。解釈補助としてselected/explained/acceptable境界数、stage 0指標、
native compression、累積入力byte数、recordごとの境界signatureも保存する。

### 5.2 Milestone profile

step 110、165、220 checkpointでlow/central/high/nativeのfull112評価を行う。native軌跡と固定budget profileを
混ぜて平均せず、境界数を変えたときにも特徴が残るかを別表で確認する。

## 6. 時間方向の集計

### 6.1 構成別step表

5構成それぞれについて、stepを行、9指標を列とする表を作る。平均だけでなく、どの時点で各指標が形成・
崩壊・回復したかを主成果とする。

### 6.2 Window

- initial: step 10--50（既存結果）
- medium: step 60--110
- late: step 120--170
- terminal: step 180--220

各windowでmean、median、precision/coverageのq20、fractureのq80、MAD、最小・最大を出す。window名は
長期収束を意味せず、本実験内の時間区分である。

### 6.3 同一step paired比較

各候補とK1G1を同じstepで比較する。これは同じraw学習量に対するfixed-training-dose比較であり、同じ境界
成熟度を仮定しない。median delta、改善step割合、許容退行違反率、terminal 5時点の方向一致を出す。

### 6.4 形成・安定化診断

- coverage閾値`.10/.20/.30`への初回到達step
- 同じ閾値を3評価時点連続で満たすstable-attainment step
- terminal 5時点の傾き、MAD、最大振幅
- record単位のsegmentation transition rate

transition rateは順位に含めず、揺れの診断にだけ使用する。220 stepでも収束したとは断定しない。

## 7. 許容退行制約と順位

平均順位の前に、K1G1との同一step差に対する制約を適用する。結果を見て単一閾値を選ばず、感度分析する。

### Primary / relaxed threshold

| 指標 | primary | relaxed |
| --- | ---: | ---: |
| category fracture occupancy | K1G1 `+.05`以内 | `+.10`以内 |
| family integrity | K1G1 `-.05`以内 | `-.10`以内 |
| family coverage | K1G1 `-.05`以内 | `-.10`以内 |
| landmark coverage | K1G1 `-.05`以内 | `-.10`以内 |

各thresholdについて全stepの違反占有率とterminal 5時点の違反占有率を出す。一時的な1回の違反と継続的な
違反を区別する。制約外の重大退行を他指標の順位で相殺しない。

制約を満たした候補間では、9指標を原子的に分離し、未丸め値でaverage tie rankを使用する。transitionは
順位から除外する。順位平均・中央値は探索的要約であり、最終判断はPareto関係と実値差で行う。

## 8. Coverage軌跡とmatched-state診断

step番号をラベルし、矢印で時間順を示す。

1. category C × category P
2. category C × category fracture occupancy
3. category C × family integrity
4. family C × family integrity
5. 文節 C × family C

coverageは単調とは限らず、同じcoverageへ複数回到達し得る。軌跡を単一関数として扱わない。同じcoverage
付近（初期許容幅`.03`）の観測点だけを比較し、複数点は保持する。観測範囲外への外挿は行わない。許容幅
`.02/.05`でも結論が変わるかを感度分析する。

matched-state比較は同一step比較を置き換えない。前者は境界状態を近づけた品質差、後者は同じ学習量での
到達品質を答えるため、二つの結論を別々に示す。

## 9. 段階と停止条件

### Phase 0: resume smoke

各構成についてstep 55 sourceを監査し、まずK1G1のsourceを隔離コピーして、`train.py`を直接使うstep 56までの
resume smokeを行う。step、data cursor、LR、optimizer stateが連続することを確認してから本実行へ進む。
smoke artifactは本runと分離し、source runと本番archiveを変更しない。

### Phase 1: 5構成をstep 220へ延長

1構成ずつforegroundで実行し、各run終了時にDriveへarchive・status・hash auditを保存する。中断時は完了済み
構成を再実行せず、未完了構成だけを再開する。

### Phase 2: 集計と候補判断

step表、window、paired、制約違反率、coverage軌跡、milestone profileを作成する。本実験ではseedを増やさない。
terminalでもcombined候補が残った場合だけ、別計画で`model_init_seed`と`data_order_seed`を分離する。

次の場合、当該候補を長時間候補へ昇格しない。

- terminal 5時点でprimary/relaxedの双方に継続的に違反する。
- 改善が単一peakだけで、terminal windowに残らない。
- 文節改善とfamily/integrity維持が異なる時点にしか成立しない。
- matched-stateでprecisionまたはfractureの優位が消え、同一step差が形成速度だけで説明できる。

## 10. 実行・保存規則

- コード修正はローカルで行い、test、commit、`200m_main`へpush後にColabでpullする。
- GPU実行とGoogle DriveアクセスはColab経由だけで行う。
- Colab再起動時はclone、Python 3.12環境、CUDA依存関係を再構築する。
- 長時間処理はforeground cellで実行し、モデル開始・終了をDrive status JSONへ逐次保存する。
- 既存run、checkpoint、raw evaluationを上書きしない。
- 一時memoは`plan/mid/`、最終reportは`results/main_200m/`へ保存する。

| artifact | 保存先 |
| --- | --- |
| 本計画 | `plan/mid/20260825_selected5_step220_trajectory_extension_plan.md` |
| extended runs | Drive `runs/full112_step220_selected5/` |
| status/audit | Drive `manifests/full112_step220_selected5/` |
| raw/analysis/plots | Drive `reports/full112_step220_selected5/` |
| milestone profiles | Drive `evals/full112_step220_selected5/` |
| final report | `results/main_200m/` |
