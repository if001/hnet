# K1G1/K3G1 ordered-reverse混合main network実験計画

作成日: 2026-08-21

## 1. 背景

112文のcategory・family統合評価と10-step dense評価により、K1G1とK3G1に次の相補性が
観測された。

- K1G1: family landmark、助動詞、助詞、複合語、structured、低fracture、境界予算への頑健性
- K3G1: 文節precision/coverage、category precision、native category

K1G1はmain networkの26層がKDA/Gated MLAの交互配置であり、K3G1はKDAを3層続けてから
Gated MLAを配置する。ただし、現在の結果は配置頻度と境界特性の関連を示しただけで、
層位置やK/G比率の因果効果は未確定である。

## 2. 目的

K1G1とK3G1の相補的な境界特性を一つのmain networkで両立できるかを小規模に検証する。
併せて、KDA/Gated MLAの成分数を同一にしたordered/reverse比較により、強みの差に層位置が
寄与するかを切り分ける。

本実験では、長期学習性能や層ごとの表現機構の証明は行わない。言語的に説明可能な境界を
proxyとし、長時間実験に進める構成へ当たりを付ける。

## 3. 検証仮説

1. K1G1型の交互区間とK3G1型の連続KDA区間を混合することで、K1G1のfamily頑健性を大きく
   損なわずに文節指標を改善できる可能性がある。
2. K1-firstとK3-firstに再現性のある差があれば、K/G総数ではなく、連続KDA区間を前半または
   後半に置く層位置が境界特性に寄与している。
3. 両混合構成が似た結果になれば、位置よりK/G比率の寄与が大きい可能性がある。両者とも
   親構成の強みを失う場合は、強みが局所区間の単純な組合せでなく、全層の反復パターンに
   依存する可能性を残す。

## 4. 比較構成

親構成はK1G1がK=13/G=13、K3G1がK=19/G=7である。混合構成はその中間となるK=16/G=10、
合計26層に固定する。2構成間はモジュール種別ごとの層数、パラメータ数、外側構成を同一にし、
main-network layerの順序だけを変える。

| ID | main-network layout | 意味 |
| --- | --- | --- |
| K1-first | `K1G1` x 7 -> `K3G1` x 3 | 前半が交互配置、後半が連続KDA |
| K3-first | `K3G1` x 3 -> `K1G1` x 7 | 前半が連続KDA、後半が交互配置 |

実装する正確なlayout文字列は次とする。

- K1-first: `K1G1K1G1K1G1K1G1K1G1K1G1K1G1K3G1K3G1K3G1`
- K3-first: `K3G1K3G1K3G1K1G1K1G1K1G1K1G1K1G1K1G1K1G1`

新構成間の比較に加え、既存のK1G1/K3G1の同条件artifactを親controlとして使う。K3T1とT26は
最初の混合実験には含めない。

## 5. 統制条件

- 同じraw-byte dataset、データ順序、batch設定、学習step数、optimizer、learning-rate schedule
- 同じencoder/decoder構成、outer-stage学習率、compression target、byte-boundary constraint
- 同じevaluation mode、probe version、low/central/high/native profile定義
- Phase 1はseed 42、追試はseed 43/44
- モデルパラメータ数、実際の層展開、config hashを学習前にmanifestで照合する
- 評価hookの前後でtraining/eval modeと乱数状態を復元し、評価挿入が学習軌跡を変えないことを確認する

## 6. 実験手順

### Phase 0: 実装とCPU検証

1. 2構成のconfigを追加する。
2. layoutの展開結果が26層、K=16/G=10であることをunit testまたはinspection CLIで確認する。
3. パラメータ数が2構成で一致すること、既存configの読み込みを壊していないことを確認する。
4. ローカルでcommit/push後、Colabでbranch `200m_main`をpullする。

### Phase 1: 55-step screening

1. 2構成をseed 42、55 stepまで同条件で学習する。
2. family 24文を10 stepごとにnative評価し、occupancy、transition、fracture、late-windowを集計する。
3. step 55でcategory 88文 + family 24文の112文をlow/central/high/nativeで評価する。
4. 同じstep/data exposureのK1G1/K3G1親controlとPareto比較する。

### Phase 2: 220-step確認

Phase 1で後述の継続条件を満たす構成だけを220 stepまで延長する。family denseは10 stepごと、
112文full probeはstep 55/110/165/220でlow/central/high/nativeを評価する。K1-first/K3-firstのどちらも
層位置仮説の判定に必要な場合は両方を延長する。

### Phase 3: seed再現性

Phase 2で意味のあるPareto改善または配置順差が残った場合だけ、seed 43/44を追加する。
seed間で方向が反転する場合は層位置効果と結論しない。

## 7. 評価と判定

単一のaggregate scoreは作らず、次の軸をPareto比較する。

- category precision / coverage / fracture、特に文節、助動詞、助詞、複合語、structured
- family precision / coverage、landmark coverage/consistency、lexeme integrity、context consistency
- time-averaged precision/coverage、unexplained/fracture occupancy、late-window mean/variance、transition
- nativeと固定profileの差、実際のcompression ratio
- validation BPBは重大な退行の検出だけに使う

Phase 1の継続条件は次の通りとする。

1. K1G1のfamily landmark/coverage、助動詞、助詞、structured、fractureに一貫した大きな退行がない。
2. 文節precision/coverageの少なくとも一方がK1G1より改善するか、K3G1の強みに近づく。
3. 改善が1--2境界、1 family、または一つのcompression profileだけに依存しない。
4. dense trajectoryで終盤のfracture/unexplained occupancyが急増していない。

K1-firstとK3-firstの差は、同じK/G数で再現したときだけ層位置の手がかりとする。両者が
似ているだけでK/G比率が原因と確定せず、位置効果を検出できなかったと判定する。

## 8. 停止条件と次の判断

- 両混合構成がK1G1のfamily/category/fractureを壊し、文節も改善しなければ混合探索を停止する。
- ordered/reverse差が小さく、親構成に対するPareto改善もなければK1G1を長期本線とする。
- 一方の混合構成がK1G1の強みを維持して文節を改善し、seed間で方向が再現した場合は、
  K1G1とともに長時間実験の候補とする。
- 配置順の差はあるが理由が不明な場合、同じK=16/G=10のinterleaved controlは別計画とし、本実験に
  無条件で追加しない。

## 9. 成果物と実行規則

| artifact | 保存先 |
| --- | --- |
| model configs | `configs/` |
| CPU test / inspection | `tests/unit/`, `scripts/` |
| raw checkpoints/results | Drive `/content/drive/MyDrive/hnet_agent_200m_main/` 以下 |
| 中間memo | `plan/mid/` |
| 最終report | `results/main_200m/` |

- branchは`200m_main`だけを使う。
- コードはローカルで修正・確認・commit・pushし、Colabでpullする。
- GPU学習とGoogle DriveアクセスはColab経由で行う。
- Colabランタイム初期化時はclone、Drive mount、Prepare、GPU/import smoke testを再実行する。
- checkpoint、データ、raw実験結果はGitにcommitしない。
