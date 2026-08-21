# main network追加中規模評価計画

更新日: 2026-08-21

## 1. 目的と前提

小・中規模学習だけで、分割がmain networkの長期学習に有利かを直接証明することはできない。
本計画では、次の仮説に基づいて長期学習候補へ当たりを付ける。

> 明らかに不自然な分割が少なく、文節、活用、助動詞、助詞、複合語、句読点などの
> 言語的に説明可能で再利用可能な位置に境界を置く構成は、main networkが規則性を
> 学習しやすい可能性が高い。

H-Netのchunkは離散tokenではないため、`分|割する`を「割するという未知token」とは解釈しない。
評価対象は、main networkへ表現を渡す位置と計算間隔が、語形・語彙・文脈をまたいで説明可能な
構造に揃うかである。

同じcheckpoint・同じprefixの推論は決定的だが、学習中はparameter更新により境界が移動する。
最終stepだけでなく、許容分割への滞在と説明困難な分割への滞在を時間方向に評価する。

## 2. 判定指標

### 2.1 維持する既存指標

- explainable boundary precision / category coverage
- best acceptable segmentation precision / recall / F1
- unexplained boundary rate
- lexeme fracture count / rate
- short / severe short fragmentation
- category別結果とblind gallery
- native圧縮率、固定境界予算に対する頑健性
- seed間の範囲と順位安定性

UTF-8途中境界率はbyte-boundary-constraint依存なので順位に使わない。速度もfused実装条件が
揃っていないため順位に使わない。validation BPBは重大退行の検出だけに使い、220 step時点の
小さなloss差から境界の長期的な学習容易性を判定しない。

### 2.2 追加するfamily指標

既存88文は各recordを独立に採点する。これに加え、同じ語幹・語彙・表現を含む4文程度を
一つのfamilyとして注釈する。

- **landmark coverage**: `分割|する／した／している／される`のように、familyで共有する
  説明可能なランドマークへ境界が置かれた割合。
- **landmark consistency**: 同名ランドマークの選択状態がfamily内で一致する割合。
  全例で境界なしでも高くなるため、coverageと必ず併記する。
- **family lexeme integrity**: family内で保護語彙を破壊しなかったrecordの割合。
- **context signature consistency**: 同じsurfaceを異なるcontrol文脈に置いたとき、許容境界内で
  分割signatureが維持された割合。
- **family explainability**: family内のprecision、coverage、unexplained、fractureをmicro/macroで
  併記する。

単一の総合点は作らない。landmark coverage、lexeme integrity、context consistency、既存category
指標のParetoとgalleryで比較する。

### 2.3 学習中trajectory指標

- acceptable occupancy: 許容offsetが観測期間内に選択された比率
- unexplained / fracture occupancy: 説明困難境界・語彙内切断への滞在率
- segmentation transition: pattern間の遷移回数、滞在時間、再出現
- late-window mean / variance: 最終1点でなく終盤区間の平均と分散
- time-averaged precision / coverage
- boundary probability marginは補助として保存する

分割の変化自体を悪いとしない。許容分割間の移動、不自然な分割への移動、安定した誤りを区別する。

## 3. 実験手順

### Phase M1: family評価器

1. 24--32文、6--8 familyの固定probeを作る。
2. 活用family、サ変語幹family、複合語family、同一surfaceのcontrol文脈を含める。
3. familyごとに共通landmarkとprotected substringを人手注釈する。
4. CPU unit testでUTF-8 byte offset、landmark集計、全境界なしの見かけ上一致を検証する。
5. 既存のK1G1、K3G1、K3T1、T26を同じprobeで評価する。

既存checkpointはseed 42/43/44、step 55/110/165/220を優先して再利用する。不足artifactだけを
再実行し、raw JSONと集約をDriveへ保存する。

### Phase M2: trajectory再集計

ここでいうcategory評価は、既存の11 category x 8文の88文probeを指す。family probe内の
`inflection`、`compound`、`context_control`だけをcategory評価の代用にしない。

まずK1G1、K3G1、K3T1、T26の全4候補について、88文category probeと24文family probeを統合した
112文を、seed 42/43/44、step 55/110/165/220、low/central/high/nativeで評価する。既存rawを再利用する
場合は、checkpoint、学習設定、compression target、constraint、profile定義が一致することをmetadataで
確認する。一致しない組合せだけ再評価する。

112文のcategory precision/coverage/fractureとfamily landmark coverage/integrity/context consistencyを
単一スコアにせずPareto比較し、その後にdense対象を1--3構成へ絞る。family指標だけでK3G1やT26を
除外しない。候補差が残る場合、選定構成をseed 42で再学習し、次の二層で観測する。

- dense core: family 24文を10 stepごとにnativeで評価
- full probe: 統合112文を55 stepごとにlow/central/high/nativeで評価

学習中の実際の分割変化を見るdense coreはnativeを使い、構成間で境界予算を揃えるcentralは
55 step checkpointで使う。高頻度に強制profileを全件実行する重複を避ける。

時間軸はcumulative raw bytesを優先する。高頻度評価はtraining中のdeterministic hookで行い、
training/eval modeと乱数状態を保存・復元する。評価挿入が学習trajectoryを変えないことをtestする。

seed 42で候補差が見えなければdense再学習を増やさない。差があるがseed依存を否定できない場合だけ
seed 43/44を追加する。既に実施したK1G1/K3T1のdense結果は保持するが、112文の4候補比較より先に
行った暫定結果として扱い、dense対象選定の根拠には使用しない。

### Phase M3: 構成比較

既存のK1G1、K3G1、K3T1、T26から、次を満たす1--2構成をanchorとして残す。

1. severe fragmentationとlexeme fractureが少ない。
2. 複数categoryでprecision/coverageがPareto劣位でない。
3. family landmark coverageとlexeme integrityが両立する。
4. late-windowでunexplained/fracture occupancyが高くない。
5. 複数seedで方向が再現し、BPBに重大退行がない。

modelごとの相補的category差がfamily・trajectory評価後にも再現した場合だけ、混合main networkを
追加する。最初は同一成分数で配置順だけが異なるordered/reverseの2構成をseed 42、55 stepで比較し、
差が明瞭な場合だけ220 stepと追加seedへ進める。差が小さければ組合せ探索を打ち切り、anchorの
長期比較へ進む。

## 4. 今回実施しない評価

以下は有用性を否定するものではないが、現在の候補選定には過剰または仮説が未成熟なので外す。

- 強制境界間の微小loss差を主指標にする反実仮想評価
- category別byte lossから220 step時点の分割優劣を推定する評価
- clean-first curriculumとouter learning-rate taper
- Transformer一般の層別機能を確認するlinear probe
- early/middle/late groupへの推論時scale intervention
- 長文距離別の大規模probe
- ordered/reverse/interleavedの3構成一括探索と多数の局所置換

これらは長期anchor決定後、必要な仮説が具体化した場合に別計画とする。

## 5. 成果物

| artifact | 保存先 |
| --- | --- |
| family probe | `configs/linguistic_boundary_family_probe_v1.json` |
| family集計実装 | `hnet/training/linguistic_boundary_families.py` |
| trajectory集計実装 | `hnet/training/linguistic_boundary_trajectory.py` |
| dense training runner | `scripts/run_dense_linguistic_training.py` |
| 評価・集約CLI | `scripts/evaluate_linguistic_boundaries.py`, `scripts/summarize_linguistic_boundary_screening.py` |
| unit test | `tests/unit/test_linguistic_boundary_families.py` |
| raw評価 | Drive `evals/linguistic_boundary_family_v1/` |
| 集約 | Drive `reports/linguistic_boundary_selection/family_v1/` |
| 中間memo | `plan/mid/` |
| 最終report | `results/main_200m/` |

## 6. 実行規則と停止条件

- branchは`200m_main`だけを使う。
- コード変更はローカルで実装・test・commit・pushし、Colabでpullする。
- GPU実行とDriveアクセスはColab経由で行う。
- Colab初期化時はclone、Drive mount、Prepare、GPU/import smokeをやり直す。
- 長時間処理はforeground cellで実行する。
- checkpoint、dataset、raw結果はGitへcommitしない。
- family annotationが人手でも一意に説明できない場合、実行前にprobeを修正する。
- 差が1--2境界または1 familyだけなら構成差と断定しない。
- seed順位が反転する場合は1候補へ確定せず、長期候補を複数残す。
