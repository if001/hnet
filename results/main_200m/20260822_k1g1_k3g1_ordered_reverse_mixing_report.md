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
採用しない。現計画の停止条件に従い、長時間候補はK1G1を維持する。ただし既存6構成を同じ112文で
比較すると、K1-firstはK1G1とK3G1の中間的な性質を実際に獲得しており、文節coverageは全候補中
同率最高だった。したがって混合の考え方自体を棄却するのではなく、seed要因を分離した小規模追試と、
K/G比率・配置を分離するcontrolを次の候補とする。

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

### 本実験のseedが変更するもの

現在の学習CLIの`--seed`はrun全体で共用され、学習開始時にPython、NumPy、PyTorch CPU、全CUDA
deviceの乱数を設定する。さらに同じ値をpacked training datasetのshuffleへ渡す。このためseed
42/43/44を変えると、少なくとも次の二つが同時に変化する。

1. モデルweightの初期値
2. 同一packed dataset内のtraining sample提示順序

データセットの内容、明示的なvalidation dataset、probe文は変わらない。validationはshuffleせず、
評価スクリプトの`--seed`は結果を識別するmetadataであって推論を乱数化しない。

したがって本レポートの3 seed比較が示すのは、**初期weightとデータ順序を合わせたrun全体の頑健性**
である。seed差をweight初期値だけ、またはデータ順序だけの効果として解釈してはならない。後述の
分離実験を行うまでは、seed 43での順位反転の原因も両者のどちらかに特定できない。

## 評価指標の定義と読み方

### 評価対象と境界の分類

本レポートは2-stage H-Netのうち、main networkへ渡されるchunkを決めるstage 1境界を評価する。
各probe文には注目範囲と、言語的に許容できる一つ以上の分割例を人手で付与している。複数の分割例に
含まれる境界位置の和集合を **acceptable boundary** とする。モデルが注目範囲内で選んだ境界のうち、
UTF-8 codepoint間にあるものを **selected evaluable boundary** とし、次のように分類する。

- **explained boundary:** selectedかつacceptableな境界。文節、活用語尾、助動詞、助詞、複合語境界、
  句読点など、probeの許容分割で言語的に説明できる位置である。
- **unexplained boundary:** selectedだがacceptableでない境界。注釈した言語的単位では説明できない位置で
  あり、少ない方がよい。ただし、注釈がすべての妥当な分割を網羅していない可能性があるため、直ちに
  誤りと断定するものではない。
- **lexeme fracture:** selected境界のうち、注釈で保護した語彙・形態素の内部を切る位置である。
  例えば一体として扱いたい複合語や語幹の内部を切る傾向を測り、少ない方がよい。
- UTF-8 codepoint内部の境界はconstraint-dependentとして言語指標の分母から除外する。本実験は
  `utf8-hard` で実行しており、UTF-8違反の有無をモデル選定の主指標にはしていない。

### Category precision / coverage / fracture

category 88文を、文節、助動詞、助詞、複合語、structuredなどのcategory別または全体で集計する。
表の `P/C/F` は次を表し、すべて0から1の範囲を取る。

| 略記 | 定義 | 高い／低い場合の意味 |
| --- | --- | --- |
| P: explainable boundary precision | explained境界数 ÷ selected evaluable境界数 | 高いほど、モデルが実際に置いた境界の多くを言語的に説明できる。低いほど説明困難な境界が多い。 |
| C: category coverage | explained境界数 ÷ acceptable境界候補数 | 高いほど、probeが期待する境界候補を広く拾う。低い場合は、境界を置かない、または別の位置へ置く傾向を表す。 |
| F: lexeme fracture rate | protected lexeme内部のselected境界数 ÷ selected evaluable境界数 | 低いほど語彙内部を壊しにくい。高いほど、計算単位として再利用しにくい断片を作る懸念が強い。 |

全体値は文ごとの率の単純平均ではなく、対象recordの境界数を合算して求めるmicro集計である。
文節P/Cも同じ定義を文節categoryのrecordだけへ適用する。

precisionだけが高くても、境界をほとんど選ばず少数の安全な位置だけを選んだ結果かもしれない。
逆にcoverageだけが高くても、過剰に境界を置いて多くのacceptable位置を偶然含んだ可能性がある。
したがって、PとCは必ず併記し、Fも含めてPareto比較する。単一の総合scoreには変換しない。

### Family precision / coverage / integrity

family 24文は、同じ語の活用違い、同じ表現の文脈違い、同種の複合語など、関連するrecordをfamilyとして
まとめたprobeである。単発の境界一致だけでなく、関連例にまたがって再利用可能な分割を行うかを見る。

| 指標 | 定義 | 高い／低い場合の意味 |
| --- | --- | --- |
| family precision | 各family内の explained ÷ selected をfamily間で平均 | 高いほど、関連例で選ぶ境界が言語的に説明しやすい。境界を一つも選ばずprecisionが未定義のfamilyは、本集計では0として扱う。 |
| family coverage | 各family内の explained ÷ acceptable をfamily間で平均 | 高いほど、活用・文脈・語彙family全体で期待境界を拾えている。低い場合は、有用な境界を選ばないfamilyが多い。 |
| family lexeme integrity | protected lexeme内部にfractureが一つもないrecordの割合をfamily間で平均 | 高いほど保護語彙を一体として保つ。低いほど語彙内部を切るrecordが多い。 |

integrityは、モデルが境界そのものをほとんど置かない場合にも高くなり得る。そのため「高integrityかつ
低coverage」は必ずしも良い分割ではなく、family precision/coverageおよびlandmarkと一緒に読む。

### Landmark coverage / consistency

familyには、例えば `分割|する` のように、関連record間で追跡したい名前付き境界をlandmarkとして
指定している。

- **landmark coverage（表のlandmark C）:** 指定landmarkを実際に選んだrecord数 ÷ そのlandmarkを
  持つrecord数。高いほど、期待する再利用可能な境界をfamily内で広く選ぶ。0は一度も選ばず、1は
  全recordで選ぶことを表す。
- **landmark consistency:** 「選ぶ／選ばない」の多数側のrecord割合。1に近いほど判断が揃い、0.5に
  近いほど揺れる。ただし、全recordで選ばない場合も1になるため、coverage=0かつconsistency=1は
  有用な一貫性ではない。本レポートの主要表ではcoverageを掲載し、consistencyはraw集計に保持する。

context-control familyについては、同じsurfaceで選んだ境界位置の完全な組合せが一致する割合を
**context signature consistency** として補助的に保存する。高いほど文脈が変わっても同じ分割だが、
動的chunkingでは文脈に応じた妥当な変化もあり得るため、高ければ常に良いとは判定しない。

### Dense trajectory指標

family 24文をstep 10, 20, ..., 220の22時点でnative評価し、単一checkpointの偶然ではなく、学習中に
どの領域へどれだけ滞在したかを測る。

| 表記 | 定義 | 高い／低い場合の意味 |
| --- | --- | --- |
| time P/C | checkpointごとにmicro集計したprecision/coverageの22時点平均 | 高いほど、学習期間を通して説明可能境界または期待境界を維持する。単点の良さより持続性を評価する。 |
| fracture occupancy | 各checkpoint・recordのうち、fractureを一つ以上含むものの割合 | 低いほど、学習中に語彙を壊す分割へ滞在しにくい。境界本数ではなく、問題を含むrecordの滞在率である。 |
| unexplained occupancy | 各checkpoint・recordのうち、unexplained境界を一つ以上含むものの割合 | 低いほど、説明困難な分割へ滞在しにくい。raw集計に保持する補助指標である。 |
| late P/C | 最後の2 checkpoint（step 210/220）をまとめたprecision/coverage | 学習終盤の状態を表す。高い方がよいが、2時点だけなのでtime P/Cと併記する。 |
| late fracture | 最後の2 checkpoint・recordにおけるfracture occupancy | 低いほど終盤に語彙内部を壊しにくい。 |
| transition | 隣接checkpoint間で、同じrecordのselected境界位置の完全な組が変わった比較数 ÷ 全比較数 | 高いほど分割が頻繁に変化し、低いほど安定する。ただし、不自然な分割へ固定されても低くなるため、良否を単独では決めない。 |

### Compression profile、native、seed集計

- **native:** モデル自身のrouter判定をそのまま使う。dense trajectoryはnativeで評価する。
- **low compression:** 平均2.5 unit/chunk相当の境界予算。比較的多く境界を置く。
- **central:** 平均3.0 unit/chunk相当。本レポートのstep別主要表に使用する。
- **high compression:** 平均3.5 unit/chunk相当。境界を少なくして大きいchunkを作る。

固定profileは学習済み境界確率の上位を指定予算まで選び、境界数の違いを統制して順位の頑健性を見る。
nativeとの違いは「境界確率の順位」と「モデル自身が選ぶ境界数」を分けて考えるためのものである。

表の `平均±標準偏差` はseed 42/43/44の3値に対する算術平均と母標準偏差である。標準偏差が小さい
ほどseed間の値は安定するが、平均値自体が良いことを保証しない。また3 seedだけなので、統計的な
有意差ではなく再現方向と効果の大きさを確認する記述統計として扱う。

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

## 既存main networkを含むstep 220比較

既存のT26、K1G1、K3G1、K3T1には、同じ112文probe、central、stage 1、step 220、seed
42/43/44の評価がある。次表はそれらと今回の2混合構成を同じ集計定義で並べた3 seed平均である。
ただし前節のとおり、各seed内で初期weightとデータ順序が同時に変わる制約は全構成に共通する。

| model | category P/C/F | family P/C | integrity | landmark C | 文節 P/C |
| --- | --- | --- | ---: | ---: | --- |
| T26 | .439/.239/.257 | .448/.332 | .806 | .417 | .301/.348 |
| K1G1 | .451/.271/.255 | **.708/.506** | .861 | **.611** | .168/.203 |
| K3G1 | **.490**/.261/**.218** | .339/.240 | .833 | .306 | **.321**/.333 |
| K3T1 | .453/.254/.265 | .619/.380 | .903 | .458 | .295/.319 |
| K1-first | .471/.264/.262 | .560/.379 | .903 | .431 | .294/**.348** |
| K3-first | .441/.256/.244 | .646/.490 | **.917** | .583 | .182/.232 |

### 良くなった点

- **K1-first:** K1G1に対して文節P/Cが`.168/.203`から`.294/.348`へ改善した。K3G1と比べても
  文節precisionはわずかに低いがcoverageは高い。category precisionもK1G1より高く、K1G1とK3G1の
  中間化は部分的に成功した。
- **K3-first:** K3G1に対してfamily P/Cが`.339/.240`から`.646/.490`へ大きく改善し、K1G1の
  `.708/.506`に近づいた。integrityは6構成中最高、landmark coverageもK1G1に次ぐ。
- 両混合とも、親の片方にしかなかった特徴を一部取り込んだため、main-network layoutがouter-stageの
  境界学習へ影響するという探索仮説には情報価値があった。

### 悪化した点と未達点

- K1-firstはK1G1よりfamily P/Cとlandmarkが低く、fractureもわずかに悪い。文節改善と引き換えに
  K1G1の再利用可能境界の強みを十分維持できなかった。
- K3-firstはK3G1よりcategory precisionと文節が低く、K1G1よりcategory coverageと文節が低い。
  familyの改善はseed 44への依存が大きく、安定した総合改善ではない。
- T26はcategory/family主要軸で新しい最大値を持たないが、文節coverageはK1-firstと同率最高である。
  Transformer baselineとして残す価値はあるものの、今回の境界proxyだけで本線へ戻す根拠は弱い。
- K3T1はintegrityとfamily precisionで比較的強いが、後述のdense終盤でfamily性能が維持されなかった。

この表ではK1-firstにも独自の優位軸があるため、厳密な意味で全候補に支配されているわけではない。
一方、「K1G1のfamilyを大きく落とさずK3G1の文節へ近づく」という事前の成功条件は満たしていない。
したがって、長時間本線には上げないが、要因分離後の小規模な構成改善候補には残す、という判断になる。

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

### 既存構成とのdense比較

同じ10-step間隔、native、family 24文、step 10--220で直接比較できる既存runはseed 42のK1G1、
K3G1、K3T1である。T26には4 checkpointの疎trajectoryはあるが、同条件の22時点dense runはないため
次表へ混ぜない。

| model（seed 42） | time P/C | fracture | late P/C | late fracture | transition |
| --- | --- | ---: | --- | ---: | ---: |
| K1G1 | .598/**.675** | **.098** | **.636/.636** | **.083** | .341 |
| K3G1 | .478/.492 | .335 | .412/.424 | .208 | .349 |
| K3T1 | .508/.417 | .311 | .233/.152 | .396 | .375 |
| K1-first | .559/.428 | .426 | .500/.424 | .542 | **.220** |
| K3-first | **.627**/.445 | .326 | .622/.348 | .292 | .226 |

K3-firstはtime precisionで最大だが、K1G1よりcoverageが大幅に低く、fractureも高い。K1-firstは
K1G1より全主要dense品質軸で悪く、低transitionも不自然な分割へ固定された可能性を除外できない。
したがってdenseでは、混合が既存anchor K1G1を総合的に改善したとはいえない。

T26と完全に比較するには、同じrunner、probe、LR horizon、compression設定でseed 42の220-step
10-step dense runを追加する必要がある。既存の4 checkpoint疎trajectoryを22時点平均と同じ値として
扱うことはできない。

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

## 次に必要な要因分離と構成改善

### Weight初期値とデータ順序のseed分離

学習CLIを少なくとも`model_init_seed`、`data_order_seed`、`train_runtime_seed`へ分ける必要がある。
`train_runtime_seed`はモデル初期化後に再設定し、dropout等の学習時乱数が初期化seedへ連動しないように
する。最小のone-factor-at-a-time設計は次の5条件である。

| 条件 | model init | data order | runtime | 評価する差 |
| --- | ---: | ---: | ---: | --- |
| baseline | 42 | 42 | 42 | 基準 |
| init-43 | 43 | 42 | 42 | 初期weightだけの効果 |
| init-44 | 44 | 42 | 42 | 初期weightだけの効果 |
| data-43 | 42 | 43 | 42 | データ順序だけの効果 |
| data-44 | 42 | 44 | 42 | データ順序だけの効果 |

データ順序だけを変える比較では、同一architectureのstep 0 state dictを保存して全runへロードする方が、
同じseedから再生成するより厳密である。各runには初期checkpoint hash、shuffle index hash、先頭N sample
IDをmanifestへ保存する。最小設計で差が見えた構成だけ、3 init × 3 dataのcrossed designへ進めれば、
初期値・順序・両者のinteractionを推定できる。

architecture間ではparameterの意味と並びが異なるため、異なる構成へ完全に同じweightをコピーすることは
できない。比較は同じdata-order/runtime条件を対応させたpaired runとし、各architecture内で上記の
分散要因を分ける。

### 次のmain-network構成

今回の2構成は順序だけでなく、親K1G1/K3G1に対してK/G比率もK=16/G=10へ変えている。次は次の順で
探索すると、何が改善を生んだかを切り分けやすい。

1. **K=16/G=10のinterleaved control:** 同じ成分数を全26層へ均等に分散し、連続KDA blockの位置と
   中間K/G比率を分離する。
2. **K1G1寄りの小さい変更:** K=14/G=12またはK=15/G=11とし、K1-firstで見えた文節改善を狙いつつ、
   K1G1のfamily/landmark低下を抑える。連続KDA区間はまず1--2個に限定する。
3. **block位置control:** 同じK/G数で連続KDA区間を前半・中盤・後半へ置き、文節とlandmarkのどちらが
   位置に追随するかを見る。
4. 55-step screeningでも10-step denseを残し、step 55の良さだけで選ばない。要因分離したseed条件で
   K1G1のfamily/低fractureを維持し、K3G1相当の文節へ近づいた構成だけを220 stepへ延長する。

現時点の優先順位は、長時間anchorがK1G1、構成改善の第一候補が「K1G1寄りで後半の連続KDA量を
減らしたK1-first」、因果切り分け用controlがK=16/G=10 interleavedである。K3G1、K3T1、T26は
それぞれ文節、integrity、Transformer baselineの比較対象として保持する。

## 仮説判定

1. **親の強みの両立:** 不成立。混合はK1G1より文節を改善した時点があるが、K1G1のcategory、
   family、低fractureを一貫して維持しなかった。
2. **層位置の寄与:** 限定的な手がかりあり。K1-firstの文節、K3-firstの初期landmarkには再現性が
   あるが、denseとfamilyの方向反転により一般的な層位置効果とは結論しない。
3. **位置よりK/G比率:** ordered/reverseが完全に同じではないため位置を無視できない。ただし本実験は
   K/G比率だけの因果効果も証明しない。
4. **長時間候補:** 現混合2構成は見送り、K1G1を維持する。K1-firstは長時間候補ではなく、seedと
   K/G比率を分離した次の小規模構成探索の出発点として保持する。

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
