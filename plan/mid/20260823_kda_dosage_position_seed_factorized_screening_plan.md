# KDA量・配置位置／seed要因分離 main-network screening計画

作成日: 2026-08-23

## 1. 背景

K1G1、K3G1、K1-first、K3-firstの112文・dense評価から、次が分かった。

- K1G1はfamily precision/coverage、landmark、低fracture、dense coverageに強い。
- K3G1はcategory precisionと文節precision/coverageに強い。
- K1-first（K=16/G=10、後半にK3区間）はK1G1より文節を改善したが、family/landmarkを失った。
- K3-first（K=16/G=10、前半にK3区間）はK3G1よりfamily/landmarkを改善したが、文節を失った。
- ordered/reverse比較はK/G数を同一にしたが、親K1G1に対してはKDA量とblock配置を同時に
  変更している。このため、改善・退行がKDA追加量、block位置、block集中のどれによるか未分離である。
- 従来の`seed`はweight初期値、training data順序、学習時乱数に共用されていた。3 seed結果はrun全体の
  頑健性を表すが、初期weightとデータ順序の効果を分離できない。

本計画ではデータセット内容とcurriculumの変更を保留し、K1G1からの小さいKDA追加とseed要因分離を
優先する。

## 2. 目的

1. K1G1のfamily/landmark/低fractureを大きく損なわず、文節precision/coverageを改善する構成を探す。
2. KDA追加量と、連続KDA区間の前半・中盤・後半位置を分離する。
3. 同じK/G数を均等配置したcontrolにより、K/G比率とblock集中の効果を分離する。
4. weight初期値とdata順序のseedを分け、候補差がどちらに依存するかを段階的に確認する。
5. 探索段階では全候補の大規模seed gridを避け、候補を絞るほど再現性確認を厚くする。

本実験で、長期main-network性能や下流生成品質の優位は証明しない。言語的に説明可能な境界proxyに
基づき、次の長時間候補へ当たりを付ける。

## 3. 固定条件

- packed training dataset、validation dataset、データ内容、outer-stage構成を固定する。
- batch、sequence length、optimizer、LR、LR horizon=220、compression target、ratio loss、
  `utf8-hard`を既存dense実験と一致させる。
- category 88文 + family 24文の112文probe v1を変更しない。
- family 24文のnative denseを10 step間隔で取得する。
- full112はlow/central/high/native、stage 0/1を保存し、主判定はstage 1とする。
- validation BPBは重大な退行検出にのみ使用し、速度は順位に使わない。
- データセットやclean-prefix curriculumの比較は本計画へ混ぜない。

## 4. 候補構成

K1G1はK=13/G=13である。内部の`G`を一つ`K`へ置換すると、総26層を保ったまま一つの`K3`
区間を作り、K=14/G=12となる。同じ考え方でKDA量を段階化する。

### 4.1 新規候補

| ID | K/G | 正確なmain-network layout | 分離する要因 |
| --- | --- | --- | --- |
| K14-front | 14/12 | `K3G1` + `K1G1` x 11 | 単一K3区間を前半に配置 |
| K14-middle | 14/12 | `K1G1` x 5 + `K3G1` + `K1G1` x 6 | 単一K3区間を中盤に配置 |
| K14-late | 14/12 | `K1G1` x 11 + `K3G1` | 単一K3区間を後半に配置 |
| K15-split | 15/11 | `K3G1` + `K1G1` x 9 + `K3G1` | K3区間を前後へ一つずつ分散 |
| K16-even | 16/10 | `K1G1` x 2 + `K3G1` + `K1G1` x 2 + `K3G1` + `K1G1` x 2 + `K3G1` + `K1G1` | K=16/G=10を全体へ分散するcontrol |

すべてmain network 26層である。config実装時は展開後のK/G/T数、総parameter数、main-network
parameter数をinspectionとunit testで確認する。

### 4.2 Control

- 主anchor: K1G1
- 文節・category specialist: K3G1
- 既存同数control: K1-first、K3-first
- 既存baseline: K3T1、T26

Phase 1ではK1G1を同じ新runnerで再学習する。K3G1、K1-first、K3-first、K3T1、T26は既存の
同条件112文artifactを比較表へ含める。新しいseed分離条件で直接比較が必要になった構成だけ追試する。
T26の22時点denseは初期screeningでは追加せず、最終候補がTransformer baselineを上回るという主張に
必要になった場合だけ同条件で実行する。

## 5. Seedの分離仕様

学習設定とCLIへ次を追加する。既存`seed`だけを含むconfig/checkpointは、3値すべてへ同じ値を設定する
fallbackで読み込めるようにする。

| seed | 用途 | 変更時に固定するもの |
| --- | --- | --- |
| `model_init_seed` | model parameterの初期化 | data order、training runtime RNG |
| `data_order_seed` | packed training sampleのshuffle | step 0 weight、training runtime RNG |
| `train_runtime_seed` | model作成後の学習時乱数 | initial weight、data order |

実装順序は次とする。

1. `model_init_seed`でPython/NumPy/PyTorch/CUDAをseedし、modelを構築する。
2. step 0 model state hashを記録し、必要な比較ではstep 0 checkpointを保存する。
3. datasetには`data_order_seed`だけを渡す。
4. model初期化後、optimizer step開始前に`train_runtime_seed`を再設定する。
5. checkpointへPython、NumPy、PyTorch CPU/CUDA RNG stateを保存し、resume時にtraining loop直前で復元する。

各run manifestへ次を保存する。

- 3 seed値
- step 0 state hash
- shuffle index hash
- 先頭N sampleの `(shard_id, chunk_offset)`
- config hash、commit、dataset manifest hash
- resume元、optimizer/data/RNG state復元結果

### 5.1 再現性audit

K1G1と新候補1構成について、同じ`(init=42, data=42, runtime=42)`を短い同条件で2回実行する。

- step 0 hashとshuffle hashが一致すること。
- 最初のbatch ID列が一致すること。
- 同一GPU/software条件でmetrics/checkpointが一致するか、差がある場合は非決定性の幅を記録すること。
- runtime seedを変える実験は、学習時の確率演算が存在し、同一triple反復幅を超える効果を持つ場合だけ
  行う。探索段階で無条件にruntime seedを3値へ増やさない。

## 6. 段階的実験

### Phase 0: 実装とローカル検証

1. 5候補configを追加する。
2. layout展開、層数、K/G数、parameter一致のunit testを追加する。
3. `model_init_seed`、`data_order_seed`、`train_runtime_seed`を実装する。
4. legacy `seed` fallback、dataset order分離、step 0再利用、RNG checkpoint round-tripをCPU unit testする。
5. runnerへ3 seedとmanifest audit情報を追加する。
6. ローカルで狭いtest、可能なら`tests/unit`を実行し、commit/push後にColabでpullする。

### Phase 1: 単一条件55-step architecture screening

対象はK1G1 + 新規5候補の6構成とする。全runを次で固定する。

- `model_init_seed=42`
- `data_order_seed=42`
- `train_runtime_seed=42`
- max step 55、LR horizon 220
- family dense: step 10/20/30/40/50
- full112: step 55、low/central/high/native

既存K3G1、K1-first、K3-first、K3T1、T26も同じ比較表へ含めるが、異なるartifact由来であることを
明示する。Phase 1では僅差の順位を確定せず、明確な退行の除外と構成特徴の確認を目的とする。

#### Phase 1継続条件

次を満たすstrict Pareto非劣位の上位2--3候補をPhase 2へ進める。

1. K1G1に対して文節precisionまたはcoverageが改善し、改善が一つの境界、一文、一profileだけに
   依存しない。
2. family coverage、landmark coverage、助動詞、助詞、複合語、structuredの複数軸でK1G1から
   一貫した大退行を起こさない。
3. centralだけでなくlow/high/nativeの少なくとも二条件で改善方向が確認できる。
4. dense fracture/unexplained occupancyが終盤へ急増しない。
5. validation BPB、圧縮率、学習安定性に重大な異常がない。

探索段階では固定のaggregate scoreを作らない。実用差の仮閾値はprecision/coverage/landmarkの
絶対差0.05、fracture occupancyの悪化0.10とするが、境界数とrecord実例を併記し、1--2境界だけの差を
昇格理由にしない。

### Phase 2: Seed要因のone-factor-at-a-time確認（55 step）

Phase 1上位2--3候補とK1G1を対象にする。baselineはPhase 1を再利用する。

| 条件 | init | data | runtime | 目的 |
| --- | ---: | ---: | ---: | --- |
| baseline | 42 | 42 | 42 | 基準 |
| init-43 | 43 | 42 | 42 | weight初期値差 |
| init-44 | 44 | 42 | 42 | weight初期値差 |
| data-43 | 42 | 43 | 42 | data順序差 |
| data-44 | 42 | 44 | 42 | data順序差 |

data条件ではarchitectureごとの同一step 0 checkpointをロードし、initial state hashの一致を必須とする。
候補とK1G1は同一data-order条件でpaired比較する。

#### Phase 2継続条件

- init系列とdata系列を別々に集計する。
- 各系列で候補−K1G1の方向が3条件中2条件以上で一致し、同一triple反復noiseを上回ること。
- 文節改善とfamily/landmark維持が別々のseedだけに依存する候補は昇格しない。
- init系列だけ、またはdata系列だけで順位が反転する場合は依存要因を明記し、220 step延長は最大1候補に
  絞る。
- どの候補もK1G1に対する有用なPareto軸を再現しなければ構成探索を停止する。

### Phase 3: 220-step持続性

K1G1とPhase 2の上位1--2候補だけを延長する。最初は次の3 paired条件をstep 220までresumeする。

- baseline `(42,42,42)`
- init差 `(43,42,42)`
- data差 `(42,43,42)`

family denseは10 step間隔を維持し、full112をstep 55/110/165/220で評価する。init-44/data-44は
55-stepで順位反転、効果の僅差、大きな分散があった場合だけ延長する。

長時間候補へ上げる直前に、2 x 2 crossed designで未観測の`(init=43, data=43, runtime=42)`を追加し、
initとdataのinteractionを確認する。探索段階では3 x 3 full gridを無条件に実行しない。

## 7. 評価と比較

主要軸は既存レポートと同じとする。

- category precision/coverage/fracture
- 文節、助動詞、助詞、複合語、structured、固有名詞、tool/identifier
- family precision/coverage、lexeme integrity
- landmark coverage/consistency、context signature consistency
- dense time/late precision/coverage、unexplained/fracture occupancy、transition
- low/central/high/native間の順位、実効compression ratio
- validation BPBは重大な退行のみ

集計は次を分ける。

1. 同一init/data条件内のarchitecture paired差
2. initだけを変えたarchitecture内分散
3. dataだけを変えたarchitecture内分散
4. crossed cellを含むinteraction

従来のrun-level seed 42/43/44平均と新しいfactorized seed結果を同じ「seed平均」として混ぜない。

## 8. 構成仮説の判定

- K14の前・中・後でcategory差が位置に追随すれば、単一K3区間の位置効果の手がかりとする。
- K14の3構成が似ていれば、位置より一つのG→K置換によるK/G量の寄与を疑う。
- K16-evenが既存K1-first/K3-firstと似ていればK/G比率寄与、異なればblock集中・位置寄与を疑う。
- K15-splitがK1-firstの文節とK3-firstのfamilyを両立すれば、前後分散を次の候補にする。
- K14-lateが文節を改善しつつK1G1のfamilyを維持すれば、KDA追加量を減らす仮説を支持する。
- どの小変更もK1G1のfamilyを壊す場合、単純なKDA block挿入探索を停止する。

これらはouter-stageとの共適応を含むarchitecture-levelの関連であり、特定層が言語categoryを直接
担当するという因果証明にはしない。

## 9. 停止条件と次の判断

- Phase 1で明確な候補がなければPhase 2を行わず、K1G1を長時間anchorとして維持する。
- seed要因分離後に改善方向が再現しなければ、追加layoutを無制限に増やさない。
- data-order効果がarchitecture差より大きければ、本計画終了後に初めてbalanced ordering/curriculumを
  別計画として検討する。
- init効果が大きければ、平均値だけでなく最悪seedとpaired差が安定する構成を優先する。
- K1G1のfamily/低fractureを維持し、文節を改善する候補がPhase 3で持続した場合だけ、K1G1とともに
  長時間実験候補へ上げる。

## 10. 成果物

| artifact | 保存先 |
| --- | --- |
| 本計画 | `plan/mid/20260823_kda_dosage_position_seed_factorized_screening_plan.md` |
| model configs | `configs/` |
| seed分離・runner code | `hnet/training/`, `train.py`, `scripts/` |
| unit tests | `tests/unit/` |
| raw checkpoints/evals/manifests | Drive `/content/drive/MyDrive/hnet_agent_200m_main/` |
| 中間memo | `plan/mid/` |
| 最終report | `results/main_200m/` |

## 11. 実行規則

- branchは`200m_main`だけを使用する。
- コードはローカルで修正・test・commit・pushし、Colabでpullする。
- GPU学習とGoogle DriveアクセスはColab経由で行う。
- Colab初期化時はclone、Drive mount、Prepare、GPU/import smoke testを再実行する。
- 長い処理はColab cellのforegroundで実行し、約10分ごとにcell状態またはmanifestを確認する。
- checkpoint、dataset、raw評価artifactはGitへcommitしない。
- ユーザーの確認なしにデータセット内容・順序設計・curriculumを変更しない。
