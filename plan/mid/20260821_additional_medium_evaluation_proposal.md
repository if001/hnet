# main network追加中規模評価案

作成日: 2026-08-21

## 1. 目的

拡張日本語probeのPhase 2では、K1G1、K3G1、K3T1、T26で強いcategoryが異なり、
category別precision/coverageもstepとともに変化した。長期実験へ進む前に、既存結果の
追加解析、少数の混合main network実験、初期データcurriculumで次を確認する。

1. architectureごとのcategory差が、成分比、配置順、seed、学習段階のどれに主に対応するか。
2. KDA、Gated MLA、Transformerをどう組み合わせると、precision、coverage、category間の
   バランス、lexeme fractureを改善できるか。
3. encoder/decoderが高い学習率で境界を形成する初期に、整形済み・広分布データを与える
   curriculumが分割の品質と安定性を改善するか。
4. 良い境界proxyが、実際のcategory別byte lossにも対応するか。

一般Transformerで報告された層別の言語情報を再確認すること自体は、本計画の目的ではない。
また、0.23B bytes・220 stepの中規模学習から細かな機能分担を断定しない。ここで求めるのは、
長期候補を絞るための粗い原因切り分けと、再現可能な改善方向である。

## 2. 既存checkpointで追加できる解析

### 2.1 Category trajectory

Phase 2 v2の48結果から、中央固定予算・stage 1のcategory別precision/coverageを
step 55、110、165、220で再集計した。主要な変化は次のとおりである。

- T26: 活用は`.398/.238`から`.178/.095`へ低下。structuredも低下した。一方、文節、
  句読点、identifierは途中から改善した。
- K1G1: structuredは`.272/.089`から`.512/.215`、identifierと句読点も改善した。
  助動詞と助詞は全期間で強いが、文節は低いままだった。
- K3G1: structuredとidentifierは改善した。助詞、複合語、固有名詞は早期の方が高く、
  文節はstep 110で最大になった後に少し低下した。
- K3T1: 活用は全期間で強く、structuredとidentifierは改善した。複合語は早期から低下し、
  固有名詞は不安定だった。

この結果は、単純な単調改善ではなく、限られた境界予算をcategory間で再配分している可能性を
示す。step 220だけで「そのnetworkが得意なcategory」を固定的に解釈しない。

保存先: Drive
`reports/linguistic_boundary_selection/phase2_v2/category_trajectory.json`

次に、seedとrecordの二段階bootstrapで各curveの信頼区間を求める。8文/categoryしかないため、
1、2境界の差をarchitecture差として扱わない。

### 2.2 境界予算に対する頑健性

各step・categoryについてlow/central/high/nativeを比較し、次を集計する。

- category順位が境界予算を変えても保たれるか。
- top-k cutoff付近のboundary probability margin。
- centralでのみ良く、nativeでは崩れるcategory。
- 境界数の増加がexplainable境界よりlexeme fractureへ使われる割合。

K3G1はcentralで良い一方native短断片が多いため、この解析を長期候補選択の必須条件とする。

### 2.3 Stage 0からStage 1への誤差伝播

stage 0とstage 1を別々に集計し、stage 1のlexeme fractureがstage 0の境界選択により既に
確定していたか、stage 1で新たに選ばれたかを分類する。これによりmain networkの差と、
encoder/router側の差を混同しにくくする。

### 2.4 Dynamic pair評価の修正

現在の完全signature passはstep 220で全候補0/6となり、差を検出できなかった。完全一致ではなく、
事前指定した許容offsetのboundary probabilityが文脈によりどちらへ動くかをpaired effect sizeで
評価する。文脈だけを変えた最小対を増やし、seedをまたいだ方向一致率を併記する。

## 3. 混合main network実験

### 3.1 検証対象

目的は一般Transformerの層役割を再現することではなく、今回観測したmodel-levelのcategory差が
なぜ生じたかを粗く切り分け、組合せにより指標を改善できるかを調べることである。

現在のprobeは最終router境界を測っているため、K1G1が助動詞・structuredに強いことから
「K1G1は前半向き」、K3T1が活用に強いことから「K3T1は中盤向き」と直接結論できない。
ただし、異なるmixerの帰納biasと配置順の相互作用がend-to-endの勾配を通じて境界学習へ影響する
可能性はある。次の3要因を区別する。

1. **成分効果**: K/G/Tを何層ずつ含むか。
2. **順序効果**: 同じ成分をearly/middle/lateのどこへ置くか。
3. **学習段階効果**: category差がstepとともに獲得・喪失されるか。

中規模では各mixerが担う細かな言語機能を特定せず、同一成分のorder controlで順序効果が
再現するかを主に確認する。

### 3.2 第一候補とorder control

uppercase blockはmixerとMLPを持つ。次の3構成はすべてK=13、G=4、T=9、合計26 blocksで、
parameter数も互いに同じになる。

1. **ordered**: `(K1G1)x4 → (K3T1)x3 → T6`
   - category差を組み合わせる一つの仮説。一般Transformerの層役割を証明する構成ではない。
2. **reverse**: `T6 → (K3T1)x3 → (K1G1)x4`
   - 同じ成分で順序だけを逆転する。
3. **interleaved**: `(K1G1 K3T1 T2)x3 → K1G1`
   - 同じ成分を全体へ分散し、段階配置そのものの効果を確認する。

既存のK3G1とK3T1をanchorにする。新構成間ではparameter差がなくても、anchorとはlayer種類の
個数が異なるため、parameter accounting、推定FLOPs、実測memoryを併記する。

### 3.3 段階実行

1. 3新構成をseed 42、step 55でscreeningする。
2. BPB重大退行、圧縮率乖離、severe fragmentationがない構成だけstep 220へ進める。
3. いずれかが他のorder controlより一貫して良い場合だけseed 43、44を追加する。
4. 配置差が小さい場合、組合せ探索を増やさずK3G1/K3T1の長期比較へ戻る。

主判定は全体aggregateだけでなく、活用、助動詞、助詞、文節、複合語、structuredの
category trajectoryとnative頑健性で行う。

## 4. Category差の粗い原因分析

大規模学習でないため、層ごとの細かな機能分担をlinear probeで断定することは目的にしない。
代わりに、候補を増やしすぎない次の比較を行う。

### 4.1 同一成分の順序差

ordered/reverse/interleavedの差が3 seedで再現すれば配置順の寄与、差がなければ主に成分比または
学習ノイズと判断する。1、2 categoryだけの勝ちではなく、aggregate、category macro、
lexeme fracture、native頑健性が同方向に動くことを要求する。

### 4.2 Anchorからの局所置換

order screeningで差が出た場合だけ、K3G1またはK3T1のearly/middle/lateの一群を1回だけ
別mixerへ置換する。複数箇所を同時に変えず、どの置換がcategory curveを変えたかを見る。
parameter数が変わる場合はMLP幅を調整し、総parameter差を1%以内にする。

### 4.3 補助的なgroup sensitivity

必要な場合だけ、early/middle/late groupの出力scaleを小さく変えた推論感度を測る。
これは分布外interventionであり、機能局在の証明には使わない。新構成を学習する前に、
影響がほぼない置換候補を除外するscreening用途に限定する。

## 5. 初期境界学習用data curriculum

### 5.1 現行条件と仮説

現行の中規模学習は外側から`lr_multipliers=[2.0, 1.5, 1.0]`を使い、outer/middleの
encoder/decoder側がinnermost main networkより速く更新される。したがって初期データ分布が
初期境界へ強く影響するという仮説には妥当性がある。

ただし、高い学習率は後続データへの再適応も速くするため、初期データの影響が長く残るとは
自動的には言えない。また、現行packed datasetはseed固定で全体shuffleされるため、物理的な
元データの先頭を整えるだけではcurriculumにならない。明示的にphaseを分ける。

静的tokenizerが本体より少量のデータで学習できることは参考になるが、H-Netの境界は
言語モデルlossとcompression lossからend-to-endで変化する。必要量をtokenizerの慣例から
決めず、境界指標の飽和から決める。

### 5.2 初期データ

初期phaseは品質だけで狭くせず、次を均等に近く含む整形済み・重複除去済みの広分布mixとする。

- 日本語の説明文、対話、ニュース・百科事典調、活用と助詞が豊富な自然文。
- 英語自然文。
- code、identifier、path、URL。
- JSON、YAML、XML、Markdown、tool call、複数turn agent trajectory。
- 数値・単位、句読点、括弧、固有名詞を含む文。
- 短文だけでなく、文書境界を保った複数文と中程度の長文。

初期phaseでは、壊れたencoding、極端な反復、テンプレートboilerplate、途中切断されたtool列、
単一domainの過剰比率を避ける。

### 5.3 比較条件

総raw bytesと各sourceの総量をそろえ、順序とLR scheduleを分離する。

1. **mixed-from-start**: 現行の全体shuffle。基準条件。
2. **curated-prefix**: 同じ総量のうち初期20--40M bytesを整形済み広分布mixにし、その後を
   大量の通常mixへ切り替える。
3. **curated-dispersed**: curated-prefixと同じcurated samplesを全期間へ分散する。
   prefix効果と単なるデータ組成効果を分ける。
4. **curated-prefix + outer-LR taper**: 境界安定後に倍率を例として`[1.2, 1.1, 1.0]`へ下げ、
   main networkを中心に学習しつつencoder/decoderもわずかに更新する。

最初はK3G1とK3T1のseed 42で比較する。curated-prefixが両方で同方向に改善した場合だけ、
hybrid screeningへ同じcurriculumを適用する。

### 5.4 phase移行条件

固定stepだけでなく、次が2回連続のcheckpointで小さくなった時点をencoder/decoderの
「落ち着いた」proxyとする。

- central/nativeのboundary Jaccard変化。
- category別precision/coverageとlexeme fractureの傾き。
- boundary probabilityのtop-k cutoff margin。
- native圧縮率とtarget gap。
- outer/middle/mainのparameter update normまたはgradient norm。

中規模screeningでは10M、20M、40M、60M raw bytes付近を観測し、20--40Mを初期候補とする。
実測curveが飽和しなければ初期phaseを延長し、早く飽和すれば短縮する。

### 5.5 実装上の注意

現在のdatasetは単一packed dirを全体shuffleするため、段階dataset切替またはphase-aware samplerが
必要である。checkpoint resumeはmodel、optimizer、step、data stateを保持できるが、optimizer
load時のparameter-group倍率を含め、LR multiplier変更が意図どおり反映されることをunit testで
確認する。optimizerをresetするとcurriculum以外の差が入るため、原則resetしない。

## 6. 境界proxy以外で追加すべき比較

### 6.1 Category別byte loss curve

88文は境界galleryには適するがloss評価には小さい。日本語、英語、code、JSON/tool、長文の
独立した固定validation shardを用意し、全checkpointでBPBを測る。境界categoryの強みが、
該当データの予測しやすさへ結び付くかを確認する。

### 6.2 Chunk系列の学習容易性proxy

以下を大量のheld-out文章で測る。

- chunk長の平均だけでなく分散と極端値。
- 同一語・同一活用が異なる文脈で同じ/異なる分割になる頻度。
- 語彙内部切断率、形態素・文節境界との相互情報量。
- 反復するchunk patternの頻度とchunk系列entropy。

「説明可能な境界は後続networkが学びやすい」という仮説に、88文のprecision/coverageより
近いproxyになる。

### 6.3 220 step以降の継続確認

上位anchorとhybrid winnerだけをstep 440または880まで延長する。今回、境界の出現・消失が
step 165→220でも続いたため、220 stepでのcategory順位が固定したとは言えない。延長時も
同じraw-byte予算とcheckpoint間隔で比較する。

### 6.4 長文・文脈距離

単文中心の88文に加え、同じtarget spanを短文、複数文、tool trajectory中へ置く。targetから
離れた前文だけを変え、境界確率がどの距離まで影響を受けるか測る。動的chunkの特徴を
短い最小対だけでなく、実際の長文・agent文脈で確認する。

## 7. 優先順位

追加中規模実験は次の順が費用対効果に優れる。

1. 既存48結果のcategory trajectory、bootstrap、budget/native、stage間解析。
2. K3G1/K3T1でmixed-from-start、curated-prefix、curated-dispersedをseed 42で比較。
3. 初期境界が改善した場合、outer-LR taperの有無を比較する。
4. 採用curriculum上で同一成分のordered/reverse/interleavedをseed 42・55 stepで比較。
5. 独立validation shardのcategory別BPBと長文context probeを併用する。
6. 勝者のみ3 seed・220 step、必要なら440/880 stepへ延長する。

この順序なら、encoder/decoderが初期データへ適応する効果とmain networkの構成差を混同せず、
category表から混合構成を大量生成することも避けられる。中規模で細かな機能局在を主張せず、
長期学習へ持ち込む再現可能なdata scheduleとarchitectureを絞り込む。
