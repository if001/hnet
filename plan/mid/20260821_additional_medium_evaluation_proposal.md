# main network追加中規模評価案

作成日: 2026-08-21

## 1. 目的

拡張日本語probeのPhase 2では、K1G1、K3G1、K3T1、T26で強いcategoryが異なり、
category別precision/coverageもstepとともに変化した。長期実験へ進む前に、既存結果の
追加解析、少数の混合main network実験、初期データcurriculumで次を確認する。

1. architectureごとのcategory差が、成分比、配置順、seed、学習段階のどれに主に対応するか。
2. KDA、Gated MLA、Transformerをどう組み合わせると、precision、coverage、category間の
   バランス、lexeme fractureを改善できるか。
3. encoder/decoderの境界が学習中にどの程度揺れ、説明可能な複数分割間の移動と、
   説明困難な分割への不安定化を区別できるか。
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

### 2.5 Dense boundary trajectory

step 55/110/165/220は粗い方向確認には使えるが、特定の分割が維持されたか、許容分割間を
移動したかを判定するには疎すぎる。固定probeを同じ推論条件で5または10 stepごとに評価し、
離散boundary maskだけでなく全offsetのboundary probabilityを保存する。

88文すべてを高頻度で評価すると費用が大きいため、次の二層に分ける。

- **dense core**: categoryを均等に含む22文を5--10 stepごとにnative/centralで評価する。
- **full probe**: 88文を25--55 stepごとにlow/central/high/nativeで評価する。

checkpointを毎回保存せず、training中のdeterministic evaluation hookで小さいprobability JSONを
保存する。stepではなくcumulative raw bytesを時間軸にする。hookは`inference_mode`で実行し、
training/eval modeと乱数状態を保存・復元して、評価挿入自体が学習trajectoryを変えないことを
unit testで確認する。

各focus spanで次を集計する。

- **acceptable occupancy**: 各許容offsetがlate-window内で選ばれた比率。
- **unexplained / fracture occupancy**: 説明困難境界とlexeme fractureが選ばれた時間比率。
- **segmentation transition**: `分|割する`、`分割|する`などpattern間の遷移回数、滞在時間、再出現。
- **acceptable probability margin**: 許容offset確率と保護語彙内部offset確率の差。
- **temporal Jaccard / rank correlation**: 連続時点のmask一致度とprobability順位相関。
- **late-window mean and variance**: 最終1点ではなく、最後の50--100M bytesの平均と分散。
- **time-averaged precision/coverage**: 全学習区間とlate-windowを分けたcurve下面積。

分割が変化すること自体を悪いとしない。次を区別する。

1. **有益な揺れ**: 複数の許容分割間を移動し、unexplained/fracture occupancyが低い。
2. **有害な揺れ**: 説明困難・語彙内部分割へ頻繁に移動し、marginも低い。
3. **安定した誤り**: Jaccardは高いが、不自然な分割へ長く滞在する。
4. **収束傾向**: late-windowで有害遷移と分散が減り、許容offsetの占有率が上がる。

モデル比較は最終stepのpatternではなく、late-windowのacceptable occupancy、fracture occupancy、
precision/coverage平均、pattern entropy、seed間のtrajectory分布で行う。

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

## 5. 保留: 初期境界学習用data curriculum

### 5.1 元論文、現行条件、仮説

元H-Net論文は、dynamic chunkingをcontent/context-dependentで、main networkを含むモデル全体と
jointに学習される仕組みとしている。また、outer stageはinner stageより多くの入力を処理し、
outer stageの高い学習率がchunking mechanismの学習を加速すると経験的に述べている。ただし、
初期に獲得した境界がその後も固定される、またはそのまま保存されるとは述べていない。

- 論文: <https://arxiv.org/abs/2507.07955>
- ICLR 2026版: <https://openreview.net/forum?id=ZbfLR9NbNF>

現行の中規模学習は外側から`lr_multipliers=[2.0, 1.5, 1.0]`を使い、outer/middleの
encoder/decoder側がinnermost main networkより速く更新される。ただし、clean初期データが
再利用可能な境界基盤を形成するという仮説は、dense trajectoryで通常学習の境界変動を確認するまで
保留する。実行を決定した場合の仮説候補は次のとおりである。

1. 初期の整形済み・広分布データにより、多くのdomainで再利用できる境界確率の基盤が速く形成される。
2. 学習が進むと完全に固定されるのではなく、基盤の一部を維持しながら現在のデータ分布に合うよう
   boundary probabilityと選択境界が揺れる。
3. 初期データが通常mixの場合にも基盤は形成されるが、頻度の高いdomainやノイズへ偏る可能性がある。

高い学習率は後続データへの再適応も速くするため、初期データの影響が長く残るとは自動的には
言えない。また、現行packed datasetはseed固定で全体shuffleされるため、物理的な元データの
先頭を整えるだけではcurriculumにならない。明示的な順序制御が必要である。

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

### 5.3 第一比較: architectureとデータ総量を固定する

最初の実験ではarchitectureをK3G1に固定する。学習率、optimizer、seed、総raw bytes、
source別の総サンプル数も固定し、同じデータmultisetの提示順だけを変える。

1. **normal-shuffle**: 通常mixを現行方式で全体shuffleする基準条件。
2. **clean-first**: 同じmultisetに含まれる整形済み・広分布subsetを初期20--40M bytesへ集め、
   その後に残りをshuffleして与える。

normal-shuffleにもclean-firstと同じclean samplesが含まれるため、最終的なデータ組成差ではなく
「初期にまとめて与えた効果」を比較できる。初回比較ではarchitecture変更、outer-LR taper、
clean data増量を同時に行わない。

K3G1 seed 42で方向を確認し、差がある場合にseed 43、44を追加する。K3T1やhybridへの展開は、
順序効果が3 seedで再現した後に行う。

既存K3G1 runとデータmultiset、shuffle index、学習設定が完全一致する場合は、normal-shuffleを
再利用できる。一致しない場合は公平性のため両条件を新たに学習する。

### 5.4 phase移行条件

clean-firstのprefix終了時をanchor checkpointとし、その後の通常mixで何が維持され、何が
現在分布へ適応するかを測る。次の3種類を分ける。

- **base formation**: prefix中にcommon probeのprecision/coverage、margin、lexeme fractureが改善する。
- **base retention**: prefix終了時と後続checkpointの間で、domain共通の説明可能境界と
  boundary probability順位がbaselineより多く維持される。
- **adaptive movement**: 現在のデータdomainに対応するprobeでは境界確率が適切な方向へ動き、
  common probe全体を破壊せずにcategory配分が変わる。

完全な境界mask一致を要求しない。境界mask Jaccardだけでなく、許容offsetのprobability相関、
top-k順位相関、出現・消失した境界のcategoryを使う。

固定stepだけでなく、次が2回連続のcheckpointで小さくなった時点をencoder/decoderの
「ある程度収束した」proxyとする。

- prefix終了anchorおよび直前checkpointに対するcentral/nativeのboundary Jaccard変化。
- category別precision/coverageとlexeme fractureの傾き。
- boundary probabilityのtop-k cutoff margin。
- native圧縮率とtarget gap。
- outer/middle/mainのparameter update normまたはgradient norm。

評価probeは、学習全体で固定するcommon probeに加え、clean subset probeと、各後続区間の
dominant domain probeを用意する。これにより「単なる忘却」と「現在分布への適応」を区別する。

中規模screeningでは10M、20M、40M、60M raw bytes付近を観測し、20--40Mを初期候補とする。
実測curveが飽和しなければ初期phaseを延長し、早く飽和すれば短縮する。

### 5.5 実装上の注意

現在のdatasetは単一packed dirを全体shuffleするため、同一multisetの一部をprefixへ送る
phase-aware indexまたはsamplerが必要である。normal-shuffleとclean-firstでsample ID、出現回数、
総bytesが一致することをmanifestとunit testで検証する。

初回比較ではLR multiplierを固定し、単一run内でdata orderだけを切り替える。checkpoint resumeや
optimizer resetによる差を入れない。outer-LR taperはclean-firstの順序効果が確認された後の
独立した第二実験とする。

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
2. architectureをK3G1に固定し、dense coreを5--10 step、full probeを25--55 step間隔で追跡する。
3. late-window occupancy、遷移、margin、時間平均precision/coverageをseed 42で確認する。
4. trajectoryがseed依存ならseed 43、44を追加する。
5. 同一成分のordered/reverse/interleavedを同じdense評価で比較する。
6. 独立validation shardのcategory別BPBと長文context probeを併用する。
7. clean-first curriculumは通常学習のtrajectoryを理解した後に再検討する。
8. 勝者のみ220 step以降へ延長する。

この順序なら、疎なsnapshotを安定した分割戦略と誤認せず、説明可能な揺れと有害な揺れを区別できる。
clean初期データの効果を先に仮定せず、通常学習の時間変化を理解してからdata orderとarchitectureを
評価できる。
