# main network追加中規模評価案

作成日: 2026-08-21

## 1. 目的

拡張日本語probeのPhase 2では、K1G1、K3G1、K3T1、T26で強いcategoryが異なり、
category別precision/coverageもstepとともに変化した。長期実験へ進む前に、既存結果の
追加解析と少数の混合main network実験で、次を確認する。

1. categoryの強みが一時的なものか、学習とともに獲得・喪失する傾向か。
2. KDA、Gated MLA、Transformerの配置順が分割品質に影響するか。
3. 良い境界proxyが、実際のcategory別byte lossにも対応するか。

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

### 3.1 解釈上の注意

「一般的なTransformerでは前層が文字・単語、後層が大きな構造を扱う」という傾向は、今回の
model-level category表から直接は確認できない。現在のprobeは最終router境界を測り、各層の
hidden representationを測っていないためである。

それでも、局所処理に強い構成を前半、統合に強い構成を後半へ置く仮説は十分検証価値がある。
配置効果と単なるlayer種類の比率を分けるため、同じ26 mixer blocks、同じK/G/T個数を持つ
order controlを比較する。

### 3.2 第一候補とorder control

uppercase blockはmixerとMLPを持つ。次の3構成はすべてK=13、G=4、T=9、合計26 blocksで、
parameter数も互いに同じになる。

1. **ordered**: `(K1G1)x4 → (K3T1)x3 → T6`
   - ユーザー提案に対応する。K/G交互を前半、K中心+周期Tを中盤、Tを後半へ置く。
2. **reverse**: `T6 → (K3T1)x3 → (K1G1)x4`
   - 同じ成分で順序だけを逆転する。
3. **interleaved**: `(K1G1 K3T1 T2)x3 → K1G1`
   - 同じ成分を全体へ分散し、段階配置そのものの効果を確認する。

既存のK3G1とK3T1をanchorにする。新構成間ではparameter差がなくても、anchorとはlayer種類の
個数が異なるため、parameter accounting、推定FLOPs、実測memoryを併記する。

### 3.3 段階実行

1. 3新構成をseed 42、step 55でscreeningする。
2. BPB重大退行、圧縮率乖離、severe fragmentationがない構成だけstep 220へ進める。
3. orderedがreverse/interleavedより一貫して良い場合だけseed 43、44を追加する。
4. 配置差が小さい場合、組合せ探索を増やさずK3G1/K3T1の長期比較へ戻る。

主判定は全体aggregateだけでなく、活用、助動詞、助詞、文節、複合語、structuredの
category trajectoryとnative頑健性で行う。

## 4. 層ごとの役割を直接確認する評価

混合構成の仮説を検証するには、最終境界だけでなく層別表現を見る必要がある。

### 4.1 Layerwise linear probe

各main-network層のhidden stateを保存し、独立した学習用annotationで次のoffset分類を行う。

- UTF-8文字境界ではなく、語彙内部 / 形態素 / 助詞・助動詞 / 文節 / 句読点 / structured delimiter
- 同じ88文をprobeの学習と評価へ兼用しない。
- 各層のlinear probe F1とselectivityを比較する。

これにより、前半で語・活用情報、後半で文節・構造情報が線形に読み出せるという仮説を直接
検証できる。最終router境界のcategory差だけから層の役割を推定するより強い証拠になる。

### 4.2 Layer-group intervention

early/middle/late groupの出力を小さくscaleする、または同一architecture内でgroupをfreezeして
短い継続学習を行い、category別byte lossと境界確率の変化を測る。単純な推論時ablationでは
分布外状態を作るため、linear probeを主、interventionを補助証拠とする。

## 5. 境界proxy以外で追加すべき比較

### 5.1 Category別byte loss curve

88文は境界galleryには適するがloss評価には小さい。日本語、英語、code、JSON/tool、長文の
独立した固定validation shardを用意し、全checkpointでBPBを測る。境界categoryの強みが、
該当データの予測しやすさへ結び付くかを確認する。

### 5.2 Chunk系列の学習容易性proxy

以下を大量のheld-out文章で測る。

- chunk長の平均だけでなく分散と極端値。
- 同一語・同一活用が異なる文脈で同じ/異なる分割になる頻度。
- 語彙内部切断率、形態素・文節境界との相互情報量。
- 反復するchunk patternの頻度とchunk系列entropy。

「説明可能な境界は後続networkが学びやすい」という仮説に、88文のprecision/coverageより
近いproxyになる。

### 5.3 220 step以降の継続確認

上位anchorとhybrid winnerだけをstep 440または880まで延長する。今回、境界の出現・消失が
step 165→220でも続いたため、220 stepでのcategory順位が固定したとは言えない。延長時も
同じraw-byte予算とcheckpoint間隔で比較する。

### 5.4 長文・文脈距離

単文中心の88文に加え、同じtarget spanを短文、複数文、tool trajectory中へ置く。targetから
離れた前文だけを変え、境界確率がどの距離まで影響を受けるか測る。動的chunkの特徴を
短い最小対だけでなく、実際の長文・agent文脈で確認する。

## 6. 優先順位

追加中規模実験は次の順が費用対効果に優れる。

1. 既存48結果のcategory trajectory、bootstrap、budget/native、stage間解析。
2. 同一成分のordered/reverse/interleavedをseed 42・55 stepで比較。
3. 独立validation shardのcategory別BPBと長文context probe。
4. 勝者のみ3 seed・220 step、必要なら440/880 stepへ延長。
5. layerwise linear probeで層配置仮説を直接検証。

この順序なら、category表から混合構成を大量生成することを避けつつ、配置順の仮説と
長期学習候補の両方を中規模範囲で絞り込める。
