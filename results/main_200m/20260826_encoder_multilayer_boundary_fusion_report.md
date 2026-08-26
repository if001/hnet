# Encoder複数層融合による境界決定の評価

## 1. 結論

H-Netの境界決定にEncoder最終層だけでなく複数層の特徴を渡す方法を、約200M parameterの
T26、K1G1、K3G1で比較した。評価対象は言語モデルの生成品質ではなく、日本語文を言語的に説明しやすい
chunkへ分割できるか、その品質が学習中に安定して形成されるかである。

結論は次の通りである。

- **T26では複数層融合を採用候補とする。** step 180--220の5時点平均で、category coverage、
  fracture、family coverage、landmark coverage、family integrity、文節precision/coverageが同時に改善した。
  許容できない退行を別指標の改善で相殺する「非相殺制約」にも5時点すべてで合格した。
- **K1G1では採用しない。** family precisionは上がったが、family coverage、landmark coverage、
  文節coverageが大きく下がった。選ぶ境界を狭くしてprecisionを上げた結果であり、分割能力全体の改善ではない。
- **K3G1ではstep 100で停止した。** 文節precisionとfamily integrityには改善があったが、category
  precision/coverage、family coverage、landmark coverageが低下し、path、identifier、固有名詞に細かい
  不自然な分割が見られた。
- 効果はmain networkによって異なる。同じ融合重みを加えれば一律に良くなるわけではなく、
  **T26の境界routerと複数層特徴の組合せに固有の改善**と判断する。
- T26でも万能ではない。活用、固有名詞、数値・単位のcoverageは悪化し、複合語とidentifierでは
  fractureが増えた。したがって、現段階では長期学習用base modelの確定ではなく、再現性確認へ進める候補である。

## 2. 実験の目的

H-Netはbyte列を固定tokenizerで分割せず、入力と文脈に応じて動的にchunkへまとめる。chunk境界は
Encoderの特徴からboundary routerが決めるため、main networkやEncoderから渡す特徴が変わると分割も変化する。

従来構成ではboundary routerへEncoder最終層だけを渡していた。しかし、最終層だけでは、文字・語彙内部の
局所的な形と、助詞・助動詞・文節などの文脈的な手がかりを同時に保持しにくい可能性がある。

この実験では、Encoderの各層を学習可能な重みで融合してboundary routerへ渡し、次を確認した。

1. 説明可能な境界を広く拾いつつ、語彙内部の不自然な分断を減らせるか。
2. 効果が一つのcheckpointだけでなく、学習stepの軌跡として持続するか。
3. TransformerのみのT26、KDAを異なる割合で含むK1G1/K3G1で効果が共通するか。
4. 分割数を外部から変えた場合にも改善が残るか。

小規模な220 step学習では、生成文やvalidation lossから長期学習後の言語能力を判断できない。そのため本実験は、
「言語的に説明しやすい分割は、後続main networkが再利用しやすい表現を学ぶ候補になる」という仮説に基づく
base model screeningである。この実験だけで、下流言語モデルの最終性能が高いとは結論しない。

## 3. 用語と比較構成

### 3.1 main network

- **T26**: 26層をTransformerで構成したbaseline。
- **K1G1**: Kimi Delta Attention（KDA）1層とGated MLA 1層を交互に置く構成。
- **K3G1**: KDA 3層とGated MLA 1層の組を繰り返す、KDA比率の高い構成。
- **KDA**: 長いcontextを効率的に扱う目的で導入したattention系mixer。本実験は2K contextなので、
  長context計算量の優劣ではなく、main networkが境界形成へ与える影響を比較する。

各main networkについて次の二つを比較した。

- **control**: Encoder最終層だけをboundary routerへ渡す従来方式。
- **fusion**: Encoderの全層をscalar weightで融合してboundary routerへ渡す方式。

### 3.2 複数層融合

各Encoder層の正規化済み出力を `h_1, ..., h_L`、学習可能なscalarを `a_1, ..., a_L` とし、
softmaxで和が1になる重みを作る。

`boundary feature = sum(softmax(a)_l * h_l)`

融合特徴を使うのはboundary routerだけである。chunk内容をmain networkへ渡す経路は従来どおり最終層を使う。
したがって、結果の差は「境界決定へ複数層を渡した効果」として解釈しやすい。

H-Netは2段階で圧縮するため、内側をstage 0、外側をstage 1と呼ぶ。stage 0に4個、stage 1に5個の
scalar weightを追加し、parameter増加は全構成で9個だけである。

## 4. 評価データと実験条件

### 4.1 full112 probe

評価には固定した112文を用いた。

- **category 88文**: 活用、助動詞、助詞、複合語、固有名詞、句読点、文節、identifier、構造化表現など
  11 categoryを評価する。
- **family 24文**: 同じ語や活用要素が異なる文脈に現れる組を評価する。単一文で偶然良い境界になったかではなく、
  関連表現で再利用可能な境界を選べるかを見る。

評価文は学習lossの教師には使っていない。

### 4.2 固定条件

|項目|設定|
|---|---|
|モデル規模|約200M parameters|
|context length|2,048 bytes|
|学習データ|日本語8、英語1、code 1のpacked mixture|
|model初期値seed|42|
|データ順seed|42|
|training runtime seed|42|
|byte境界制約|`utf8-hard`|
|native評価|10 stepごと、step 10--220|
|profile評価|step 110、165、220|
|主評価stage|stage 1|

seedを初期weight、データ順、runtimeに分離し、今回のscreeningでは三つとも42に固定した。したがって、
controlとfusionの差にはseed要因を混ぜていない。一方、別seedでの再現性はまだ確認していない。

`utf8-hard`はUTF-8の複数byte文字の途中へ境界を置かない設定である。UTF-8違反数はこの設定でほぼ決まるため、
モデル選択指標には用いない。

## 5. 評価指標

precisionとcoverageは役割が異なるため、一つのF値や順位へ合成しない。

|指標|何を確認する指標か|値の読み方|
|---|---|---|
|category precision|モデルが選んだ境界のうち、11 categoryの許容分割で説明できる割合|大きいほど、選んだ境界に不自然なものが少ない|
|category coverage|注釈した許容境界候補のうち、モデルが実際に拾った割合|大きいほど、説明可能な境界を広く利用している|
|category fracture record occupancy|保護すべき語彙内部を分断した文の割合|小さいほどよい。1文に複数fractureがあっても、その文を1件として数える|
|family precision|family 24文で、選んだ境界のうち説明可能な割合|大きいほどよい。ただしcoverage低下との交換に注意する|
|family coverage|family 24文で、許容境界を拾った割合|大きいほど、関連表現で使える境界を失っていない|
|landmark coverage|family間で共通して期待する特定境界を選んだ割合|大きいほど、活用や文脈が変わっても再利用できる境界を保つ|
|family integrity|family文のうち、保護語彙を内部で壊さなかった割合|大きいほどよい|
|文節 precision|category 88文中の文節subsetで、選んだ境界が説明可能な割合|大きいほど、選択した文節境界の正確さが高い|
|文節 coverage|文節subsetの許容境界を拾った割合|大きいほど、文節構造を広く利用している|

例えばprecisionだけを上げるなら、境界をほとんど選ばない方法でも達成できる。そのため、precision上昇と
coverage低下を相殺して「改善」とは扱わない。またfractureとfamily integrityには許容退行幅を設け、
他指標が良くても許容範囲外なら候補から外した。

### 5.1 学習軌跡

境界は学習中に揺れるため、単一stepを主結論にしない。

- step 100 screeningではstep 60--100の5時点を見る。
- step 220判定ではterminal windowであるstep 180--220の5時点を見る。
- **同一step比較**は、同じoptimizer update数に対する品質を比較する。
- **coverage matched-state比較**は、category coverageが0.03以内の時点同士を比較し、分割量が近い場合の
  precision、fracture、integrityを見る。

### 5.2 compression profile

nativeはモデル自身が選んだ境界数を使う。profile評価では境界確率上位を使って分割数を外部から揃える。

- **low compression**: 2.5 units/chunk。chunkを多めに作る。
- **central**: 3.0 units/chunk。
- **high compression**: 3.5 units/chunk。chunkを少なめに作る。

これにより、改善が単に「境界を多く／少なく選んだ」結果か、異なる分割量でも残るかを確認する。

## 6. step 100 screening

表はstep 60--100の5時点における、`fusion - control`の平均差である。precision、coverage、integrityは
正が改善、fractureは負が改善を表す。

|構成|category P|category C|fracture|family P|family C|landmark C|integrity|文節 P|文節 C|判断|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|T26|+0.091|-0.003|-0.164|-0.206|+0.067|+0.008|-0.033|-0.005|-0.035|step 220へ継続|
|K1G1|+0.045|-0.075|-0.141|+0.049|-0.176|-0.050|+0.025|-0.059|-0.139|family C退行を監視して継続|
|K3G1|-0.019|-0.027|-0.061|-0.017|-0.139|-0.158|+0.067|+0.041|+0.009|step 100で停止|

T26ではcategory precisionとfractureが持続的に改善した。K1G1はprecisionとfractureが改善したが、
coverageが狭くなった。K3G1は文節とintegrity以外の主要指標が悪化し、step 100ではpath、camelCase、URL、
固有名詞などに細かい分割が見られたため、長く学習する候補にしなかった。

## 7. step 220の結果

### 7.1 terminal windowの絶対値

step 180--220の5時点平均を示す。`F`はfracture record occupancyであり、この列だけ小さいほどよい。

|構成|category P|category C|category F|family P|family C|landmark C|integrity|文節 P|文節 C|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|T26 control|0.396|0.274|0.302|0.722|0.545|0.667|0.908|0.143|0.217|
|T26 fusion|0.388|0.287|0.270|0.629|0.582|0.783|0.958|0.179|0.243|
|K1G1 control|0.425|0.464|0.425|0.449|0.667|0.850|0.833|0.159|0.322|
|K1G1 fusion|0.424|0.356|0.414|0.720|0.515|0.700|0.792|0.000|0.000|

### 7.2 controlからの変化

|構成|category P|category C|category F|family P|family C|landmark C|integrity|文節 P|文節 C|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|T26 fusion|-0.008|+0.014|-0.032|-0.093|+0.036|+0.117|+0.050|+0.036|+0.026|
|K1G1 fusion|-0.001|-0.109|-0.011|+0.271|-0.152|-0.150|-0.042|-0.159|-0.322|

T26 fusionはcategory precisionとfamily precisionを少し下げたが、coverageを広げながらfractureを減らし、
family landmarkとintegrityも改善した。単に境界数を減らしてprecisionを上げたものではない。terminal 5時点の
いずれにもrelaxed制約違反はなく、改善項目が同時に成立した。

coverage matched-stateでは、T26 fusionのcategory precision差はおおむね-0.009から-0.002と小さく、
fractureは同等または低かった。したがってcategory precisionの小さな低下の一部は、より多くの境界を拾ったことに伴う
precision/coverage trade-offと解釈できる。

K1G1 fusionはfamily precisionが0.449から0.720へ上がった一方、family coverageは0.667から0.515、
landmark coverageは0.850から0.700へ下がった。文節precision/coverageはterminal平均でともに0となった。
これは、選ぶ境界を限定したため残った境界のprecisionだけが高くなった状態であり、目的とする改善ではない。

## 8. T26のcompression profile

表は同じstep・同じprofileにおける `T26 fusion - T26 control` である。主要な制約とcoverageを示す。

|step|profile|category C|category F|family C|landmark C|integrity|文節 P|文節 C|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|110|low|-0.004|-0.193|-0.030|+0.042|+0.042|-0.032|-0.043|
|110|central|+0.013|-0.068|+0.121|+0.125|+0.042|+0.023|+0.043|
|110|high|+0.038|-0.011|+0.182|+0.208|-0.083|+0.034|+0.043|
|165|low|-0.013|-0.102|-0.121|-0.042|+0.125|+0.031|+0.043|
|165|central|-0.026|-0.011|+0.091|+0.125|0.000|+0.040|+0.043|
|165|high|+0.013|+0.011|+0.091|+0.125|0.000|0.000|0.000|
|220|low|-0.009|-0.102|-0.242|-0.167|-0.125|+0.011|0.000|
|220|central|-0.004|-0.023|+0.030|+0.042|0.000|+0.067|+0.087|
|220|high|+0.017|-0.011|+0.061|+0.083|0.000|+0.063|+0.043|

centralとhigh compressionでは、3 checkpointを通してfamily coverageとlandmark coverageが概ね改善した。
step 220でもcategory fractureを増やさず、文節precision/coverageを改善した。したがってT26の改善はnativeの
境界数だけに依存していない。

一方、step 220のlow compressionではfamily coverage、landmark coverage、integrityが悪化した。細かく分割する
条件では、途中層の特徴を加えたrouterが語彙内部の候補まで選びやすくなる可能性がある。T26 fusionを採用候補にするが、
高い境界密度への頑健性は未解決である。

## 9. category別にわかったこと

T26 fusionのterminal windowでは、次の改善が見られた。

- 文節: precision `+0.036`、coverage `+0.026`、fracture `-0.125`。
- 助詞: precision `+0.033`、coverage `+0.162`。
- context control: precision `+0.077`、coverage `+0.015`、fracture `-0.050`。
- structured表現: coverage `+0.058`、fracture `-0.025`。
- 句読点: fracture `-0.375`。

一方、次の弱点が残った。

- 活用: precision `-0.438`、coverage `-0.243`。
- 固有名詞: precision `-0.173`、coverage `-0.100`。
- 数値・単位: coverage `-0.100`。
- 複合語: fracture `+0.225`。
- identifier: coverageは`+0.065`だがfractureも`+0.175`。

全体fractureは改善していても、category別には複合語とidentifierが悪化している。これは全体平均だけでは見えないため、
次の再現性実験でも非相殺制約に加えてcategory別fractureを確認する必要がある。

## 10. 融合重みからわかること

初期値は最終層を優先しつつ、中間層にも重みを残すよう設定した。学習後のT26は次の重みになった。

|stage|step 55|step 220|解釈|
|---|---|---|---|
|stage 0 最終層|0.705|0.724|学習とともに最終層をやや強めた|
|stage 0 中間層合計|0.295|0.276|約28%を中間層に残した|
|stage 1 最終層|0.644|0.639|ほぼ維持|
|stage 1 中間層合計|0.356|0.361|約36%を中間層に残した|

T26の改善は、最終層を捨てて特定の中間層へ切り替えた結果ではない。最終層を主成分にしながら、複数の中間層を
少量ずつ加える形が維持された。またK1G1/K3G1でも重み分布はT26と近かったが評価結果は異なった。
したがって、softmax重みの大小だけでは品質を説明できず、融合特徴とmain networkの相互作用が重要である。

## 11. 最終判断と次の検証

Encoder複数層融合という要因は、**T26に限定して有望**である。T26 controlを完全に置き換える確定構成ではなく、
長期学習候補へ進める前のquality candidateとする。

次に必要な確認は次の通りである。

1. model初期値だけを43へ変え、データ順を42に固定してT26 fusionの再現性を確認する。
2. model初期値42を固定し、データ順だけを43へ変えて、改善が特定のデータ順に依存しないか確認する。
3. terminal全体指標に加え、活用・固有名詞・複合語・identifierの退行が再現するかを確認する。
4. 2K条件で再現した後、context curriculum候補として8K/32Kでも境界品質と長文位置別driftを評価する。

K1G1/K3G1へ同じ融合をそのまま適用する実験は終了する。これらの構成で複数層を使う場合は、global scalar mixではなく、
boundary stage別・入力別のgate、またはfamily/lexeme integrityを直接守る別の仕組みが必要である。

## 12. 成果物

- 学習run: `/content/drive/MyDrive/hnet_agent_200m_main/runs/boundary_feature_fusion_v1/`
- profile評価: `/content/drive/MyDrive/hnet_agent_200m_main/eval/boundary_feature_fusion_v1_profiles/`
- step 100集計: `/content/drive/MyDrive/hnet_agent_200m_main/analysis/boundary_feature_fusion_v1_step100/`
- step 220集計: `/content/drive/MyDrive/hnet_agent_200m_main/analysis/boundary_feature_fusion_v1_step220/`
- profile指標: `profile_metrics.csv`、`profile_paired_deltas.csv`
- 融合重み: `mix_weight_trajectory.json`
- report用集計: `report_summary.json`

実装commitは `bb67b1f`、共通軌跡集計runnerは `f102afe` と `cd7f154` である。
