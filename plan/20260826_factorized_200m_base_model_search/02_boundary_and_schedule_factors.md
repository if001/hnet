# 境界機構・学習schedule要因の独立実験

作成日: 2026-08-26

## 1. P1: Encoder複数層の境界特徴融合

### 1.1 対象とする変更

ここでは「Multi-scale boundary」という語を、階層stageの追加ではなく、各outer stageのEncoder複数層を
融合して現在のboundary routerへ渡す変更として扱う。main-network中間状態を同一forwardの現在境界へ直接
使う方法は循環依存になるため、最初の候補にしない。

現行controlはEncoder最終層`h^L`だけから隣接表現のcosine dissimilarityを計算する。variantは各Encoder層を
同じ次元へ正規化・射影し、学習可能なglobal scalarで融合する。

`z_i = sum_l softmax(alpha_l) * P_l(LN(h_i^l))`

routerのcosine計算、hard mask、compression objectiveは変更しない。

### 1.2 最初の比較

- F0: final layer only（現行control）
- F1: stage別learned scalar mix。stage 0とstage 1で別の`alpha`を持つ

F1で一つの層へ完全collapseする場合だけ、layer dropoutまたは弱いentropy regularizationをF2として別実験する。
input-dependent gating、main-network teacher、two-pass refinementはこのscreeningへ混ぜない。

### 1.3 固定・監査

- T26/K1G1/K3G1のmain network、parameter初期値、data順を固定する。
- fusion射影以外のparameterをcontrolと同じ初期値にする。
- alphaの初期値は最終層優位と一様の2案をpilotで比較するが、full112 testで選ばない。synthetic/devで決める。
- stage別alpha、gradient norm、層ごとの寄与、router marginを10 stepごとに保存する。

### 1.4 仮説と判定

- T26のintegrity/family Pを壊さずcategory Cを上げられるか。
- K1G1のcoverageを維持してfractureを下げられるか。
- K3G1のcategory P/Cを維持してfamily/landmark/integrityを回復できるか。
- 改善が単に境界数を増減した結果でないことをmatched-coverageで確認する。

## 2. P2: Family consistency loss

### 2.1 目的

活用・派生・文脈variant間で再利用可能なlandmarkを保ち、保護語彙内部のfractureを減らす。full112を教師に
使わず、training corpusから作るfamily pairと独立dev/test splitを使う。

### 2.2 lossの最小仕様

最初は三つを一度に入れず、次の順で分離する。

1. C1 landmark consistency: pair間で対応が明確な位置のboundary probabilityをJS divergenceまたはMSEで揃える。
2. C2 integrity margin: 保護span内部のboundary probabilityがspan外landmarkより一定margin低くなるようにする。
3. C3 C1+C2: C1/C2が単独で通過した場合だけ組み合わせる。

主lossは`LM + ratio + lambda_family * L_family`とし、lambdaはtraining stabilityと独立devでlow/mediumの2点を
pilotする。full112 testの最大値でlambdaを選ばない。

### 2.3 データ交絡を避ける

- controlとvariantは同じLM batchを同順序で使う。
- auxiliary pairは別iteratorとし、LM token数・main data cursorを変えない。
- primary controlは同じauxiliary pairを同じ順でforwardするsham条件とし、`lambda_family=0`にする。これにより
  auxiliary forwardが消費するruntime RNGや実行順の差をloss効果へ混ぜない。
- auxiliary raw bytes、category分布、言語比率をmanifestへ保存する。
- 人名・数値だけなど単一familyに偏らせず、活用、助動詞、助詞、複合語、structured、identifierを均衡化する。
- evaluation probeと文字列一致するpairは除外する。

### 2.4 判定

family C/landmark/integrityの改善を主目的とするが、固定境界化を避けるためcategory P/C、文節、context-dependent
familyも確認する。transition低下だけを成功とせず、文脈で変わるべき境界が残るかを分割galleryで監査する。

## 3. P3: encoder/decoder-only warmup

### 3.1 目的

outer-stageのEncoder/Decoderとrouterを短時間先行学習してからjoint trainingへ移ることで、初期の境界形成を
安定化できるかを見る。「初期境界が固定される」ことは仮定せず、joint開始後の揺れ、適応、最終windowを評価する。

### 3.2 parameter scope

warmup中に更新するものを明示する。

- outer-stage encoder、decoder、routing module、dechunk/residual projection
- compression ratio lossに関係するparameter
- innermost main networkはfreeze

LM head/embeddingを更新するかは実装依存を監査し、W1では必要最小限に固定する。optimizer stateはwarmup用とjoint用を
分け、joint開始時にLR scheduleをstep 0から開始する。

### 3.3 三つの対照

warmupは追加計算とデータcursorを交絡させやすいため、次を分ける。

- W0: warmupなし、joint 220 step。
- W1: E/D warmup 20 step後、joint data cursorを共通開始位置へ戻してjoint 220 step。preconditioning効果を見るprimary。
- W2: E/D warmup 20 + joint 200 step。総optimizer update数をW0と揃えるcompute-control。

W1は総計算量が増えるためW0への勝利だけで効率改善とは言わない。W2はmain networkの学習量が20 step少ないため、
W1/W2を併記して解釈する。warmup長20はpilot値で、10/40への拡張はW1で効果がある場合だけ行う。
joint開始時にはruntime RNGも共通stateへ戻し、warmupによるRNG消費を学習時乱数差へ混ぜない。

### 3.4 評価

- warmup中も10 stepごとにboundaryを評価するが、joint stepとは別軸で表示する。
- joint開始を0として10--220の軌跡をcontrolと比較する。
- boundary形成の早期化だけか、terminal品質も改善するかを区別する。
- warmup終了直後とjoint開始後の最大drawdown、回復step、alpha/router gradientを保存する。

## 4. P1--P3の組合せ順

1. F1、C1/C2、W1をそれぞれ独立にscreenする。
2. F1とfamily lossのwinnerだけ、代表anchorで2 x 2を行う。
3. 加法的または相補的なら3 anchorsへ展開する。
4. warmupは上記winnerに最後に追加し、warmup単独との差を見る。
5. 三要因全部を一度に入れたrunを最初から作らない。

P1/P2は境界決定そのもの、P3は学習scheduleへ作用する。同じ改善が出ても、到達品質、形成速度、持続性を別々に報告する。
