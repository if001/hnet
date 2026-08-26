# MoE要因の独立実験

作成日: 2026-08-26

## 1. 共通方針

MoEは総parameter、active parameter、通信、load balancingが増えるため、境界機構の小変更より後に行う。
FFN-MoEとMixer-MoEを分け、最初のrunで両方を有効にしない。MoEはinnermost main networkだけへ入れ、
outer Encoder/Decoderとboundary routerはcontrolと同じにする。

primary comparisonは「1 token/chunk当たりactive parameterと概算FLOPsをdense controlに近づける」とする。
総parameterは別に報告し、約200Mという表現がactive sizeかstored sizeかを曖昧にしない。可能なら小さい
total-parameter-matched controlも補助で作る。

## 2. P4: FFN-MoE

### 2.1 最小pilot

main networkのmixer構成T/K/Gを変えず、FFN部分だけをMoE化する。

- M0: dense FFN control
- M1: 4 routed experts、top-1、capacity factor固定
- M2: M1でcollapseした場合だけ、shared expert 1 + routed experts 4

expert数、top-k、capacity、aux lossを同時探索しない。T26でM0/M1をstep 100まで確認し、安定した仕様だけを
K1G1/K3G1へ適用する。

### 2.2 保存する値

- expert utilization、token/chunk assignment、load balance loss
- dropped/overflow率、routing entropy、expert間表現類似度
- raw bytesだけでなくexpert別に処理したmain chunk数
- category/familyごとのexpert選択分布。ただしcategory別専門化を因果関係として断定しない
- total/active parameter、peak memory、step time。速度は境界順位に混ぜない

### 2.3 判定

境界指標が改善しても、一つのexpertへ90%以上集中する場合はMoE効果とみなさない。T26のintegrity、K1G1のcoverage、
K3G1のcategory P/Cという各anchorの強みを壊さず、少なくとも一つの弱点をterminalで改善することを要求する。

## 3. P5: Mixer-MoE

### 3.1 定義

FFNではなくsequence mixerをexpert化し、chunkごとにT、KDA、Gated MLAなどからactive mixerを選ぶ。
これは固定layoutそのものを学習可能にする案であり、FFN-MoEより境界形成へのinteractionが大きい。

### 3.2 段階

1. X0: 固定layout control。
2. X1: main network中盤の同じ4位置だけをMixer-MoEへ置換し、他22層をanchorのまま残す。
3. X2: X1が安定し効果がある場合だけ8位置へ拡張する。

各MoE位置は同じexpert poolとtop-1 routingを使う。まずT26で実装とroutingを確認し、その後同じ4位置を
K1G1/K3G1へ入れる。位置数とexpert poolを同時に変えない。

### 3.3 注意点

- T/K/G expertはparameter量、state、cache、長文計算量が同一でない。active FLOPsとcache memoryをexpert別に測る。
- token/chunk単位routingが推論時に再現できること、autoregressive cacheが破綻しないことを先にtestする。
- load balancing lossが言語境界を人工的に均等化しないよう、aux loss強度を独立pilotする。
- boundary改善が特定categoryを特定expertへ固定しただけでなく、family variantや文脈変化へ適応するかを見る。
- mixer選択が層ごとにcollapseした場合、学習可能MoEではなく固定mixed layout探索へ戻す。

## 4. MoEの停止条件

- step 100までにrouter collapseが解消しない。
- dense controlよりfracture `+.10`超またはintegrity `-.10`超がstep 60--100で継続する。
- active computeを揃えたときの改善がなく、総parameter増加だけで説明できる。
- OOM/overflow/dropがanchor間で比較不能になる。
- FFN-MoEが無効な場合、理由なくMixer-MoEへ進むのではなく、Mixer-MoE固有の仮説と実装可能性を再確認する。

## 5. 組合せ

MoE winnerをEncoder層融合やfamily lossと組み合わせるのは、MoE単独のstep 220結果が得られた後とする。
最初は`MoE on/off x boundary winner on/off`の2 x 2を一つのanchorで行い、MoEとboundary objectiveのinteractionを
確認する。FFN-MoEとMixer-MoEの併用は本探索の対象外とする。
