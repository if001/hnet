# 2K→8K→32K context curriculum独立計画

作成日: 2026-08-26

## 1. 位置づけ

KDAは長いcontextを対象にするため、2Kだけの境界probeでT26との価値を判断できない。一方、context lengthを
変えるとpacking、batch、累積bytes、memory、optimizer dynamicsが同時に変わる。このため、P1--P5の短context
探索とは別runにし、3 anchorsと短context上位最大2候補だけを対象にする。

## 2. 比較条件

最小比較は次の三つである。

- L0 fixed-2K: 全期間2K。既存短context挙動との接続用。
- L1 curriculum: 2K→8K→32K。
- L2 direct-transition: 2K phase後に8Kを飛ばして32Kへ移るcontrol。8K中間段階の効果を見る。
- L3 direct-32K: 初期状態から32Kで学習する補助control。予算が許す最終2候補だけで行う。

primaryの総update数を600とする場合、L0は2Kを600、L1は各contextを200、L2は2Kを200後に32Kを400、
L3は32Kを600 updateとする。1 update当たりtarget raw bytesも揃え、単なる総データ量差を避ける。

L2/L3を全候補で実行すると高価なため、L1で残ったT26とKDA候補1つに限定してよい。T26、K1G1、K3G1のL1は、
KDA量と長文効果を解釈する基準として原則残す。

## 3. curriculum単位

context phaseごとに少なくとも200 optimizer updatesを確保し、境界軌跡を観測する。

| phase | context | updates | dense評価 |
| --- | ---: | ---: | --- |
| A | 2K | 200 | 10 step間隔 |
| B | 8K | 200 | 10 step間隔、移行直後は1/5 stepも追加 |
| C | 32K | 200 | 10 step間隔、移行直後は1/5 stepも追加 |

これは600 stepの長期候補確認であり、P1--P5の220 step screeningとは別である。GPU予算で短縮する場合も各phase
100 step未満にはせず、「32Kまで到達した」だけで良否を判断しない。

## 4. bytesとdata順の統一

- micro batchをcontextに反比例させ、gradient accumulationで1 optimizer update当たりのtarget raw bytesを揃える。
- memory都合で完全一致しない場合、累積raw bytesを主横軸、optimizer updateを副横軸にする。
- 同じdocument streamとbyte offset列から2K/8K/32K windowを作り、長さ変更で文書分布やshuffle seedを変えない。
- document境界を跨ぐpacking、padding、truncation規則を固定し、各phaseの有効token率を保存する。
- LRをstep基準にするかbyte基準にするかを混ぜない。primaryは累積raw-byte scheduleとし、step scheduleを補助controlにする。

### 4.1 実行時に固定したcanonical stream

- 32,768 bytesをcanonical blockとし、2Kでは16分割、8Kでは4分割、32Kでは分割せずに読む。
- micro batchは2K=`16`、8K=`4`、32K=`1`、gradient accumulationは全条件`32`とする。これにより
  1 optimizer updateは全phaseで正確に1,048,576 input bytesになる。
- shuffleは個別sequenceではなくcanonical block単位で一度だけ行う。contextを変更してもblock hash、shard、byte offset列を
  変えない。
- 600 updateには19,200 canonical block、約629M input bytesが必要である。corpus反復を避けるため、既存8:1:1分布を維持した
  `SOURCES_JA8_EN1_CODE1_CONTEXT_1B`をP6専用にpackし、必要block数を開始前に監査する。
- phase Aはblock 0--6,399、phase Bは6,400--12,799、phase Cは12,800--19,199を使う。checkpointには消費済み
  canonical block数を保存し、phase移行と中断再開の双方で同じstream位置から続ける。

## 5. 評価

### 5.1 既存の短probe

full112は各phaseで継続し、context拡張で既存の言語境界品質が崩れないかを見る。phase移行前後のdrawdownと回復stepを
重点的に記録する。

### 5.2 長文probe

full112の単純paddingでは長context能力を測れないため、評価専用の長文probeを別versionで作る。

- 同じ語・活用familyが長距離に再出現するfamily consistency
- 文書前半の定義、identifier、tool fieldを後半で再利用するcross-span consistency
- 先頭・25%・50%・75%・末尾に同じlocal probeを置くposition robustness
- distractor数を増やしたときのboundary drift
- 章、段落、文、文節、語彙の複数距離におけるboundary維持
- 日本語、code、JSON/tool、混在文を分離したcategory結果

長文probeはtraining curriculumのデータから独立させ、2K/8K/32Kで共通のlocal spanを比較できるようにする。

### 5.3 quality / cost

- validation BPBまたは長文task値は参考だが、明確な崩壊検出には使う。
- full112と長文boundaryの9指標、位置別drawdown、compressionを主品質とする。
- peak GPU memory、tokens/sec、bytes/sec、active main chunks、cache sizeを測る。
- 現在のunfused等の実装差がある場合、wall-clockだけでarchitectureを選ばず、理論量とprofileを分ける。

## 6. phase移行の判定

2K→8K、8K→32Kへ自動的に進めず、次を満たすことを確認する。

- 直前phaseのterminal windowでNaN、compression collapse、継続的なrelaxed制約違反がない。
- phase移行後20 step以内の一時悪化と、100/200 stepでも残る構造的退行を分ける。
- T26がmemory上実行不能な場合、OOMを品質0として集計しない。実行可能上限とresource制約を別結果にする。
- KDA候補が長文costを改善しても、fracture/integrityの許容範囲を超える場合はquality candidateにはしない。

## 7. 最終比較

各構成について次の二軸を出す。

1. boundary quality trajectory: 短probeと長文probeのterminal/q20/q80、移行drawdown、回復。
2. long-context efficiency frontier: qualityに対するpeak memory、bytes/sec、active compute。

T26、K1G1、K3G1の比較により、Transformer-only、低KDA、高KDAの差を確認する。短context winnerを加える場合も、
その改善が2Kだけで消えるのか、8K/32Kでも維持されるのかを明示する。最終的な長期候補は単一総合点ではなく、
非相殺制約を通過したPareto frontierから選ぶ。
