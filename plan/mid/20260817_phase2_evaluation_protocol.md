# Phase 2 評価・集約プロトコル

作成日: 2026-08-17

## 目的

Phase 2 の T26、K1T1、128k tokenizer baseline 各2 seedと、共通SFT後の
各checkpointを、結果を見る前に固定した規則で比較する。Phase 2は200M本学習へ
進めるH-Netを1構成へ絞るためのscreeningであり、H-Net対tokenizerの最終結論には
しない。

## 固定する評価

1. 125M raw bytesごとの4 checkpointについて、validation BPB、圧縮率、
   raw bytes/sec、peak memoryのcurveをseed別・平均で集計する。
2. 最終pretraining checkpointへ`configs/boundary_probe_v1.json`の全140 recordsを
   与え、同一文字列に対するBPBを日本語、構造化入力、code、長文等のcategory別に
   集計する。tokenizer baselineもraw UTF-8 byte数で正規化する。
3. T26とK1T1の最終checkpointへLearned、Fixed、Random、Morph、1 byte相当の
   left/right shiftを適用する。Randomはseed 0から4を固定する。
4. 同一pretraining seed間でT26からK1T1、K1T1からT26へのboundary transferを行う。
   境界数が異なる場合はtargetの境界数を保ち、source境界を優先位置として使う。
5. 共通SFT後に、単一tool選択とJSON引数生成の固定Level A proxyをgreedy decodeで
   評価する。JSON妥当率、tool accuracy、引数exact matchを記録する。

## 事前判定規則

- Phase 1で固定した実用同等幅は`0.009626833718514239 BPB`とする。
- K1T1を選ぶ条件は、2 seed平均でT26を同等幅より大きく改善するか、agent proxyの
  主要指標を10 percentage points以上改善し、その方向が両seedで一致することとする。
- 上記を満たしても、category BPBの重大退行、学習不安定、throughputまたはmemoryで
  明確に支配される場合はK1T1を選ばない。
- 差が同等幅内、指標間で不一致、または全モデルがfloorに近い場合は、単純で
  throughputの高いT26を選ぶ。
- tokenizerとの比較は中規模での方向確認に留め、200Mでの成功を断定しない。

## 既知の制約

- Phase 2 loggerには実測FLOPs/byteが保存されていない。parameter数はFLOPsの代理に
  しないため、今回のPareto評価はBPB、raw bytes/sec、peak memoryで行い、
  FLOPs/byteは計測基盤の未完了項目として報告する。
- 4 checkpointのvalidationは共通mixed validationであり、category別curveではない。
  category別比較は最終checkpointの固定probeで補う。
- SFT seedは42に固定されており、pretraining seedのみ2つである。Phase 5で要求する
  SFT 3 seed分散の代替にはならない。

