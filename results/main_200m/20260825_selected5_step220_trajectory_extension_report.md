# selected5 step 220 境界軌跡延長レポート

作成日: 2026-08-25

## 結論

K1G1、K15-split、K14-middle、K14-late、K3-firstをstep 55から220まで同一条件で延長し、full112 probeを10 stepごとに評価した。step 180--220のterminal windowでは、9指標の単純順位平均はK14-lateが最小だった。しかし、K1G1に対する許容退行幅を先に適用すると、K14-lateを含む全候補がterminalの全5時点で少なくとも1つの制約に違反した。したがって、順位平均だけを根拠に長時間候補を確定することはできない。

- K14-middleはcategory precisionと低fractureの強みが220まで残った。terminalのK1G1同一step比較では両指標を5/5時点で改善した。一方、family coverageとlandmark coverageは5/5時点で退行し、初期に見えた均衡型という評価は長く保たれなかった。
- K14-lateはterminalでcategory P/C/F、landmarkを比較的高く保ち、9指標の順位平均は2.556で最良だった。ただしfamily integrityは`.667`まで低下し、K1G1の`.833`を全terminal時点で下回った。
- K15-splitは文節P/Cを5/5時点で改善し、fractureも5/5時点で低減したが、family coverageとlandmarkは5/5時点で大きく退行した。文節specialistという特徴が最も安定している。
- K3-firstは後半にcategory fractureとfamily integrityが悪化し、terminalでは4つの品質制約を全時点で違反した。今回の長時間候補からは外すのが妥当である。
- K1G1はcategory coverage、family coverage、integrityをterminalで高く保ち、anchorとして依然有効である。

今回の5構成では、K1G1を全面的に置き換える構成は得られなかった。次の構成探索では、K14-middleのcategory precision/低fractureと、K1G1またはK14-lateのfamily coverage/landmark/integrityを同時に維持することを目的にすべきである。

## 1. 実験条件と完了監査

計画: `plan/mid/20260825_selected5_step220_trajectory_extension_plan.md`

- 対象: K1G1、K15-split、K14-middle、K14-late、K3-first
- seed: `(model_init_seed=42, data_order_seed=42, train_runtime_seed=42)`
- resume: 各構成のstep 55 checkpointからoptimizer、data state、RNG stateを復元
- max step / LR horizon: 220 / 220
- probe: category 88文 + family 24文（full112）
- 時系列: native、stage 1、step 10/20/.../220（22時点）
- milestone profile: step 110/165/220のlow/central/high/native
- constraint: `utf8-hard`

resume smokeでstep 55から56へのoptimizer step、micro batch、累積入力byte、RNG復元を確認した。5 runすべてについて、step 55/110/165/220 checkpoint、22個のdense raw、各raw 112 records、step 1--220のtraining metricsを監査した。milestone profileも5構成 x 3時点の15ファイル、各112 records x 4 conditionsが揃っている。

なお、checkpoint読込時にCUDA RNG stateがGPU tensorとして復元される問題が見つかったため、CPU `uint8` tensorへ正規化してから復元する修正を行った。関連unit test 17件は通過し、その後に全延長を実施した。

## 2. 指標と集計

主指標はnative・stage 1の次の9つである。

- category precision (P): 選択した評価可能境界のうち、category probeで説明可能な境界の割合。大きいほどよい。
- category coverage (C): 許容境界のうち選択した割合。大きいほどよい。
- category fracture record occupancy (F): 語彙単位内のfractureを1個以上含むcategory文の割合。小さいほどよい。
- family precision / coverage: 活用・派生・文脈variant 24文に対するP/C。大きいほどよい。
- landmark coverage: family間で共通する重要境界を選択した割合。大きいほどよい。
- family integrity: family文で保護語彙を壊さなかった文の割合。大きいほどよい。
- 文節 precision / coverage: category 88文中のbunsetsu subsetに対するP/C。大きいほどよい。

各step内では境界数をmicro集計し、window内ではstepを等重みで平均した。precisionは選択境界が0件のstepで未定義とし、0に置換していない。windowはinitial=10--50、medium=60--110、late=120--170、terminal=180--220である。

K1G1基準のprimary許容退行幅は、fracture `K1G1 + .05`以下、family integrity/coverage/landmarkは各`K1G1 - .05`以上である。relaxedは各幅を`.10`とした。順位は制約を通過した候補間の補助指標であり、制約違反を他指標の順位で相殺しない。

## 3. Window別の軌跡

列順はcategory P/C/F、family P/C、landmark C、integrity、文節P/Cである。

| window | 構成 | category P/C/F | family P/C | landmark | integrity | 文節 P/C |
| --- | --- | --- | --- | ---: | ---: | --- |
| initial | K1G1 | .332/.226/.218 | .484/.442 | .492 | .917 | .117/.165 |
|  | K15-split | .359/.208/.132 | .560/.358 | .425 | .892 | .364/.443 |
|  | K14-middle | .425/.238/.195 | .470/.552 | .633 | .833 | .184/.252 |
|  | K14-late | .414/.249/.268 | .509/.545 | .650 | .775 | .186/.261 |
|  | K3-first | .443/.215/.227 | .784/.461 | .633 | .792 | .273/.270 |
| medium | K1G1 | .350/.383/.409 | .646/.788 | .861 | .882 | .138/.261 |
|  | K15-split | .385/.290/.267 | .632/.475 | .528 | .785 | .321/.500 |
|  | K14-middle | .437/.330/.136 | .563/.758 | .840 | .917 | .154/.261 |
|  | K14-late | .439/.330/.189 | .589/.813 | .819 | .896 | .164/.283 |
|  | K3-first | .429/.306/.375 | .592/.540 | .743 | .569 | .243/.370 |
| late | K1G1 | .427/.417/.362 | .528/.717 | .819 | .903 | .179/.341 |
|  | K15-split | .397/.316/.350 | .560/.364 | .472 | .729 | .306/.478 |
|  | K14-middle | .474/.360/.254 | .573/.621 | .681 | .812 | .216/.341 |
|  | K14-late | .449/.435/.407 | .527/.833 | .931 | .694 | .217/.391 |
|  | K3-first | .410/.328/.470 | .513/.399 | .549 | .528 | .213/.362 |
| terminal | K1G1 | .425/.464/.425 | .449/.667 | .850 | .833 | .159/.322 |
|  | K15-split | .409/.332/.375 | .618/.345 | .458 | .833 | .304/.478 |
|  | K14-middle | .506/.379/.295 | .540/.442 | .533 | .775 | .183/.252 |
|  | K14-late | .474/.446/.345 | .457/.661 | .867 | .667 | .203/.287 |
|  | K3-first | .394/.320/.527 | .589/.479 | .658 | .550 | .173/.296 |

形成は単調ではない。K14-middleはmediumでfamily C/landmark/integrityが`.758/.840/.917`だったが、terminalでは`.442/.533/.775`へ低下した。一方、category P/Fは`.437/.136`から`.506/.295`となり、fractureは悪化したものの他構成およびK1G1より低い。K14-lateはlateからterminalにcategory FとPが改善したが、integrityは`.694`から`.667`へ低下した。K15-splitはterminalの文節Cが全5点で`.478`と安定する一方、family C/landmarkは低い水準に固定された。

隣接step間の境界signature変化率のterminal平均は、K1G1 `.409`、K15-split `.330`、K14-middle `.364`、K14-late `.375`、K3-first `.338`だった。これは変化量の診断であり、高低だけで品質を決めない。

## 4. Terminal windowの比較

### 4.1 同一step paired差

表はK1G1との差の中央値と、改善したstep数（全5時点）を示す。Fは負が改善、それ以外は正が改善である。

| 構成 | category P | category C | category F | family C | landmark | integrity | 文節 P | 文節 C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K15-split | -.014 (0/5) | -.137 (0/5) | -.057 (5/5) | -.333 (0/5) | -.417 (0/5) | .000 (0/5) | +.153 (5/5) | +.174 (5/5) |
| K14-middle | +.083 (5/5) | -.077 (0/5) | -.136 (5/5) | -.242 (0/5) | -.333 (0/5) | -.042 (0/5) | +.032 (4/5) | -.043 (0/5) |
| K14-late | +.047 (5/5) | -.013 (0/5) | -.102 (5/5) | -.030 (2/5) | .000 (2/5) | -.167 (0/5) | +.044 (5/5) | -.043 (0/5) |
| K3-first | -.035 (0/5) | -.141 (0/5) | +.102 (0/5) | -.182 (0/5) | -.208 (0/5) | -.292 (0/5) | +.023 (4/5) | .000 (0/5) |

同一step比較の意図は同じraw学習量に対する品質を比較することであり、境界形成の位相が同じだとは仮定しない。terminalで方向が5時点連続する差は、初期の形成速度だけでは説明しにくい持続的なtrade-offを示す。ただし220 stepが収束点であるとは限らない。

### 4.2 許容退行幅

各セルは違反step数/22、括弧内はrelaxed基準である。`any`は4条件のいずれかに違反したstep数である。

| 構成 | fracture | integrity | family C | landmark | any |
| --- | ---: | ---: | ---: | ---: | ---: |
| K15-split | 1 (0) | 10 (8) | 18 (18) | 19 (19) | 21 (21) |
| K14-middle | 2 (0) | 8 (5) | 12 (8) | 11 (7) | 17 (12) |
| K14-late | 7 (3) | 15 (12) | 5 (1) | 3 (1) | 20 (15) |
| K3-first | 12 (7) | 20 (20) | 18 (16) | 14 (14) | 21 (21) |

terminalだけを見ると全候補が5/5時点で`any`違反だった。K15-splitはfamily Cとlandmark、K14-middleも同じ2軸、K14-lateはintegrity、K3-firstは4軸すべてが主因である。K14-middleは22時点全体では最も違反が少ないが、後半でfamily側の退行が増えた。

### 4.3 順位は補助診断

terminal平均の9指標を同順位平均で集計すると、K14-late `2.556`、K1G1 `2.722`、K15-split `2.833`、K14-middle `3.000`、K3-first `3.889`となった。K14-lateが全軸で極端に下位になりにくいことは確認できる。しかしintegrityの許容退行を破っているため、順位平均の最良をそのまま採用判断にはしない。

## 5. Category coverageを揃えた比較

異なるstep間でcategory coverage差が許容幅以内の全組を作り、category P/Fとfamily integrityの差の中央値を求めた。主許容幅`.03`では次の通りである。

| 構成 | pair数 | category P差 | category F差 | integrity差 |
| --- | ---: | ---: | ---: | ---: |
| K15-split | 43 | +.093 | -.045 | -.125 |
| K14-middle | 101 | +.133 | -.102 | -.125 |
| K14-late | 107 | +.053 | -.011 | -.167 |
| K3-first | 38 | +.098 | +.108 | -.375 |

許容幅`.02/.05`でも符号はすべて同じだった。K14-middleは同程度のcategory coverageでprecisionを上げfractureを下げる傾向が最も明確だが、family integrityの低下は残る。K3-firstはprecisionが上がってもfractureとintegrityが同時に悪化する。

この比較のpairは互いに独立ではなく、同じstepが複数の相手と組になる。従って有意差検定や勝率として扱わず、coverage量の違いだけでは説明できない方向性の診断として使用する。

## 6. Milestone profile頑健性

step 110/165/220 x low/central/high/nativeの12条件について、K1G1より改善した条件数を示す。FはK1G1より小さい条件を勝ちとした。

| 構成 | category P/C/F | family P/C | landmark | integrity | 文節 P/C |
| --- | --- | --- | --- | ---: | ---: | --- |
| K15-split | 1/0/5 | 2/0 | 0 | 0 | 12/12 |
| K14-middle | 7/1/10 | 3/4 | 4 | 0 | 4/2 |
| K14-late | 7/1/7 | 1/4 | 4 | 0 | 4/2 |
| K3-first | 4/1/2 | 3/2 | 2 | 1 | 10/7 |

profileを変えてもK15-splitの文節specialist性は12/12で残る。K14-middleの低fractureは10/12、category precisionは7/12で残るが、integrityは0/12である。K14-lateもcategory Pは7/12だがintegrityは0/12で、native terminalのtrade-offと整合する。

## 7. 構成別の特徴と次の判断

1. K1G1はcoverage/integrity anchorとして残す。terminalのcategory C、family C、integrityが強く、候補構成の退行を測る基準として機能する。
2. K14-middleはprecision・fracture specialistとして残す。良さはcoverageを揃えても観測されるが、family C/landmarkの後半低下を解消する必要がある。
3. K14-lateはbalanced candidateとして残す余地がある。terminal順位とlandmarkはよいが、family integrityを回復できる構成変更が前提となる。
4. K15-splitは文節specialistとして対照に残す。全面候補ではなく、文節P/C改善の由来を切り出すためのablationに向く。
5. K3-firstは今回の候補探索から除外する。後半のfracture/integrity退行が大きい。

次の小規模構成探索では、K14-middleのKDA量・位置をそのまま増減するだけでなく、後半で失われたfamily境界を維持する方向を明示的に比較する。具体的にはK14-middleを基準に、K1G1区間を後半へ戻す構成、またはK14-lateの後半配置を一部だけ取り入れた構成を少数作り、まず110/165/220のfamily C・landmark・integrity制約を通過するかを確認する。文節改善を主目的にしない限り、K15-split量を増やす優先度は低い。

## 8. 限界

- seedは`i42/d42/r42`の1組だけで、weight初期値・data order・runtime乱数に対する再現性は未確認である。
- 220 stepは長期収束ではなく、今回のデータ量・LR schedule内の軌跡である。
- full112は説明可能な境界仮説を測る有限probeであり、main networkの最終lossや生成品質を直接測っていない。
- matched-coverage比較はcross-stepの非独立pairであり、効果量の記述に限定する。
- profile条件は外部から境界budgetを変える診断で、nativeの代替ではない。

構成別22時点の未丸め値は`results/main_200m/20260825_selected5_step220_per_step_metrics.md`にCSVとして分離した。
