# selected7 step 220 境界軌跡延長レポート

作成日: 2026-08-25

追補: K3G1とT26を同条件でstep 220まで延長し、section 9以降に7構成の再集計を追加した。section 1--8は初回5構成の結果を保持しており、候補間順位についてはsection 9以降の7構成版を優先する。

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

## 9. K3G1・T26ベースライン追加実験

### 9.1 追加する理由

T26はmain networkがTransformer 26層だけの基準構成である。K3G1はKDAを多く含む基準構成であり、K14-middle、K14-late、K15-splitなどはKDAの量・位置を変えて、TransformerとKDAの中間を探索する構成である。このため、K1G1だけでなく次の2つを比較基準に加える必要がある。

- T26: KDAを含めないTransformer基準
- K3G1: KDAを高用量で含む基準

Kimi Delta Attentionは長い系列で計算量を抑えることを狙うため、最終的にはT26との長文・計算量比較が重要である。ただし今回のprobeと学習はsequence length 2048であり、KDAの長文計算量優位や長文精度を直接検証する実験ではない。ここでは、同じraw学習量を与えたときに境界品質がT26からどう変わるかを確認する。

### 9.2 条件と監査

K3G1とT26も初回5構成と同じ`i42/d42/r42`、データ順、LR schedule、`utf8-hard`、full112 probeを用い、step 55 checkpointからoptimizer・data state・RNG stateを復元してstep 220まで延長した。

- K3G1: 2026-08-25 09:45--10:47 UTC、return code 0
- T26: 2026-08-25 10:47--11:15 UTC、return code 0
- 各構成: step 55/110/165/220 checkpoint、native 22時点、各112 records
- milestone profile: step 110/165/220 x 4 conditions、合計6ファイル

学習時間は実装上の速度差を含む参考値であり、選定指標には使わない。特にKDA側のfused実装差や長系列での漸近的な計算量差をこの時間だけから評価しない。

## 10. K3G1とT26の時間軌跡

列順はcategory P/C/F、family P/C、landmark C、integrity、文節P/Cである。

| window | 構成 | category P/C/F | family P/C | landmark | integrity | 文節 P/C |
| --- | --- | --- | --- | ---: | ---: | --- |
| initial | K3G1 | .372/.208/.345 | .524/.364 | .425 | .742 | .150/.139 |
|  | T26 | .397/.172/.225 | .682/.442 | .600 | .842 | .103/.087 |
| medium | K3G1 | .427/.340/.485 | .595/.742 | .917 | .438 | .157/.217 |
|  | T26 | .340/.304/.441 | .776/.677 | .903 | .889 | .127/.188 |
| late | K3G1 | .434/.328/.434 | .447/.465 | .542 | .618 | .126/.174 |
|  | T26 | .399/.293/.333 | .765/.611 | .743 | .917 | .119/.174 |
| terminal | K3G1 | .442/.344/.436 | .353/.309 | .283 | .692 | .076/.096 |
|  | T26 | .396/.274/.302 | .722/.545 | .667 | .908 | .143/.217 |

K3G1はT26よりcategory P/Cが高い。特にterminalではPが`.442`対`.396`、Cが`.344`対`.274`である。一方、category Fは`.436`対`.302`で悪く、family P/C、landmark、integrity、文節P/CもT26を下回った。すなわち、高KDA構成はこの学習範囲でcategory境界を多く説明できる方向へ動くが、語彙fractureとfamily・文節境界の品質を同時に保てていない。

K3G1のmediumではfamily C/landmarkが`.742/.917`まで形成されたが、terminalでは`.309/.283`へ低下した。T26はmedium以降のintegrityを`.889/.917/.908`と高く保ち、family precisionもterminalで`.722`だった。K3G1のterminal transition `.259`は7構成で最小だが、低いfamily/文節品質の分割が安定している可能性があり、小さいtransition自体を改善とは解釈しない。

## 11. 7構成でのterminal再比較

### 11.1 Terminal平均と順位

| 構成 | category P/C/F | family P/C | landmark | integrity | 文節 P/C | 9指標順位平均 |
| --- | --- | --- | ---: | ---: | --- | ---: |
| K14-late | .474/.446/.345 | .457/.661 | .867 | .667 | .203/.287 | 3.000 |
| K1G1 | .425/.464/.425 | .449/.667 | .850 | .833 | .159/.322 | 3.167 |
| K14-middle | .506/.379/.295 | .540/.442 | .533 | .775 | .183/.252 | 3.444 |
| K15-split | .409/.332/.375 | .618/.345 | .458 | .833 | .304/.478 | 3.611 |
| T26 | .396/.274/.302 | .722/.545 | .667 | .908 | .143/.217 | 3.889 |
| K3-first | .394/.320/.527 | .589/.479 | .658 | .550 | .173/.296 | 5.000 |
| K3G1 | .442/.344/.436 | .353/.309 | .283 | .692 | .076/.096 | 5.889 |

T26を加えてもK14-lateの順位平均は最良である。ただし既述のとおりintegrity退行があるため、順位だけで採用しない。K3G1はcategory P/CではT26を上回るが、他軸の退行が大きく最下位になった。これはKDAを含めること自体が悪いという意味ではなく、KDAの高用量・配置が今回のboundary objectiveでは偏ったことを示す。

### 11.2 T26との同一step paired比較

表はterminalの`候補 - T26`中央値と改善step数/5である。Fは負が改善、他は正が改善である。

| 構成 | category P | category C | category F | family P | family C | landmark | integrity | 文節 P | 文節 C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K3G1 | +.036 (5/5) | +.073 (5/5) | +.136 (0/5) | -.356 (0/5) | -.212 (0/5) | -.375 (0/5) | -.250 (0/5) | -.056 (0/5) | -.087 (0/5) |
| K15-split | +.014 (4/5) | +.056 (5/5) | +.068 (0/5) | -.151 (1/5) | -.182 (0/5) | -.208 (0/5) | -.083 (0/5) | +.154 (5/5) | +.261 (5/5) |
| K14-middle | +.109 (5/5) | +.107 (5/5) | .000 (2/5) | -.192 (0/5) | -.121 (0/5) | -.167 (0/5) | -.125 (0/5) | +.045 (4/5) | +.043 (3/5) |
| K14-late | +.077 (5/5) | +.184 (5/5) | +.034 (0/5) | -.292 (0/5) | +.091 (5/5) | +.208 (5/5) | -.250 (0/5) | +.069 (5/5) | +.087 (4/5) |

K14-middleはK3G1よりKDA量を減らすことで、K3G1で大きかったfractureをT26とほぼ同程度まで改善しつつ、T26よりcategory P/Cを上げている。しかしfamily P/C、landmark、integrityはT26に届かない。K14-lateはT26よりcategory P/C、family C、landmark、文節P/Cを一貫して上げるが、fractureが僅かに増え、integrityが大幅に低下する。したがって、K14-lateはT26からcoverageを得る代わりにintegrityを失う構成、K14-middleはprecision/fractureを重視する構成と整理できる。

K15-splitはT26より文節を明確に改善するが、familyとintegrityを失う。K3G1からKDA量・位置を変えた構成がK3G1の弱点を一部改善していることは確認できるものの、T26が持つfamily integrityを維持する組合せにはまだ到達していない。

## 12. Coverage整合・profile頑健性

K1G1とのcategory coverage差`.03`以内で比較すると、K3G1はcategory P `+.110`、F `+.102`、integrity `-.292`、T26はP `+.078`、F `+.057`、integrity `.000`だった。いずれもprecision上昇だけでなくfracture増加を伴うが、K3G1のtrade-offが大きい。

step 110/165/220 x 4 conditionsの12 profileで、K3G1がT26より改善した数はcategory P/C/F=`10/12/5`、family P/C=`2/1`、landmark=`0`、integrity=`5`、文節P/C=`10/10`だった。K3G1のcategory coverage優位は全profileで残る一方、family/landmark優位はほぼない。nativeだけの偶然ではなく、境界budgetを変えても同じ方向のbiasが観測される。

## 13. 更新した判断

1. T26をK1G1と並ぶ基準として残す。T26はterminalのfamily precisionとintegrityが最も強く、KDA混合による退行を検出する基準になる。
2. K3G1を高KDA dosage基準として残すが、長時間候補には直接選ばない。category P/Cは高いものの、fracture、family、文節の退行が大きい。
3. K14-middleはK3G1からの改善として有効である。K3G1のfractureを抑え、T26よりcategory P/Cを上げるが、family/landmarkを回復する余地がある。
4. K14-lateはcoverage・landmark重視の中間点である。T26より複数coverageを改善する一方、integrityを許容範囲へ戻す必要がある。
5. 次の探索では「K3G1を何層含めるか」だけでなく、T26の高integrityを壊すKDA位置を特定する。K14-middleとK14-lateの差から、同じKDA量でも位置によりfamily C/landmarkとintegrityのtrade-offが変わるため、KDA dosageとpositionを分離して比較する。

今回の結果だけではKDAの長文精度・計算量上の目的を否定も確認もできない。main-network候補を絞った後、より長いsequence lengthでT26との品質対計算量を測る実験が別途必要である。

## 14. TransformerからKDA/Gated MLAへ置換する量・位置の考察

### 14.1 比較している構成

ここでいう「置換」は、T26のmain blockにある26個のTransformer層を、KDA (`K`) とGated MLA (`G`) の26層へ置き換えることを指す。K3G1は26層すべてがKDAという意味ではなく、main blockからTransformerを除き、`K=19/G=7`にした高KDA構成である。

| 構成 | T/K/G | 主な配置 | 比較上の役割 |
| --- | --- | --- | --- |
| T26 | 26/0/0 | Transformerのみ | 非置換baseline |
| K1G1 | 0/13/13 | `K1G1`を均等反復 | 最小KDA量の部分置換baseline |
| K14-middle | 0/14/12 | K3連続区間を中央に1個 | K14の位置control |
| K14-late | 0/14/12 | K3連続区間を後方に1個 | K14の位置control |
| K15-split | 0/15/11 | K3連続区間を前後に分割 | 中間dosage・分割配置 |
| K3-first | 0/16/10 | K3連続区間を前方に3個 | 高めdosage・前方集中 |
| K3G1 | 0/19/7 | `K3G1`中心 | main block完全置換・高KDA baseline |

K14-middleとK14-lateはparameter数とK/G総数が同じで、主な差がK3連続区間の位置である。そのため、この2構成間の差はKDA量では説明できず、配置に関連する差として読むことができる。一方、T26からK1G1、K14、K15、K16、K3G1への並びには量と配置が同時に変わる箇所があり、純粋なdosage曲線ではない。

### 14.2 各評価値からわかること

#### Category precision

terminalのcategory PはT26 `.396`、K1G1 `.425`、K14-middle `.506`、K14-late `.474`、K15-split `.409`、K3-first `.394`、K3G1 `.442`だった。Transformerを一部置換すると常に上昇するわけでも、KDA量に比例して上昇するわけでもない。最大は中間量のK14-middleであり、K3G1ではそこから低下する。

したがって、category Pには「KDAが多いほどよい」という単調関係はなく、中央に少量の連続KDA区間を置くK14-middleが、不要な境界を増やさず説明可能境界を選ぶ点でsweet spotになっている。K14-middleとK14-lateの`.032`差から、同じK14でも位置がprecisionを変えることもわかる。

#### Category coverage

category CはT26 `.274`に対し、すべての部分・完全置換構成が`.320`以上だった。K1G1 `.464`、K14-late `.446`が特に高く、K3G1も`.344`でT26を上回る。milestone 12 profileでもK3G1はT26に12/12条件で勝っており、KDA/Gated MLAへの置換は、少なくとも今回のprobeでは説明可能なcategory境界を拾いやすくする方向と関連している。

ただし、coverage最大は高KDAのK3G1ではなくK1G1である。従ってcoverage増加もdosageに比例せず、KDAとGated MLAを交互に置くこと、または連続KDA区間の位置が重要である。

#### Category fracture

FはT26 `.302`、K14-middle `.295`が低く、K14-late `.345`、K15-split `.375`、K1G1 `.425`、K3G1 `.436`、K3-first `.527`の順に悪化した。高KDA量では語彙内部を切るrecordが増える傾向はあるが、K1G1とK15の逆転や、K16がK19より悪いことから単調ではない。

重要なのは、同じK14でもmiddle `.295`とlate `.345`に差がある点である。KDA量を14まで増やすこと自体がfractureを増やすのではなく、中央配置ではT26と同等以上に抑えられる。fractureはdosageよりpositionと連続区間の作り方に敏感な指標と考えられる。

#### Family precision

family PはT26 `.722`が最高で、K15-split `.618`、K3-first `.589`、K14-middle `.540`、K14-late `.457`、K1G1 `.449`、K3G1 `.353`だった。全体として高KDAのK3G1で大きく低下し、T26が最も選択境界をfamily規則で説明しやすい。

部分置換で必ず単調低下するわけではなく、K15/K16はK13/K14-lateより高い。このためfamily PもKDA量だけでは決まらないが、T26のTransformer表現がfamily内の不要境界を抑える強いbaselineになっていることは明確である。

#### Family coverage

family CはK1G1 `.667`とK14-late `.661`がT26 `.545`を上回る一方、K14-middle `.442`、K15-split `.345`、K3G1 `.309`は下回った。同じK14でmiddleとlateに`.219`の差があり、今回の9指標中でも配置依存が特に大きい。

後方K3区間を持つK14-lateはfamily境界を広く拾い、中央K3区間のK14-middleは拾う数を減らしてprecisionとfractureを改善する。これは「後半層がfamilyを担当する」という層役割の証明ではない。main-network配置の違いがencoder/decoderへ返す勾配と表現を変え、最終的なboundary routerの選択biasが変わったという範囲の解釈である。

#### Landmark coverage

landmarkはK14-late `.867`、K1G1 `.850`がT26 `.667`を上回り、K3-first `.658`はT26と同程度、K14-middle `.533`、K15-split `.458`、K3G1 `.283`は下回った。family Cとほぼ同じ配置依存を示し、KDAを多くするだけでは再利用可能なfamily共通境界を形成できない。

K3G1はmedium windowで`.917`まで上がった後、terminalで`.283`まで低下した。高KDA構成はlandmarkを形成できないのではなく、学習後半に維持できなかった。この非単調軌跡は、step 55単点や最終stepだけで配置を選ぶ危険性を示している。

#### Family integrity

integrityはT26 `.908`が最高で、K1G1/K15-split `.833`、K14-middle `.775`、K3G1 `.692`、K14-late `.667`、K3-first `.550`だった。置換量が増えると概ね低下する方向はあるが、K14-lateがK19のK3G1より低く、K15がK14より高いため、量だけの効果ではない。

Transformer-onlyのT26は、保護語彙を壊さないという制約に最も強い。部分置換構成のcoverage改善を採用する場合も、integrityを独立した非相殺制約にする必要がある。K14-lateの高coverageを順位平均で採用できない主因もこの点である。

#### 文節 precision / coverage

文節P/CはK15-split `.304/.478`が突出し、K1G1 `.159/.322`、K3-first `.173/.296`、K14-late `.203/.287`もT26 `.143/.217`の一部または両方を上回った。一方、K3G1は`.076/.096`まで低下した。

従って、文節改善には適量・分割配置のKDAが有効である可能性があるが、完全置換・高KDAまで進めると逆効果になる。K15-splitの前後に分けたK3区間が文節に強いことは、単なるKDA層数ではなく、複数深度へ連続KDA区間を配置することが関係するという仮説を与える。ただし、K15はfamily C/landmarkを失うため文節だけの最適化にはできない。

#### Transition

terminal transitionはT26 `.455`、K1G1 `.409`、K14-late `.375`、K14-middle `.364`、K3-first `.338`、K15-split `.330`、K3G1 `.259`だった。例外はあるものの、KDA量が多いほど隣接step間のboundary signatureが変わりにくくなる傾向がある。

これはKDAが境界を安定化する可能性を示すが、K3G1ではfamily・文節品質が低いままtransitionも低い。従って「良い境界へ収束した」のではなく、「特定の分割biasが動きにくくなった」可能性も同程度にある。transitionは品質指標ではなく、形成・固定の強さを測る診断値として扱う。

### 14.3 全体としての考察

観測結果は、T26とK3G1の間を単純に線形補間するものではない。主に次の3種類の効果が重なっている。

1. 置換の有無: T26からKDA/Gated MLAへ置換するとcategory coverageを上げやすい一方、family integrityを失いやすい。
2. KDA dosage: 高KDAのK3G1ではcategory P/Cの一部を保つが、family・landmark・文節が弱くなる。中間dosageにPareto上有用な点がある。
3. KDA position/連続性: 同量のK14-middle/lateでprecision・fracture対family C・landmarkが大きく入れ替わり、K15-splitで文節が突出する。配置効果はdosage効果と同等以上に大きい。

境界学習の観点では、Transformer-onlyのT26は語彙・familyの整合性を保つ方向、KDA/Gated MLA混合は説明可能なcategory境界を広く選ぶ方向にbiasを変えるように見える。連続KDAを中央へ置くと選択を絞ってprecision/fractureを改善し、後方や前後へ置くとfamily landmarkまたは文節coverageを増やす、という配置別の仮説が立つ。

ただし、この説明はboundary結果からの機構仮説であり、各層が特定の言語単位を担当することを直接測定したものではない。また、1 seed、220 step、sequence length 2048であり、architectureごとに同じseed値でもweight tensorの対応は同一ではない。次の構成比較ではK/G総数を固定して位置だけを変える対照を優先し、その後に同一位置でKDA量だけを変えることで、dosageとpositionを分離する必要がある。

現時点の目的に対する有力な役割分担は次の通りである。

- T26: family precision・integrityの上限baseline
- K1G1: category/family coverageとintegrityの均衡baseline
- K14-middle: category precision・低fractureの候補
- K14-late: category/family/landmark coverageの候補
- K15-split: 文節specialist
- K3G1: 完全置換・高KDA側の下限/上限を測るcontrol

したがって次の目標はK3G1へさらに近づけることではなく、K14-middleまたはK14-lateを起点に、T26のintegrityとfamily precisionを回復しながら、部分置換で得たcategory coverageまたは文節改善を残すことである。
