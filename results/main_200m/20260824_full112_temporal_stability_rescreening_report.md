# full112時間安定性 main-network再screeningレポート

作成日: 2026-08-24

## 結論

step 55単点ではなく、category 88文 + family 24文をstep 10/20/30/40/50で評価した結果、
main-networkごとの強みは一時点だけの偶然ではなく、複数stepに現れる境界選択biasを含むことが確認できた。
一方、全評価軸でK1G1を上回る構成はなかった。

- K14-middleは、category precision、family coverage、landmark coverage、文節precisionをK1G1より比較的
  安定して改善した。今回の候補では時間方向のバランスが最もよい。
- ただしK14-middleもfamily integrityがK1G1より低く、step 55の固定budget profileではfamily・文節改善が
  一貫しない。K1G1を直ちに置き換える根拠にはならない。
- K15-splitは文節precision/coverageとcategory fractureで明確に強いが、family coverageとlandmarkを失う。
  文節specialistという従来判断が、full112の時間軌跡でも支持された。
- K14-frontはlandmark・文節を改善するがfamily integrityが弱い。K14-lateとK16-evenはfractureが増える。
- K3T1とT26はfamily側、K1-first/K3-firstはprecision・文節側に強みを持つが、coverageまたはintegrityとの
  trade-offが残る。

したがってK1G1をanchorとして維持する。Phase 2のseed軌跡へ進める価値がある候補はK14-middleに限定する。
これは長時間候補への昇格ではなく、観測した均衡改善がweight初期値・data orderを変えても残るか、family
integrity低下が再現するかを判定するためである。K15-splitは既存のseed要因分離でtrade-offが確認済みのため、
同じ目的で再延長しない。

## 1. 実験条件

計画: `plan/mid/20260824_full112_temporal_stability_rescreening_plan.md`

- 対象: K1G1、K14-front/middle/late、K15-split、K16-even、K3G1、K3T1、T26、
  K1-first、K3-firstの11構成
- seed: `(model_init_seed=42, data_order_seed=42, train_runtime_seed=42)`
- max step: 55、LR horizon: 220
- constraint: `utf8-hard`
- 時系列評価: full112、native、step 10/20/30/40/50
- profile評価: full112、step 55、low/central/high/native
- 主判定: stage 1
- 速度は順位に使用しない

11 runすべてが正常終了した。各runについてstep 55 checkpoint 1個、5時点のchunk report、各時点112注釈を
監査した。step 55 profile評価も11ファイル、各112注釈・4条件が揃っている。

## 2. 指標と集計方法

- precision: 選択された評価可能境界のうち、言語的に説明可能な境界の割合。大きいほど不要な境界が少ない。
- coverage: probeが許容する説明可能境界のうち、実際に選択した割合。大きいほど有用な境界を広く拾う。
- fracture record occupancy: fracture境界を1個以上含む文の割合。小さいほどよい。
- family precision/coverage: 活用・派生・文脈variantからなるfamily 24文での説明可能性と被覆。
- landmark coverage: family間で共通する、名前を付けた重要境界を選択した割合。大きいほどよい。
- family integrity: family文で保護対象の語彙単位を壊さなかった文の割合。大きいほどよい。
- transition rate: 隣接する評価時点で選択境界signatureが変化したrecordの割合。高低だけで良否を決めない。

各step内で境界数をmicro集計し、その後stepを等重みで平均した。全stepの境界をpoolしていない。lateは
step 30/40/50である。precisionは選択境界が0件のstepでは未定義とし、0に置換していない。K1G1はstep 10で
category・文節の選択境界が0件だったため、precisionのK1G1 paired比較は両者で定義されたstep 20--50の
4時点で行った。coverageとrecord occupancyは全5時点を使用した。

`P q20`はprecisionが定義された時点、`C q20`は5時点の下側20%点、`F q80`はfracture occupancyの
5時点における上側80%点である。

## 3. 5時点の時間安定性

### 3.1 全体

| 構成 | category P mean/q20 | category C mean/q20 | category F mean/q80 | family P/C | landmark C | integrity | 文節 P/C | transition |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |
| K1G1 | .332/.303 | .226/.099 | .218/.332 | .484/.442 | .492 | .917 | .117/.165 | .673 |
| K14-front | .420/.399 | .227/.164 | .230/.341 | .416/.527 | .667 | .783 | .201/.278 | .690 |
| K14-middle | .425/.410 | .238/.176 | .195/.257 | .470/.552 | .633 | .833 | .184/.252 | .699 |
| K14-late | .414/.397 | .249/.169 | .268/.425 | .509/.545 | .650 | .775 | .186/.261 | .727 |
| K15-split | .359/.254 | .208/.089 | .132/.198 | .560/.358 | .425 | .892 | .364/.443 | .577 |
| K16-even | .382/.366 | .213/.126 | .273/.400 | .623/.473 | .633 | .767 | .196/.209 | .682 |
| K3G1 | .372/.357 | .208/.133 | .345/.548 | .524/.364 | .425 | .742 | .150/.139 | .662 |
| K3T1 | .325/.310 | .216/.144 | .291/.432 | .650/.545 | .675 | .783 | .111/.122 | .722 |
| T26 | .397/.384 | .172/.089 | .225/.343 | .682/.442 | .600 | .842 | .103/.087 | .605 |
| K1-first | .405/.393 | .216/.171 | .295/.389 | .725/.467 | .642 | .767 | .266/.287 | .625 |
| K3-first | .443/.419 | .215/.147 | .227/.300 | .784/.461 | .633 | .792 | .273/.270 | .636 |

K14-middleはcategory P/Cとfractureの3軸でK14群の中で最も均衡し、family coverage、landmark、文節も
K1G1を上回った。ただしintegrityは`.833`でK1G1の`.917`に届かない。K15-splitは文節と低fractureに強い
一方、family coverage `.358`とlandmark `.425`が低い。

transitionは`.577`から`.727`まで広く分布した。K15-splitのtransitionが最小だが、これは同じ分割を
維持したという診断値であり、その分割が全categoryに良いことを意味しない。K14-lateの最大値も、説明可能な
別境界への移行を含むため、それだけで失格とはしない。

### 3.2 K1G1との同一step paired比較

この比較では、K1G1のstep 10と各構成のstep 10、step 20同士、以後同様にstep 50までを対応付けた。
同じstepでは、同じdata orderの同じraw batch列、同じ累積入力byte数、同じoptimizer update数を経験している。
目的は、各時点で「同じraw学習量を与えたときの境界品質」がarchitectureによってどう異なるかを、学習中の
共通変動を揃えて比較することである。これは学習効率を含むfixed-training-dose比較である。

この比較は、構成が異なってもcategory、文節、family等の境界が同じ順序・同じ速度で形成されることを仮定
しない。実際には、ある構成ではcategory precisionが早く形成され、別の構成では文節境界が後から形成される
など、境界学習の位相がずれる可能性がある。また、raw入力byte数が同じでも、compressionと分割が異なるため、
main networkが処理する実効chunk数や「境界形成の成熟度」が同じとは限らない。

したがって同一stepでの正の差は、「同じ学習量でその時点までに高い品質へ到達した」ことを表すが、遅れて
形成される構成の最終品質が低いことまでは表さない。単一のearly stepだけの差は形成速度の差である可能性が
あり、architecture固有の持続的なbiasと解釈するには、複数stepでの方向一致とlate windowで差が残ることを
併せて確認する。本レポートのpaired勝率とlate集計はこの目的で用いる。一方、形成位相そのものを揃えるには、
今後、同じcoverage、選択境界数、またはcompression水準で比較するmatched-state評価が別途必要である。

表は`median delta (改善step数/比較可能step数)`を示す。fractureは負が改善である。precisionは主に4時点、
coverageとfractureは5時点の比較である。

| 構成 | category P | category C | category F | family C | landmark C | 文節 P | 文節 C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K14-front | +.103 (4/4) | +.009 (3/5) | +.011 (1/5) | +.121 (3/5) | +.208 (4/5) | +.117 (4/4) | +.043 (3/5) |
| K14-middle | +.127 (4/4) | +.009 (3/5) | .000 (2/5) | +.121 (4/5) | +.167 (4/5) | +.078 (4/4) | +.043 (3/5) |
| K14-late | +.103 (4/4) | +.026 (4/5) | +.091 (1/5) | +.091 (4/5) | +.083 (4/5) | +.121 (4/4) | +.087 (3/5) |
| K15-split | +.147 (3/4) | .000 (2/5) | -.102 (4/5) | -.121 (1/5) | .000 (1/5) | +.250 (4/4) | +.348 (4/5) |
| K16-even | +.063 (3/4) | .000 (1/5) | +.068 (1/5) | .000 (2/5) | +.125 (4/5) | +.097 (3/4) | +.087 (3/5) |
| K3G1 | +.031 (3/4) | .000 (2/5) | +.170 (0/5) | -.030 (1/5) | -.083 (1/5) | +.042 (2/4) | .000 (2/5) |
| K3T1 | -.005 (2/4) | -.026 (1/5) | +.057 (0/5) | +.091 (4/5) | +.208 (4/5) | +.010 (3/4) | .000 (2/5) |
| T26 | +.089 (3/4) | -.013 (1/5) | +.034 (1/5) | .000 (2/5) | +.167 (3/5) | +.019 (3/4) | .000 (1/5) |
| K1-first | +.085 (3/4) | .000 (2/5) | +.068 (0/5) | -.030 (1/5) | +.125 (4/5) | +.155 (4/4) | +.087 (3/5) |
| K3-first | +.134 (3/4) | .000 (1/5) | .000 (2/5) | .000 (1/5) | +.167 (4/5) | +.154 (4/4) | +.087 (3/5) |

K14-middleのcategory precisionと文節precisionは比較可能な4時点すべてで改善した。family coverageと
landmarkも4/5で改善し、K14-front/lateよりcategory fractureが低い。これがK14-middleを次の再現性確認
候補とする主な根拠である。

一方、K15-splitは文節P/Cとcategory fractureを同時に改善するが、family coverageが5時点中4時点で
K1G1以下だった。K3G1はfractureが全5時点で悪化し、K3T1はfamilyを改善する代わりにcategory・文節を
改善しない。

### 3.3 late window

| 構成 | category P/C/F | family P/C | landmark C | integrity | 文節 P/C |
| --- | --- | ---: | ---: | ---: | --- |
| K1G1 | .313/.336/.322 | .529/.657 | .708 | .944 | .112/.246 |
| K14-front | .421/.306/.337 | .527/.747 | .931 | .681 | .205/.304 |
| K14-middle | .454/.322/.254 | .597/.778 | .861 | .778 | .178/.246 |
| K14-late | .428/.338/.367 | .517/.747 | .861 | .681 | .211/.319 |
| K15-split | .492/.309/.174 | .747/.596 | .708 | .875 | .355/.580 |
| K16-even | .387/.302/.371 | .585/.646 | .861 | .653 | .178/.261 |
| K3G1 | .368/.291/.534 | .587/.535 | .625 | .625 | .126/.145 |
| K3T1 | .326/.301/.443 | .700/.747 | .917 | .653 | .102/.130 |
| T26 | .409/.249/.318 | .805/.687 | .944 | .792 | .086/.116 |
| K1-first | .416/.289/.383 | .713/.616 | .847 | .653 | .250/.333 |
| K3-first | .463/.298/.299 | .712/.616 | .847 | .653 | .261/.319 |

lateでもK14-middleはcategory precision、低fracture、family coverage、landmarkを同時に維持した。ただし
integrity `.778`はK1G1 `.944`より低い。K15-splitの文節coverage `.580`は突出しているが、family coverageは
K1G1を下回る。このため、平均だけでなくlateでもcombinedな完全改善は成立しない。

## 4. Category別の特徴

5時点平均で各categoryのprecisionまたはcoverageが最大だった構成は次の通りである。

| category | precision最大 | coverage最大 |
| --- | --- | --- |
| auxiliary | K1-first | K1G1 |
| bunsetsu | K15-split | K15-split |
| compound | K14-middle | K1-first |
| context_control | K15-split | K3G1 |
| identifier | K14-front | K3T1 |
| inflection | K3G1 | K3T1 |
| number_unit | K3T1 | T26 |
| particle | K3-first | K14-late |
| proper_noun | K3-first | K3G1 |
| punctuation | K14-late | K14-middle |
| structured | K3G1 | K1G1 |

単一構成が全categoryを支配していない。KDAの量・位置・順序により、選びやすい境界categoryが変わるという
従来観測が時間平均でも残った。ただし、この表は「各層が特定categoryを担当する」ことを示さない。
architectureがrouterへ与える表現・勾配の違いによって、55-step時点までの境界選択biasが変わったという
範囲の結論である。

## 5. step 55 profile頑健性

K1G1より高かったprofile数を4条件中の個数で示す。

| 構成 | category P/C | family P/C | landmark C | 文節 P/C |
| --- | --- | --- | ---: | --- |
| K14-front | 4/3 | 1/2 | 4 | 3/3 |
| K14-middle | 3/1 | 1/1 | 0 | 2/1 |
| K14-late | 4/0 | 1/1 | 0 | 3/2 |
| K15-split | 4/4 | 2/2 | 1 | 4/4 |
| K16-even | 4/3 | 2/1 | 0 | 4/4 |
| K3G1 | 4/4 | 2/1 | 1 | 4/4 |
| K3T1 | 4/2 | 2/2 | 2 | 3/3 |
| T26 | 4/2 | 2/2 | 4 | 2/2 |
| K1-first | 4/3 | 2/1 | 2 | 4/4 |
| K3-first | 4/3 | 2/1 | 2 | 4/4 |

K14-middleのnative時間軌跡は均衡しているが、固定budgetを変えるとfamily coverage、landmark、文節coverageの
優位は弱い。K14-frontはprofile方向一致が比較的高いものの、時間軌跡でfamily integrity低下が継続した。
K15-splitは文節改善が全profileで残り、specialistとしての特徴は強い。

profileは境界数を外から揃えた条件を含み、nativeと同じ意味ではない。このためprofile勝数だけで候補を
決めず、native時間軌跡に対する頑健性診断として使用した。

## 6. 仮説に対する回答

1. **あるstepだけ高い構成を選ぶ問題は実在する。** category coverageやfractureはstepごとに順位が変わり、
   step 55単点だけではK14-middleの均衡性やK15-splitの継続的trade-offを判別できなかった。
2. **モデルごとのcategory差には時間的に残る成分がある。** category最大構成は分散し、複数step平均でも
   K15-splitの文節、K3T1のfamily、K14群のcategory precisionなどが観測された。
3. **現在のordered/reverse mixingだけでは強みを完全には合成できない。** precision・文節を上げる構成は
   coverage、fracture、またはintegrityを失う。K1/K3の順序変更だけでもこのtrade-offは解消しない。
4. **K14-middleは最も近いが未確定である。** 時間軌跡では複数軸を改善する一方、integrityとprofile頑健性が
   弱い。architecture効果かseed固有かを分離する価値はあるが、長時間学習候補への直接昇格はできない。

## 7. 判断と次の実験

K1G1をanchorとし、Phase 2はK1G1とK14-middleに絞る。

- init差: `(43,42,42)`
- data差: `(42,43,42)`。architecture別の同一step 0 stateを使用する
- full112 nativeをstep 10/20/30/40/50で再評価する
- 主確認: category P/C/F、family C/landmark/integrity、文節P/Cのpaired方向
- baselineを含む2系列で順位反転・僅差なら、init-44/data-44だけを追加する

K14-middleを昇格する条件は、category precisionと文節precisionの改善が再現し、family coverage/landmarkを
維持しながらintegrity低下が継続しないことである。integrity低下がinit/dataの双方で残る場合はcombined候補
ではなく、K14-middleもspecialistとして停止する。

## 8. 制約

- 55 stepのscreening proxyであり、長期main-network loss、生成品質、学習容易性を直接証明しない。
- full112は有限probeであり、categoryごとの文数も同一ではない。
- precision未定義時点があるため、precision平均の有効時点数とcoverageの5時点を同一視しない。
- seed三因子はbaseline 42だけであり、本レポートの時間安定性はseed再現性ではない。
- transitionは評価間隔10 stepの観測であり、その間の細かな揺れを測っていない。
- UTF-8違反は`utf8-hard`設定で抑制されており、main-network順位の指標には使用していない。

## 9. Artifacts

- 計画: `plan/mid/20260824_full112_temporal_stability_rescreening_plan.md`
- 学習監査: Drive `manifests/full112_temporal_stability/phase1_artifact_audit.json`
- profile監査: Drive `manifests/full112_temporal_stability/phase1_step55_profile_audit.json`
- 集計JSON: Drive `reports/full112_temporal_stability/phase1_full112_temporal_analysis.json`
- checkpoints/raw: Drive `runs/full112_temporal_stability/phase1/`
- step 55 profiles: Drive `evals/full112_temporal_stability/phase1_step55/`
