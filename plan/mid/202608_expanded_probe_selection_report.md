# 拡張日本語probeによるmain network選定レポート

作成日: 2026-08-21

## 1. 結論

長期実験の第一候補を **K3G1**、比較を残す第二候補を **K3T1** とする。

K3G1はPhase 2の中央固定予算・stage 1で、4候補中最高の説明可能境界precision、最低の
lexeme fracture rate、最高の後半step間Jaccardを示した。coverageも最高値との差が小さく、
活用、助詞、文節、複合語、structuredのいずれかだけに偏らず比較的広いcategoryを捉えた。

ただしseedごとの順位反転とnative分割の短断片が残るため、K3G1が長期学習で最良とはまだ
断定しない。K3T1は活用・助詞・文節で強く、nativeでsevere short fragmentationがなく、
K3G1とは異なる性質を持つ比較候補として残す。T26とK1G1は長期本線から外すが、必要なら
対照実験のcheckpointとして再利用できる。

## 2. 実験範囲

- probe: 11 category x 8文 = 88文
- Phase 1: 6候補、seed 42、step 55
- Phase 2: T26、K1G1、K3G1、K3T1
- seed: 42、43、44
- checkpoint: step 55、110、165、220
- 主比較: `central`固定境界予算、stage 1
- 補助比較: native分割、validation BPB、native圧縮率
- constraint: `utf8-hard`
- Phase 2 raw評価: 48 JSON、各88 records、全件正常終了

Phase 1ではK3G1を短断片だけで除外せず、K3G1を含む4候補をPhase 2へ進めた。K3G1は
ratio weight 0.08、inner/outer target 3.0/2.5へ再較正し、3 seedを220 stepまで学習した。

## 3. Phase 2主要結果

step 220、中央固定予算、stage 1の3 seed平均を示す。precision、coverage、best F1、
lexeme fractureは境界単位、short/severeは88文中の該当record数である。

| model | precision | coverage | best F1 | lexeme fracture rate | short records | severe records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T26 | 0.439 | 0.239 | 0.410 | 0.257 | 0.67 | 0 |
| K1G1 | 0.451 | **0.271** | 0.448 | 0.255 | 1.67 | 0 |
| **K3G1** | **0.490** | 0.261 | **0.451** | **0.218** | 1.00 | 0 |
| K3T1 | 0.453 | 0.254 | 0.447 | 0.265 | 1.00 | 0 |

K3G1のprecision seed範囲は0.407--0.583、coverageは0.252--0.269である。precisionの
seed幅は大きい一方、coverageは4候補中最も狭い。K1G1はcoverage平均が最高だが
0.222--0.363とseed差が大きく、precisionも0.379--0.545で安定した優位ではない。

## 4. Category別の特徴

step 220のprecision / coverage（3 seed平均）の主要categoryは次のとおりである。

| model | 活用 | 助動詞 | 助詞 | 文節 | 複合語 | structured |
| --- | --- | --- | --- | --- | --- | --- |
| T26 | .178/.095 | .552/.312 | .770/.333 | .301/.348 | .527/.267 | .297/.089 |
| K1G1 | .444/.214 | **.747/.375** | .674/**.444** | .168/.203 | .553/**.383** | **.512/.215** |
| K3G1 | .485/.238 | .637/.292 | **.753/.444** | **.321/.333** | **.572/.333** | .470/.178 |
| K3T1 | **.610/.286** | .621/.271 | .742/.429 | .295/.319 | .527/.350 | .427/.170 |

K3G1は単一categoryの最大値を集めたモデルではないが、弱点が比較的小さい。K3T1は
活用・助詞・文節に強く、日本語の言語的分割という仮説に対する補完候補になる。K1G1は
助動詞とstructuredで強い一方、文節とseed安定性が弱い。T26は活用とstructuredが明確に
弱く、速度を順位に使わない今回の基準では長期本線へ残す根拠が弱い。

固有名詞とidentifier/tool文字列は全候補でlexeme fractureが多い。これはモデル間順位より、
220 stepでは語彙的まとまりの獲得が不十分であることを示す。

## 5. 分割例

K3G1の中央固定予算では、seedによって次のような説明可能な分割が得られた。

- `値が|正しければ、そのまま|次へ|進む|。`
- `分割の|結果は人が|確認|する。`
- `分割|の結果|は人が確認|する。`
- `人が|結果を|確認した。`
- `ログを|読み|込んでいる。`

一方、次の問題は残った。

- `自|然言語|処理|モデルを...` のような複合語内部切断
- `東京都千代|田区` のような固有名詞内部切断
- JSON/tool文字列で英単語やkey/value内部を切る分割
- 同じ文章でもseedにより境界位置が変わる例

したがって「自然な分割だけを行うモデル」ではなく、「4候補中、説明可能性と不自然切断の
バランスが最も良い候補」という判断である。

## 6. 学習中の変化

中央固定予算・stage 1のrecord別boundary Jaccardは、全候補で学習後半ほど上昇した。

| model | step 55→110 | 110→165 | 165→220 |
| --- | ---: | ---: | ---: |
| T26 | 0.637 | 0.670 | 0.800 |
| K1G1 | 0.616 | 0.640 | 0.787 |
| **K3G1** | 0.639 | **0.687** | **0.817** |
| K3T1 | 0.595 | 0.639 | 0.768 |

K3G1ではstep 165→220の1 seed・88文あたり平均で、unexplained境界が13.3個出現し12.3個
消失、lexeme fractureが6.7個出現し6.3個消失した。Jaccardは高くなるが境界学習は停止して
おらず、encoder/decoderの継続学習で分割が変化するという前提を支持する。

dynamic/control pairの厳格なpass判定では、全モデル・全stepでdynamicはほぼ獲得されず、
step 220では全モデル0/6だった。この判定は今回の候補順位には使用しない。動的分割の不存在を
証明する結果でもなく、3 seed x 2 dynamic pairの完全signature条件が厳しすぎる可能性がある。

## 7. Native分割と固定予算

step 220 nativeの3 seed平均は次のとおりである。

| model | precision | coverage | lexeme fracture rate | short records | severe records |
| --- | ---: | ---: | ---: | ---: | ---: |
| T26 | 0.400 | 0.278 | 0.305 | 1.00 | 0.00 |
| K1G1 | 0.383 | 0.318 | 0.288 | 5.33 | 1.00 |
| K3G1 | 0.420 | **0.368** | **0.215** | 6.67 | 1.67 |
| K3T1 | **0.443** | 0.268 | 0.291 | **1.00** | **0.00** |

K3G1はnativeでもlexeme fractureが最少でcoverageが高いが、境界数が増えると短断片とsevere
判定が増える。このためK3G1を単独確定せず、nativeが比較的安定したK3T1を残す。長期実験では
固定予算評価に加え、native圧縮率の推移とnative galleryを継続監視する必要がある。

## 8. Validationと圧縮率

step 220の3 seed平均を示す。

| model | validation BPB | L1/L0 | L2/L1 | L2/L0 |
| --- | ---: | ---: | ---: | ---: |
| T26 | 1.6767 | 3.842 | 2.486 | 9.547 |
| K1G1 | 1.6773 | 3.756 | 2.513 | 9.434 |
| K3G1 | 1.6833 | 3.838 | 2.448 | 9.394 |
| K3T1 | 1.6840 | 3.805 | 2.943 | 11.198 |

BPB差は最大でも約0.0073で、短期proxyとしてmain networkを決定できる大きさではない。
速度もfused実装差を含まないため順位に使用しない。K3T1のnative L2/L0は他候補より高く、
K3G1との長期比較ではraw-byte予算に加えて実効圧縮率を必ず併記する。

## 9. 長期実験への提案

1. 本線をK3G1、比較線をK3T1とする。
2. 同じraw-byte予算、同じデータ順序、seedを用いる。
3. checkpointごとに今回の88文probeを中央固定予算とnativeの両方で評価する。
4. 特に複合語、固有名詞、identifier、tool/JSONのlexeme fractureを追跡する。
5. downstream main networkの学習しやすさは長期loss/BPBと生成・SFT評価で初めて検証する。
6. 長期途中でK3G1 nativeのsevere短断片が減らない場合は、K3T1を優先する判断点を設ける。

## 10. Artifact

- raw: Drive `evals/linguistic_boundary_phase2_v2/`
- 集約CSV/JSON/gallery: Drive `reports/linguistic_boundary_selection/phase2_v2/`
- trajectory分析: Drive `reports/linguistic_boundary_selection/phase2_v2/phase2_v2_analysis.json`
- 実行status: Drive `manifests/linguistic_boundary_phase2_v2_evaluation_status.json`

本評価は「言語的に説明可能な境界がmain networkの長期学習を助ける」という仮説に基づく
候補選定であり、その仮説自体を短期実験で実証したものではない。
