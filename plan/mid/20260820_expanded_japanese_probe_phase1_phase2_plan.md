# 拡張日本語probeによるPhase 1再評価・Phase 2追加実験計画

作成日: 2026-08-20

## 1. 目的

24文章probeと単純な連続短fragment判定に依存した候補選抜を見直す。Phase 1の6候補を
拡張probeで再評価し、言語的に説明可能な分割、語彙内部の不自然な切断、seed間再現性を
用いてPhase 2候補を選ぶ。その後、不足候補を同じ学習条件でPhase 2へ進める。

長期性能は小規模・中規模では確定できない。本実験は、長期間学習へ投入するmain networkの
あたりをつけるproxy評価である。

## 2. 既存評価の修正点

### 2.1 連続短fragment

従来の「1 codepoint以下のchunkが3個以上連続」を直ちにpathologicalとする判定は、
`値|が|正`や`分|割|の`のような、助詞・語幹を含む説明可能な短単位も同じ重さで扱う。
この値は記述統計として残すが、単独の除外条件にしない。

次を分離して集計する。

- `short_fragmentation`: 1 codepoint以下のchunkが3個以上連続した記述値。
- `severe_short_fragmentation`: 1 codepoint以下が4個以上連続し、複数の説明困難境界を含む。
- `lexeme_fracture`: annotationで保護した語彙単位内部に境界を作ること。
- `best_segmentation_f1`: 複数の許容分割の和集合ではなく、各完全分割との一致度の最大値。

`自|然|言|語|処理`では`自然`、`言語`内部の切断をlexeme fractureとして数える。
`東|京|都|千代田|区`では`東京`、`千代田`内部を区別する。一方、助詞`が`、`の`、`は`が
1文字chunkになるだけではlexeme fractureにしない。

### 2.2 Probe規模

11 categoryを各8文章、合計88文章へ拡張する。既存24文章は保持し、次を追加する。

1. 活用語尾
2. 補助動詞・助動詞
3. 助詞
4. 文節・節境界
5. 漢字・カタカナ複合語
6. 句読点・括弧・引用
7. 固有名詞
8. 数値・単位
9. identifier・path・URL
10. JSON・code・tool形式
11. 文脈dynamic/control pair

許容分割は単一goldにせず複数を登録する。語彙として保持したい部分は
`protected_substrings`へ登録する。

## 3. 評価条件

- 候補: T26、K1T1、K1G1、K3G1、K3T1、M3T1
- Phase 1: seed 42、step 55
- 主比較: `central`固定境界予算、stage 1
- 感度分析: low/high compression、native
- constraint: checkpoint学習時と同じ設定
- UTF-8途中境界率と速度は順位に使用しない

Phase 1では次を併記する。

- explainable boundary precision / coverage
- best segmentation precision / recall / F1
- unexplained boundary rate
- short / severe short fragmentation
- lexeme fracture count/rate
- dynamic/control pair
- category別gallery

## 4. Phase 1候補選定

単一総合点だけでは決めない。次の順で2～4候補をPhase 2へ進める。

1. severe short fragmentationとlexeme fractureが極端に多くない。
2. precisionとcoverageがPareto劣位でない。
3. 活用、助動詞、助詞、文節、複合語など複数categoryを捉える。
4. BPBに重大な退行がない。
5. 24文章probeだけで除外したK3G1を先入観なく再評価する。

差が1～2境界程度なら同等とし、Phase 2候補を増やす。

## 5. Phase 2

選定候補をseed 42、43、44、step 55/110/165/220で比較する。既存checkpointは再利用し、
不足構成だけを同一raw-byte予算で学習する。

K3G1が選ばれた場合、まず55-stepでratio weightとcompression targetを再確認し、T26/K1G1に
近いnative圧縮率を得る条件を決めてから3 seed・220 stepを実行する。

checkpoint推移では次を確認する。

- 指標の平均・seed範囲
- step間boundary Jaccard
- 説明困難境界の出現・消失
- lexeme fractureの出現・消失
- 短fragmentから語・文節単位への変化
- 活用・助動詞境界の獲得
- native分割と固定予算分割の差
- validation BPBとnative圧縮率

## 6. Artifact

| artifact | 保存先 |
| --- | --- |
| 拡張probe | `configs/linguistic_boundary_probe_v2.json` |
| 評価実装 | `hnet/training/linguistic_boundaries.py` |
| unit test | `tests/unit/test_linguistic_boundary_evaluation.py` |
| Phase 1 raw | Drive `evals/linguistic_boundary_phase1_v2/` |
| Phase 1集約 | Drive `reports/linguistic_boundary_selection/phase1_v2/` |
| Phase 2 raw | Drive `evals/linguistic_boundary_phase2_v2/` |
| Phase 2集約 | Drive `reports/linguistic_boundary_selection/phase2_v2/` |
| 最終レポート | `plan/mid/202608_expanded_probe_selection_report.md` |

## 7. 実行規則

- branchは`200m_main`のみ使用する。
- コード修正はローカルで行い、test、commit、push後にColabでpullする。
- GPU実行とGoogle DriveアクセスはColab経由で行う。
- Colab初期化時はclone、Drive mount、Prepare、GPU/import smokeをやり直す。
- 長時間処理はforeground cellで実行する。
- checkpoint、dataset、raw結果はGitへcommitしない。

## 8. 停止条件

- v2 annotationがsurfaceを復元できない場合はGPU評価前に修正する。
- 固定境界予算のstage mappingが崩れた場合は評価を停止する。
- Phase 1差が小さい場合は無理に候補を削らずPhase 2へ進める。
- Phase 2でもseed順位が反転する場合は1候補へ確定せず、長期比較候補を複数残す。
