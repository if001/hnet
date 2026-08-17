# 言語的境界によるmain network候補選抜計画

作成日: 2026-08-17

## 1. 目的

小規模・中規模学習では、動的chunk分割が長期間学習後のmain network性能を改善する
ことを証明しない。長期間学習へ投入するmain networkへ、言語的な分割をproxyとして
ある程度のあたりをつける。

本計画は次の仮説を前提とする。

> 明らかに不自然な断片化が少なく、文節、活用語尾、助動詞、助詞、複合語境界、
> 句読点などで言語的に説明できるchunkを形成する構成は、長期間学習でmain networkが
> 利用しやすい可能性が高い。

この仮説の最終検証は長期間学習で行う。小規模・中規模では分割品質、再現性、BPBの
重大な退行がないことを使い、長期間学習へ進めるmain networkを選ぶ。

## 2. 既存Phase 2結論の扱い

- T26とK1T1のvalidation BPB差は実用同等幅内であり、優劣を確定しない。
- KDA側にfused実装がない現在の速度差はarchitecture選定に使わない。
- learned/Fixed/Random/Morph/shift介入は、learned boundaryが各モデル内で機能している
  ことの健全性確認に留め、言語的分割品質の順位には使わない。
- SFT後agent proxyは全候補がfloorだったため、小規模・中規模の選定条件から外す。
- したがってT26の選定は暫定結論から外し、main networkは未選定へ戻す。

## 3. UTF-8境界の扱い

H-Net実装は`byte-boundary-constraint`を`off`、`utf8-soft`、`utf8-hard`から選べる。
緩い設定ではUTF-8 continuation byte途中の境界が仕様上あり得る。

- UTF-8途中境界をmain networkの良否、除外条件、病的断片化の定義に使わない。
- 実験ではcheckpointの学習時設定を変更せず、その設定をmanifestへ記録する。
- inference時にも原則として学習時と同じconstraintを使う。
- UTF-8途中境界率は設定監査用の診断値としてのみ保存する。
- 人が読む分割表示では、途中byteを`<0xHH>`として可逆に表示する。
- 言語スコアはUTF-8 codepointとして復元できるspanを対象にする。復元不能spanは
  `constraint-dependent`として別集計し、不自然境界の分母・分子へ入れない。

## 4. 比較候補

小規模screeningでは、同条件のcheckpointを準備できる次のmain networkを対象とする。

- T26
- K1T1
- K3G1
- K1G1
- K3T1
- M3T1

利用可能な既存checkpointを先に監査する。不足構成は、同じmodel規模、データ順、
raw-byte予算、context length、checkpoint間隔、byte-boundary-constraintで追加学習する。
構成名だけが異なる条件を優先し、parameter数の差をmanifestへ記録する。

## 5. 評価データ

### 5.1 人手設計probe

各recordは本文、category、注目span、複数の許容境界、説明困難な境界、control種別を持つ。
単一の正解分割は置かない。

主categoryは次のとおりとする。

1. 文節・節境界
2. 活用語幹・活用語尾
3. 補助動詞・助動詞
4. 助詞周辺
5. 複合語・頻出語
6. 句読点・括弧・引用
7. 固有名詞、数値、単位
8. 英単語、identifier、path
9. code、JSON、tool形式
10. 文脈依存minimal pair
11. 文脈を変えても安定すべきcontrol pair

例えば`笑っている`では`笑/っている`と`笑って/いる`をともに説明可能とする。
`分/割の結/果は人/が`のような短い断片の連続は、個々の境界だけでなく局所chunk列
として説明困難度を評価する。複合語内部境界は常に誤りとはせず、辞書頻度、周辺chunk、
階層、文脈を併用する。

### 5.2 corpus由来probe

固定validationから日本語をseed固定で抽出し、Fugashi/UniDicと規則を使って候補位置を
付与する。形態素解析結果は唯一のgoldではなく、説明可能境界候補の一つとして扱う。
人手設計probeへの過適合を避けるため、候補選定は両probeの方向一致を要求する。

## 6. 境界予算の統一

分割品質は境界数に依存するため、各候補を元のhard maskだけで直接比較しない。

1. routerのboundary probabilityとvalid maskを保存する。
2. stage別・record別に共通の境界数を定める。
3. 各候補のboundary probability上位位置から、同数の境界を選ぶ。
4. 主比較は共通境界予算で行い、学習時hard maskは補助比較として残す。
5. stage 0とstage 1を分けて評価する。

共通予算は候補全体の中央値を主条件とし、低・中央・高の3予算で感度分析する。
先頭境界などモデル実行上必須の位置は全条件で保持する。UTF-8安全位置への強制射影は
行わない。

## 7. 指標

### 7.1 主要指標

- `explainable_boundary_precision`: 全評価可能境界に占める説明可能境界の割合。
- `category_coverage`: categoryごとの候補位置に境界が現れる割合。
- `unexplained_boundary_rate`: どの許容categoryでも説明できない境界の割合。
- `pathological_fragmentation_rate`: 説明困難な短いchunkが局所的に連続する割合。
- `context_adaptation_score`: minimal pairで許容境界の変化と同方向に境界が変わる割合。
- `unnecessary_variation_rate`: control pairで理由なく境界が変わる割合。

### 7.2 補助指標

- stage別chunk長分布、1/2/3 codepoint相当spanの分布。
- category別境界数とboundary probability margin。
- seed間・checkpoint間の主要指標の範囲と順位安定性。
- validation BPBと圧縮率。
- 学習時hard maskと共通境界予算maskの一致率。
- constraint設定とUTF-8途中境界率。これは選定得点へ入れない。

単一の加重総合点だけで決めず、主要指標の表、Pareto、分割galleryを併記する。

## 8. 実験段階

### Phase A: 評価器実装とsynthetic検証

- annotation schemaと日本語probeを作る。
- boundary probabilityから共通予算maskを作る。
- 許容境界、説明困難境界、pair指標、断片化指標を実装する。
- CPU unit testでは手作りboundary maskを用い、候補checkpointやCUDAへ依存しない。
- 数件のcheckpointで分割galleryを作り、offset/BOS/stage mappingを目視監査する。

### Phase B: 小規模全候補screening

- T26、K1T1、K3G1、K1G1、K3T1、M3T1を同じprobeで評価する。
- 原則2 seed以上、学習前半・後半の2 checkpoint以上を使う。
- 明らかな断片化、説明可能性、文脈適応、seed/checkpoint再現性を比較する。
- BPBに重大な退行がある候補は、境界が良く見えても中規模へ進めない。
- 上位2から3候補を中規模候補とする。

### Phase C: 中規模候補確認

- 上位候補を同じraw-byte予算、データ順、2 seed以上で学習する。
- 125M raw bytes間隔など複数checkpointで同じ境界評価を行う。
- 分割の説明可能性とcategory分布が学習量とともに安定・改善するかを見る。
- 候補間BPBが実用同等なら、速度ではなく境界proxyの再現性を優先する。

### Phase D: 長期間学習候補の選定

次を満たす1構成を選ぶ。

1. 説明困難境界と病的断片化が候補中で少ない。
2. 複数の言語categoryを一貫して捉え、句読点だけに偏らない。
3. 文脈依存pairで動的変化があり、control pairでは不要な変動が少ない。
4. 複数seed・checkpointで傾向が再現する。
5. validation BPBが実用同等範囲から重大に退行しない。

この選定は長期性能の証明ではなく、前提仮説に基づく候補選抜であることをレポートへ
明記する。

## 9. 実行・保存規則

- branchは`200m_main`だけを使う。
- コード変更はローカルで実装・test・commit・pushし、Colabでpullする。
- GPUを使う学習、checkpoint評価、Drive操作はColabで行う。
- runtime再起動時はDrive mount、repository clone、Prepare依存導入、GPU/import smokeを
  最初から実行する。
- 長い処理はforeground cellで実行する。待機中は10分以内に状態確認cellを実行する。
- 一時plan、判断memo、監査記録は`plan/mid/`へ保存する。
- Drive rootは`/content/drive/MyDrive/hnet_agent_200m_main`とする。
- raw boundary record、集約CSV/JSON、分割gallery、status manifestをDriveへ保存する。
- checkpoint、dataset、download artifactはGitへcommitしない。

## 10. 予定artifact

| artifact | 保存先 |
| --- | --- |
| annotation schema | `configs/linguistic_boundary_probe_v1.json` |
| 評価CLI | `scripts/evaluate_linguistic_boundaries.py` |
| 集約CLI | `scripts/summarize_linguistic_boundary_screening.py` |
| unit test | `tests/unit/test_linguistic_boundary_evaluation.py` |
| 小規模raw結果 | Drive `evals/linguistic_boundary_phase1/` |
| 中規模raw結果 | Drive `evals/linguistic_boundary_phase2/` |
| status manifest | Drive `manifests/linguistic_boundary_screening_status.json` |
| 分割gallery | Drive `reports/linguistic_boundary_selection/gallery/` |
| 選定レポート | `plan/mid/202608_linguistic_boundary_selection_report.md` |

## 11. 停止条件

- annotation規則で人が見ても妥当性を判断できない場合は、候補順位を出さずprobeを修正する。
- 共通境界予算のmappingがstage間で不正確な場合は評価を停止する。
- 利用可能checkpoint間で学習条件が比較不能なら、その構成を同じ条件で再学習する。
- 全候補が近接し順位がseedで反転する場合は、無理に1構成へ決めず中規模候補を増やす。
