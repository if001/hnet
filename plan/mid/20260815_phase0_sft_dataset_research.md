# Phase 0 SFTデータセット調査・暫定採用方針

作成日: 2026-08-15

## 結論

Phase 2のagent proxy SFTでは、主tool trajectoryとして
`Agent-Ark/Toucan-1.5M` の `SFT` configを採用する。
補助として`Team-ACE/ToolACE`を残す。既存の日本語instruction、英語、codeを
混ぜ、tool dataだけで一般応答能力を上書きしない。

`Salesforce/APIGen-MT-5k`はCC BY-NC 4.0かつgated、
`Salesforce/xlam-function-calling-60k`はCC BY 4.0だがgatedである。
今回の再現可能な自動実験では両者を0件にし、認証・利用条件への依存を避ける。

長文専用データはPhase 2の短いSFTへ大量投入しない。Toucanの長いtrajectoryを
上位長分位から抽出してagent proxyへ少量使い、100k context級のACCはPhase 3以降の
16k preflight設計時に、切り詰めではなくcontext scheduleと合わせて再検討する。

## 一次情報と実測

### Agent-Ark/Toucan-1.5M

- 配布: https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M
- 論文: https://arxiv.org/abs/2510.01179
- License: Apache-2.0
- 495 MCP、2,000超のtool、150万超trajectory。single/parallel/multi-step、
  multi-turnと実tool responseを含む。
- 公開configは`Kimi-K2`、`OSS`、`Qwen3`、`SFT`。
- Colabから`SFT`をstreamingで64件実測した。
  - fields: `uuid`, `subset_name`, `question`, `target_tools`, `tools`, `messages`
  - `tools`と`messages`はJSON文字列。
  - message文字数 min/median/max: 1,501 / 4,693 / 38,806
  - turns min/median/max: 7 / 11 / 25
  - `tool_call`の連続によるparallel callと、複数roundのsequential callを確認。
- 注意: syntheticであり、公開MCPの接続失敗trajectoryやteacher由来誤りを含みうる。
  task/eval contaminationとtool responseの時点依存を別途検査する。

### Team-ACE/ToolACE

- 配布: https://huggingface.co/datasets/Team-ACE/ToolACE
- 論文: https://arxiv.org/abs/2409.00920
- License: Apache-2.0
- 約11.3k、英語・中国語、2から12 turns。
- function名の空白・括弧、definition/call不一致が報告されているため、主データではなく
  形式多様性の補助に限定する。

### Salesforce/xlam-function-calling-60k

- 配布: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- 論文: https://arxiv.org/abs/2406.18518
- License: CC BY 4.0、gated。
- 60kの検証済みsingle-turn/parallel function callingとして有用だが、Colabの
  unauthenticated実行では再現不能なため今回の自動matrixから外す。

### Salesforce/APIGen-MT-5k

- 配布: https://huggingface.co/datasets/Salesforce/APIGen-MT-5k
- 論文: https://arxiv.org/abs/2504.03601
- License: CC BY-NC 4.0、gated。
- retail/airlineの5k multi-turn trajectory、最大56 turns。形式・実行・policy・意味の
  段階検証があるが、非商用制約を主比較へ持ち込まないため0件とする。

### ACC: Agent Context Compilation

- 配布: https://huggingface.co/datasets/groundhogLLM/ACC-dataset
- 論文: https://arxiv.org/abs/2605.21850
- License: Apache-2.0
- search 3,369、SWE 4,368、SQL 3,065の計10,802 long-context QA。
- tool trajectoryを100k級のevidence-grounded QAへcompileする発想は長文評価に適するが、
  Phase 2の初期contextへ単純truncateすると学習目標を壊すため保留する。

### pyromind/agentic-tool-call-dataset-12k

- 配布: https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k
- License表記: Apache-2.0。Toucan等の二次変換データ。
- short 10kは平均8 turns/3 calls、long 2kは平均82 turns/40 calls。
- Colabの`datasets==4.0.0`ではdataset cardの`Json` featureを解釈できず直接load不可。
  upstream Toucanを直接使えるため、今回この二次変換へ依存しない。

## Phase 2暫定mix

`hnet/sft/configs/phase2_agent_proxy.json`を使用する。

- 日本語instruction/reasoning/chat: 8,000 examples
- English: 1,000
- Code: 1,000
- Toucan SFT: 2,500
- ToolACE: 500
- xLAM/APIGen-MT: 0

tool比率を通常SFTより高い約23%にして、短いSFTでagent転移傾向を見やすくする。
Phase 2の全候補へ同一example順、同一chat template、同一step数で適用する。

## 未完了の検査

- Toucan SFT全体の言語分布、byte長分位、turn/call分位。
- JSON parse失敗率、tool definition/call name一致率、最終assistant欠落率。
- BFCL/MCP-Universeおよび自作agent evalとの重複・類似度検査。
- 2k/4k/8k/16k context bucket別の保持率と、truncateせずpackingする規則。
- tool responseをloss対象に含めるか、assistant spanだけをsuperviseするかのablation。
