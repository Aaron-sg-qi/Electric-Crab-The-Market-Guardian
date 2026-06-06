# 🦀 Electric Crab – The Market Guardian

# 电蟹：真实预测市场的多智能体风险审计系统

**Multi-Agent AI Prediction Market Risk Auditor**
**面向真实 Polymarket 市场、真实 xAPI 外部任务、可选链上存证的 AI Agent 系统**

![Electric Crab](EC.png)

---

## 1. 项目介绍

**Electric Crab 是一个面向真实 Polymarket 预测市场的多智能体 AI 风险审计系统。**

它会拉取真实预测市场数据，通过多个专业 AI Agent 分析市场概率、流动性、波动、情绪、风险偏差和多智能体分歧，并生成可解释的审计结果。

系统还可以生成并真实发送 xAPI / xapi.to 外部研究和通知任务，并可选地将预测哈希发布到链上，形成可验证的预测凭证。

## 2. 项目愿景

预测市场正在成为新的事实发现机制。

在 Polymarket 这样的市场中，政治事件、体育赛事、加密资产、AI 进展、公共事件都可以被转化为可交易概率。

用户看到的往往只是一个数字：

```txt
YES: 40%
NO: 60%
```

但这个数字背后可能隐藏着：

* 低流动性
* 巨鲸集中下注
* 短期情绪炒作
* 市场异常波动
* 信息不对称
* 新闻滞后
* 社交媒体叙事操纵
* 少量资金对薄弱市场的价格影响

Electric Crab 的目标不是再做一个下注工具，而是构建预测市场的：

> **Market Guardian：市场守护者。**

它帮助用户回答：

> 这个市场概率可信吗？
> 风险来自哪里？
> AI 多智能体判断和市场价格是否偏离？
> 是否需要进一步外部研究？
> 这份预测是否可以被验证？

---

## 3. 为什么叫 Electric Crab？

在世界杯的故事里，章鱼保罗曾经凭借预测比赛结果成为“预言者”。

但今天，预测不再发生在水族箱里，而是发生在链上。

所以我们没有做一只链上的章鱼。

我们做了一只更适合预测市场的生物：

> **Electric Crab，链上的电蟹。**

它有两只蟹钳：

* 一只代表 **YES**
* 一只代表 **NO**

当真实预测市场出现时，Electric Crab 不凭直觉判断，而是通过多智能体系统进行抓取、拆解、分析、评分、共识判断和外部任务生成。

---

## 4. 比赛级核心亮点

### 真实市场数据

Electric Crab 可以连接真实 Polymarket Gamma API，拉取真实活跃市场，并对真实市场进行分析。

```bash
python electric_crab_core.py real
```

### 真实 xAPI / xapi.to 任务

Electric Crab 可以生成真实 xAPI 外部任务，用于：

* 实时市场研究
* Twitter / X 社交信号检索
* Crypto 价格查询
* 审计通知
* `@Mention All` 通知发送

```bash
python electric_crab_core.py real notify send
```

### 多智能体审计

系统不是单一黑箱模型，而是由多个 Agent 组成的市场审计委员会。

每个 Agent 给出独立判断、概率、信心分数和解释。

### 可解释风险评分

Electric Crab 会输出：

* Market Probability
* Electric Crab Probability
* Predicted Outcome
* Outcome Confidence
* Deviation
* Risk Level
* Trust Score
* Main Risk Factors
* Agent Votes
* Consensus Level
* xAPI Tasks
* Prediction Hash

### 可选链上哈希存证

系统可以把预测结果生成 SHA-256 哈希，并通过 `electric_crab_chain.py` 发布到 EVM 兼容链上。

链上只保存哈希，不上传完整预测内容，从而实现低成本、隐私友好、可验证的预测凭证。

---

## 5. 系统架构

Electric Crab 是一个：

```txt
Real Market Data
      ↓
Multi-Agent Analysis
      ↓
Probability Audit
      ↓
Risk Scoring
      ↓
xAPI External Tasks
      ↓
Optional On-Chain Hash Proof
```

整体流程：

1. 从真实 Polymarket 市场拉取数据
2. 将市场数据交给多个 Agent 分析
3. 聚合 Agent 投票，形成多智能体共识
4. 输出 YES / NO 判断和置信度
5. 计算市场概率与模型概率偏差
6. 生成风险等级和信任评分
7. 生成 xAPI 外部研究和通知任务
8. 可选生成预测哈希并上链存证
9. Dashboard 展示完整审计结果
10. 导出 JSON 报告

---

## 6. Agent 组成

| Agent                   | 作用                            |
| ----------------------- | ----------------------------- |
| `PriceAgent`            | 读取真实市场隐含概率，作为基础价格信号           |
| `LiquidityAgent`        | 判断流动性是否足够，市场是否容易被操控           |
| `WhaleAgent`            | 分析大户集中风险和潜在操纵风险               |
| `VolatilityAgent`       | 监测市场价格波动和异常不稳定性               |
| `SentimentAgent`        | 分析情绪信号和叙事热度                   |
| `ClassicalMLAgent`      | 使用传统机器学习估算概率                  |
| `DeepLearningAgent`     | 使用 PyTorch 神经网络估算概率           |
| `GPUScoringAgent`       | 使用 GPU / CPU 批量评分，支持多市场监控     |
| `MultiAgentCoordinator` | 汇总所有 Agent 投票，形成多智能体共识        |
| `XAPIGatewayAgent`      | 生成真实 xAPI / xapi.to 外部研究和通知任务 |

---

## 7. 系统输出

每个真实市场都会生成一份完整审计结果。

| 输出字段                        | 含义                           |
| --------------------------- | ---------------------------- |
| `Market Probability`        | 真实市场隐含概率                     |
| `Electric Crab Probability` | 多智能体 AI 模型估算概率               |
| `Predicted Outcome`         | Electric Crab 判断更偏向 YES 或 NO |
| `Outcome Confidence`        | YES / NO 判断置信度               |
| `Deviation`                 | 市场概率与模型概率之间的偏差               |
| `Risk Level`                | 风险等级：LOW / MEDIUM / HIGH     |
| `Trust Score`               | 市场可信度评分                      |
| `Main Risk Factors`         | 主要风险来源                       |
| `Consensus Level`           | 多 Agent 共识程度                 |
| `Agent Disagreement`        | Agent 判断分歧程度                 |
| `Agent Votes`               | 每个 Agent 的投票、信心和解释           |
| `Decision Signal`           | 市场可能被低估、高估或基本合理              |
| `xapi.to Tasks`             | 外部研究、社交信号和通知任务               |
| `Prediction Hash`           | 可选预测哈希，用于链上或链下验证             |

---

## 8. 真实 Polymarket 数据模式

Electric Crab 支持真实 Polymarket 数据模式。

运行：

```bash
python electric_crab_core.py real
```

系统会尝试从 Polymarket Gamma API 获取活跃市场，并进行多智能体审计。

真实模式会输出：

* 市场标题
* 市场隐含概率
* 成交量
* 流动性信息
* Electric Crab 多智能体概率
* YES / NO 判断
* 风险等级
* 信任评分
* Agent Votes
* xAPI 外部任务
* Prediction Hash

说明：当前 MVP 中，部分高级字段如巨鲸集中度、社交情绪、链上钱包行为可能需要外部 API 或 xAPI 任务进一步补充。系统会在数据质量字段中区分真实数据、启发式估计和待外部验证信号。

---

## 9. 真实 xAPI / xapi.to 发送模式

Electric Crab 不只是生成本地分析结果，还可以生成并发送 xAPI 任务。

### Dry Run 模式

只生成任务，不真实发送：

```bash
python electric_crab_core.py real notify
```

### Real Send 模式

配置环境变量后，尝试真实发送 xAPI 任务：

Windows：

```bash
set XAPI_API_KEY=your_api_key
set XAPI_ENABLE_CLI=true
python electric_crab_core.py real notify send
```

Linux / Mac：

```bash
export XAPI_API_KEY=your_api_key
export XAPI_ENABLE_CLI=true
python electric_crab_core.py real notify send
```

真实发送模式可以用于：

* 审计结果通知
* `@Mention All`
* 外部研究任务分发
* 社交信号检索任务
* 市场跟踪任务

---

## 10. xAPI 任务示例

Electric Crab 会为真实市场生成类似任务：

```txt
/xapi
搜索这个预测市场相关的最新公开信息，并总结可能影响概率的关键因素：
Will BTC close above $100K this month?
```

社交信号任务：

```txt
/xapi
搜索 Twitter/X 上关于「Will BTC close above $100K this month?」的最新讨论，
提取情绪、热度和关键观点。
```

审计通知任务：

```txt
@Mention All

Electric Crab Multi-Agent Polymarket Audit Completed.

Audited markets: 5
Risk summary: HIGH=1, MEDIUM=3, LOW=1

Top results:
1. Will BTC close above $100K this month?
   Prediction: YES
   Market Probability: 54.2%
   Electric Crab Probability: 63.8%
   Risk: MEDIUM
   Trust Score: 72.4
```

---

## 11. Dashboard 前端

Electric Crab 提供 Streamlit Dashboard。

启动：

```bash
streamlit run app.py
```

浏览器打开：

```txt
http://localhost:8501
```

Dashboard 支持：

* 真实 Polymarket 数据模式
* Demo 数据模式
* Deep Learning 开关
* RL Risk Optimizer 开关
* GPU / Batch Scoring 开关
* xAPI 任务生成开关
* xAPI 真实发送开关
* 真实市场数量选择
* JSON 审计报告导出

---

## 12. Dashboard 展示内容

Dashboard 会展示：

* Market Audit Summary
* 每个市场的详细审计卡片
* YES / NO 判断
* Outcome Confidence
* Market Probability
* Electric Crab Probability
* Deviation
* Risk Level
* Trust Score
* Consensus Level
* Agent Disagreement
* Main Risk Factors
* Agent Votes
* xAPI Tasks
* Raw xAPI Output
* Decision Showcase
* Prediction Proof
* Data Quality
* Export JSON

这使评委可以直观看到：

> Electric Crab 如何从真实市场数据出发，经过多 Agent 审计，输出可解释风险判断，并生成外部任务和可验证结果。

---

## 13. 可选链上 Hash Proof

Electric Crab 支持可选链上哈希存证。

核心思路：

> 不把完整预测结果上链，只把预测结果哈希上链。

流程：

1. Electric Crab 生成完整预测审计 JSON
2. 对预测 payload 生成 SHA-256 哈希
3. 使用 `electric_crab_chain.py` 将哈希发布到 EVM 兼容链
4. 返回链上交易哈希
5. 后续用户可以用同一份 payload 重新计算哈希并验证

安装 Web3：

```bash
pip install web3
```

配置环境变量：

Windows PowerShell：

```powershell
$env:CHAIN_RPC_URL="https://your-rpc-url"
$env:CHAIN_PRIVATE_KEY="0xYOUR_PRIVATE_KEY"
$env:CHAIN_ACCOUNT_ADDRESS="0xYOUR_WALLET_ADDRESS"
$env:CHAIN_CONTRACT_ADDRESS="0xDEPLOYED_CONTRACT_ADDRESS"
```

发布预测哈希：

```bash
python electric_crab_chain.py publish demo-001 <prediction_hash>
```

发布并等待确认：

```bash
python electric_crab_chain.py publish-wait demo-001 <prediction_hash>
```

读取链上记录：

```bash
python electric_crab_chain.py get demo-001
```

这种方式可以证明：

* 预测在结果发生前已经生成
* 预测内容没有被事后篡改
* 完整预测内容可以留在链下
* 链上只保存轻量级证明

---

## 14. 项目结构

```txt
Electric-Crab-The-Market-Guardian/
│
├── app.py
│   └── Streamlit Dashboard 前端
│
├── electric_crab_core.py
│   └── 核心逻辑：真实市场抓取、多 Agent 分析、风险评分、xAPI 集成
│
├── electric_crab_extensions.py
│   └── 扩展模块：Polymarket Collector、Deep Learning、RL Optimizer、GPU Scorer、xAPI Client
│
├── electric_crab_chain.py
│   └── 可选链上预测哈希发布模块
│
├── requirements.txt
│   └── Python 依赖
│
├── EC.png
│   └── 项目图片
│
└── README.md
```

---

## 15. 安装方式

建议使用 Python 3.10 或以上版本。

创建虚拟环境：

```bash
python -m venv venv
```

Windows 激活：

```bash
venv\Scripts\activate
```

Linux / Mac 激活：

```bash
source venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

推荐 `requirements.txt`：

```txt
numpy
pandas
scikit-learn
httpx
streamlit
starlette
torch
web3
```

如果演示环境不适合安装 PyTorch，可以使用轻量版本：

```txt
numpy
pandas
scikit-learn
httpx
streamlit
starlette
web3
```

系统会自动使用 fallback scoring，仍可完成真实市场审计和 Dashboard 展示。

---

## 16. 比赛演示启动方式

### 启动 Dashboard

```bash
streamlit run app.py
```

打开：

```txt
http://localhost:8501
```

在 Sidebar 中选择：

```txt
Data Source: Real Polymarket Data
```

然后点击：

```txt
Run Electric Crab Audit
```

这会拉取真实 Polymarket 市场并进行审计。

---

## 17. 命令行运行方式

### 真实 Polymarket 市场审计

```bash
python electric_crab_core.py real
```

### 真实市场 + xAPI 通知任务

```bash
python electric_crab_core.py real notify
```

### 真实市场 + 真实 xAPI 发送

Windows：

```bash
set XAPI_API_KEY=your_api_key
set XAPI_ENABLE_CLI=true
python electric_crab_core.py real notify send
```

Linux / Mac：

```bash
export XAPI_API_KEY=your_api_key
export XAPI_ENABLE_CLI=true
python electric_crab_core.py real notify send
```

### 启动 Dashboard

```bash
streamlit run app.py
```

### 发布预测哈希到链上

```bash
python electric_crab_chain.py publish-wait <event_id> <prediction_hash>
```

---

## 18. 输出示例

```txt
Market: Will BTC close above $100K this month?

Prediction: YES
Outcome Confidence: 63.8%

Market Probability: 54.2%
Electric Crab Probability: 63.8%
Deviation: 9.6%

Risk Level: MEDIUM
Trust Score: 72.4

Consensus Level: MEDIUM_CONSENSUS
Agent Disagreement: 0.094

Main Risk Factors:
- Moderate probability deviation
- High market volatility
- Extreme sentiment signal

Decision Signal:
MARKET_MAY_BE_UNDERPRICED
```

中文解释：

```txt
市场：BTC 本月是否会收于 100K 美元以上？

Electric Crab 判断：YES
判断置信度：63.8%

市场概率：54.2%
Electric Crab 多智能体概率：63.8%
概率偏差：9.6%

风险等级：MEDIUM
信任评分：72.4

Agent 共识：中等共识
Agent 分歧：0.094

主要风险因素：
- 中等概率偏差
- 高市场波动
- 极端情绪信号

决策信号：
市场可能被低估
```

---

## 19. 评委可以看到的价值

Electric Crab 的比赛价值在于：

### 1. 使用真实预测市场数据

不是静态 mock 页面，而是可以拉取真实 Polymarket 市场并分析。

### 2. 多智能体透明审计

不是一个黑箱模型，而是多个 Agent 的协同判断。

### 3. 风险可解释

系统会告诉用户风险来自哪里，而不是只输出一个概率。

### 4. xAPI 外部任务集成

系统可以生成并真实发送外部研究和通知任务，连接 AI 分析与外部行动。

### 5. 可选链上证明

通过预测哈希上链，可以证明预测结果在某个时间点前已经存在。

### 6. Dashboard 可演示

Streamlit 前端可以直接展示完整产品体验。

## 20. 最终愿景

Electric Crab 是预测市场经济中的 Market Guardian。

它像一只蹲在链上的电蟹，举着两只蟹钳：

```txt
YES / NO
```

但它不靠直觉预测。

它通过真实市场数据、多智能体 AI、风险评分、xAPI 外部任务和可选链上哈希存证来审计预测市场。

它帮助用户看清：

* 哪些概率可信
* 哪些市场高风险
* 哪些价格可能被扭曲
* 哪些事件需要进一步研究
* 哪些预测结果可以被验证

Electric Crab 不是另一个预测市场。

它是预测市场的信任层。

**Electric Crab – The Market Guardian**
**谁来审计事实？电蟹来审计。**
