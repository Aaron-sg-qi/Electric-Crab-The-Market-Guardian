# Electric-Crab-The-Market-Guardian
Prediction markets are becoming the world's truth machines—but who audits the truth? Electric Crab is an autonomous AI Agent that acts as the Market Guardian: detecting manipulation, quantifying risk, and providing explainable probability audits in real time. We are building the trust layer for the future prediction economy.

🦀 Electric Crab – The Market Guardian 预测市场风险审计和透明度智能体
Electric Crab 是一个面向 Polymarket 的多智能体市场风险守护者。它不是告诉用户“买什么”，而是帮助用户看清一个预测市场背后的概率偏差、流动性风险、巨鲸风险、Agent 分歧和信息透明度。

1. 项目背景

预测市场越来越重要，Polymarket 这类平台把公众事件、政治事件、体育赛事、加密市场都变成了可交易概率。

但普通用户看到的往往只有一个数字：

某事件发生概率：40%

问题是，这个 40% 到底可信吗？

它可能受到这些因素影响：

低流动性
巨鲸集中交易
短期情绪炒作
市场波动
信息不对称
少数大户操纵
新闻滞后

所以 Electric Crab 要解决的问题不是“再做一个预测模型”，而是：

让预测市场的风险变得可见。

2. 项目核心问题

用户在 Polymarket 上看到一个市场概率时，通常不知道：

这个价格是不是被低流动性扭曲了？
是否有巨鲸集中风险？
市场价格和 AI 模型判断差距多大？
不同分析 Agent 是否意见一致？
这个市场是否值得信任？
风险来自哪里？

Electric Crab 的目标就是回答这些问题。

3. 项目解决方案

Electric Crab 构建了一个 Multi-Agent Prediction Market Intelligence System。

系统会拉取市场数据，然后让多个专业 Agent 分别分析：

PriceAgent              读取市场隐含概率
LiquidityAgent          分析流动性风险
WhaleAgent              分析巨鲸集中风险
VolatilityAgent         分析价格波动风险
SentimentAgent          分析情绪信号
ClassicalMLAgent        使用传统机器学习估算概率
DeepLearningAgent       使用神经网络估算概率
GPUScoringAgent         使用并行批量评分
MultiAgentCoordinator   汇总所有 Agent 投票
XAPIGatewayAgent        生成 xapi.to 外部研究和通知任务

所以它不是一个黑箱模型，而是一个多智能体委员会。

4. 系统输出什么？

每个市场都会输出：

Market Probability
Electric Crab Probability
Deviation
Risk Level
Trust Score
Main Risk Factors
Consensus Level
Agent Disagreement
Agent Votes
Decision Signal
xapi.to Tasks
@Mention All Notification

这些字段对应评委最容易理解的价值：

Market Probability              市场认为的概率
Electric Crab Probability       AI 多智能体判断的概率
Deviation                       市场和 AI 判断差距
Risk Level                      风险等级
Trust Score                     市场可信度
Main Risk Factors               为什么有风险
Agent Votes                     每个 Agent 怎么判断
Consensus Level                 Agent 是否一致
xapi.to Tasks                   下一步外部研究和通知

# Multi-Agent Prediction Market Intelligence

Electric Crab 是一个面向预测市场的多智能体风险分析系统，结合 真实市场数据、多智能体预测、深度学习、强化学习式风险评分 和 xAPI 外部任务生成，帮助用户理解市场概率、风险水平、信任分数及主要因素，并生成透明化的研究任务与通知。

透明化：展示每个 Agent 的投票、信心分数和风险信号

多智能体：结合 ML、DL、RL、GPU 并行评分

xAPI 集成：生成研究任务与通知，展示应用能力

实时市场数据：可对真实 Polymarket 市场进行分析

# 1.多智能体预测

PriceAgent、LiquidityAgent、WhaleAgent、VolatilityAgent、SentimentAgent

ClassicalMLAgent、DeepLearningAgent、GPUScoringAgent

输出 Multi-Agent Probability、Deviation、Consensus Level

# 2.风险分析

RL 风格 Adaptive Risk Optimizer

Risk Level、Trust Score、主要风险因素

# 3.深度学习支持

PyTorch MLP 模型计算市场概率

GPU / CPU 并行批量评分

实时 Polymarket 数据

拉取公开市场事件并进行多智能体分析

# 4.xAPI 集成

生成 @Mention All 外部研究任务

支持 dry run 和真实发送模式

可导出 JSON 任务

⚙️ 安装

在项目根目录创建虚拟环境并激活：

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

安装依赖：

pip install -r requirements.txt

requirements.txt 示例：

numpy
scikit-learn
httpx
torch
fastapi
uvicorn
pydantic
streamlit
pandas
🧩 使用方法
1️⃣ 后端命令行
查看模拟市场（Demo Data）
python electric_crab_core.py
查看真实 Polymarket 数据
python electric_crab_core.py real
生成 xAPI @Mention All 任务（Dry Run）
python electric_crab_core.py notify
真正发送 xAPI 任务（需要设置 API Key）
set XAPI_API_KEY=你的_api_key  # Windows
export XAPI_API_KEY=你的_api_key # Linux / Mac

python electric_crab_core.py real notify send

2️⃣ 前端 Dashboard

使用 Streamlit 启动前端：

streamlit run app.py

浏览器打开 http://localhost:8501

Sidebar 设置

Data Source: Demo Data / Real Polymarket Data

Use Deep Learning: 开启或关闭深度学习预测

Use RL Risk Optimizer: 开启或关闭 RL 风格风险优化

Use GPU / Batch Scoring: GPU 并行评分

Generate @Mention All xapi.to Task: 生成任务

Try Real xapi.to Send: 实际发送（需配置 API Key）

Number of Real Markets: 设置拉取市场数量

页面展示

Market Audit Summary: 概览所有市场的概率、风险、Deviation、Trust Score

Detailed Market Cards: 每个市场详细信息

Market Probability / Electric Crab Probability / Deviation

Risk Level / Trust Score / Consensus Level

TL;DR / Decision Signal / Main Risk Factors

Agent Votes

xAPI Tasks / Raw xAPI Output / Decision Showcase

@Mention All Notification: 生成或发送外部任务

Export JSON: 导出所有市场分析和任务结果

📦 文件说明
electric_crab_core.py       # 核心逻辑：市场抓取、Agent 分析、多智能体融合、xAPI 集成
electric_crab_extensions.py # 扩展模块：Polymarket Collector、Deep Learning、RL Risk Optimizer、GPU Batch Scorer、并行工具
app.py                      # Streamlit 前端 Dashboard
requirements.txt            # Python 依赖

⚠️ 注意事项

Deep Learning / GPU Scoring 依赖 PyTorch

xAPI CLI 执行需配置 XAPI_API_KEY 和启用 CLI

当前系统用于 展示和教育目的，不提供投资建议

🔗 Demo 运行方式

先运行 Streamlit 前端，展示市场、风险、Agent Votes、xAPI 任务

使用 Demo Data 或 Real Polymarket Data 演示分析能力

展示 @Mention All 任务 与 JSON Export

强调多智能体决策的透明性和风险可解释性
