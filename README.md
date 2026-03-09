# 九丘知识库问答系统

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

基于 **OpenAI** + **Milvus** + **FastAPI** 的本地 CRM 知识库 RAG 问答系统。

## 系统架构

```
 Markdown 文档
       │
       ▼
  文档解析 & 分块  (document_loader.py)
       │
       ▼
  OpenAI Embedding  (text-embedding-3-small)
       │
       ▼
  Milvus 向量存储  (本地 Docker)
       │
  ─────┼───── 用户查询
       │          │
       ▼          ▼
  向量检索  ←  问题向量化
       │
       ▼
  GPT-4o-mini 生成答案  (RAG)
       │
       ▼
  FastAPI + SSE 流式返回
       │
       ▼
  浏览器网页界面
```

## 项目结构

```
crm_kb/
├── app/
│   ├── config.py          # 全局配置（读取环境变量）
│   ├── document_loader.py # Markdown 解析与分块
│   ├── vector_store.py    # Milvus 连接 + Embedding + 检索
│   ├── rag.py             # RAG 问答核心（普通 + 流式）
│   ├── feishu_bot.py      # 飞书机器人长连接（WebSocket 持久连接）
│   └── main.py            # FastAPI 应用入口
├── scripts/
│   └── build_index.py     # 一键构建向量索引
├── static/
│   └── index.html         # 前端聊天界面
├── data/                  # 放置 CRM markdown 文档（可为空，自动向上查找）
├── docker-compose.yml     # Milvus + etcd + minio
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
└── start.sh               # 一键启动脚本
```

## 快速开始

### 1. 配置环境变量

```bash
cd /root/crm/crm_kb
cp .env.example .env
nano .env   # 填写 OPENAI_API_KEY
```

### 2. 一键启动

```bash
./start.sh
```

或者分步执行：

```bash
# 启动 Milvus
docker compose up -d
sleep 30  # 等待就绪

# 安装 Python 依赖
pip install -r requirements.txt

# 构建知识库索引（首次或文档更新后执行）
python scripts/build_index.py

# 启动 Web 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 访问

- **聊天界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/chat` | 普通问答 |
| `POST` | `/api/chat/stream` | 流式问答（SSE） |
| `POST` | `/api/index` | 重建知识库索引 |
| `GET` | `/api/stats` | 知识库统计 |

### 问答接口示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "最近有哪些客户进展？", "top_k": 5}'
```

## 文档管理

- 将 CRM Markdown 文件放入 `crm_kb/data/` 目录，或保留在上级 `/root/crm/` 目录
- 添加/更新文档后，运行 `python scripts/build_index.py` 或调用 `POST /api/index` 重建索引

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | 必填 | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 兼容第三方代理 |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | 问答对话模型 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 向量化模型 |
| `MILVUS_HOST` | `localhost` | Milvus 地址 |
| `MILVUS_PORT` | `19530` | Milvus 端口 |
| `MILVUS_COLLECTION` | `crm_knowledge_base` | 集合名称 |

---

## 飞书机器人接入

> **推荐方式：长连接（WebSocket）**
> 无需注册公网域名，无需配置加密策略。仅需在 `.env` 中填写 App ID / App Secret，
> 服务启动时 SDK 自动向飞书建立出站 WebSocket 长连接，断线自动重连。

### 架构说明

```
飞书用户 → @机器人 发送文本问题
    │
    ▼
飞书平台 ──WS Push──▶ lark_oapi.ws.Client（出站长连接，本地端发起）
                           │
                       _on_message() 事件处理器
                           │
                       线程池执行 RAG answer()
                           │
                       飞书 IM API 回复消息
```

### 配置步骤

#### 1. 创建飞书应用

1. 打开 [飞书开放平台](https://open.feishu.cn/app) → **创建企业自建应用**
2. 记录 **App ID** 和 **App Secret**

#### 2. 添加应用权限

进入应用 → **权限管理** → 开通以下权限：

| 权限 | 用途 |
|------|------|
| `im:message` | 读取接收到的消息 |
| `im:message:send_as_bot` | 以机器人身份回复消息 |
| `im:message:send_urgent_as_bot` | 发送应用消息（含卡片） |
| `cardkit:card:create` | 创建 Card Kit 流式卡片 |
| `cardkit:card:update` | 更新卡片内容（流式刷新） |

#### 3. 开启长连接（使用长连接接收事件）

进入应用 → **事件订阅** → 选择**「使用长连接接收事件」**

> 无需填写请求 URL，无需配置加密策略。

订阅事件：添加 **接收消息** (`im.message.receive_v1`)

#### 4. 配置 .env

```bash
# 飞书机器人（仅需两项）
FEISHU_APP_ID=cli_xxxxxxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### 5. 安装依赖

```bash
pip install lark-oapi
```

#### 6. 发布应用

进入应用 → **版本管理与发布** → 创建版本 → 提交审核（或在测试版中直接启用）

#### 7. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务启动时日志会输出：

```
[feishu] WebSocket 长连接客户端启动中…
[feishu] 飞书长连接线程已启动（daemon=True）
```

#### 8. 使用方式

- **私聊**：直接向机器人发送文本消息
- **群聊**：将机器人拉入群，发消息时 **@机器人名称** + 问题内容

**示例：**

```
@CRM助手 张三最近跟进了哪些客户？
```

机器人将自动回复 RAG 查询结果，并附上来源记录（最多 3 条）。

### 注意事项

- 长连接为**出站连接**，本地开发无需内网穿透，也无需公网 IP
- RAG 查询在独立线程池（4 并发）中执行，不阻塞 WebSocket 事件循环
- SDK 内部自动重连，进程存活期间连接始终保持
- 未配置 `FEISHU_APP_ID` 时服务正常启动，仅跳过飞书连接（不影响 Web API）
