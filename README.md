# Deep Agents Template

Deployment template for a deep agent built with `create_deep_agent(...)`.

## What this template gives you

- A deployable deep agent graph at `src/deep_agent/graph.py`.
- Explicit workflow prompt (plan, delegate, critique, finalize).
- Two predefined sub-agents (`researcher`, `critic`).
- Human-in-the-loop interrupts on `execute` and `write_file`.
- A `uv`-managed local workflow with a small `Makefile` wrapper and starter tests.

## Prerequisites

- An API key for your model provider (Anthropic by default)
- A [LangSmith](https://smith.langchain.com/) account (Plus plan or higher) to deploy

## Quickstart

1. Sync the project and configure environment:

```bash
uv sync
cp .env.example .env
```

2. Start the dev server:

```bash
uv run langgraph dev
```

3. Deploy to LangSmith:

```bash
uv run langgraph deploy
```

4. 检查依赖项

```bash
pip install -e .
```

See the [CLI docs](https://docs.langchain.com/langsmith/cli#deploy) for deploy options.

To set up CI instead, push this repo to GitHub and configure your deployment through the LangSmith UI.

## Tests and lint

```bash
make test
make integration-tests
make lint
make format
```

Integration tests are skipped unless `ANTHROPIC_API_KEY` is set.

## Reference docs

- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents quickstart: https://docs.langchain.com/oss/python/deepagents/quickstart
- LangSmith CLI: https://docs.langchain.com/langsmith/cli

## 笔记

LangGraph默认遵循核心工作流：**接收输入 → 调用模型（思考）→ 可能调用工具 → 结束**

### Configurable配置

中间件就是为这个流程提供了一套标准化的“接口”：

1. **`before_agent`**: 整个智能体启动前，做最后的准备工作（如数据验证）。
2. **`wrap_model_call`**: “包裹”住调用模型的核心操作，可以在模型思考前后执行逻辑（如模型重试）。
3. **`after_model`**: 模型回答之后、行动之前，进行干预（如敏感操作审批）。

* Demo示例

  ```python
  from langchain.agents.middleware import AgentMiddleware
  from langgraph.runtime import Runtime
  
  # 定义一个简单的日志中间件
  class LoggingMiddleware(AgentMiddleware):
      async def __call__(self, run, agent, input, **kwargs):
          print(f"🤔 [日志] 用户的问题: {input}")
          # 调用智能体核心逻辑（必须）
          result = await run(agent, input, **kwargs) 
          print(f"🤖 [日志] 智能体的回答: {result}")
          return result
  
  # 在创建智能体时，通过 middleware 列表启用它
  agent = create_agent(
      model="your-model",
      tools=[...],
      middleware=[LoggingMiddleware()] # 像搭积木一样加上
  )
  ```

### AgentState详解

```python
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

| 字段                  | 作用       | 更新方式         | 是否持久化   | 是否可输入         | 典型用途             |
| :-------------------- | :--------- | :--------------- | :----------- | :----------------- | :------------------- |
| `messages`            | 对话历史   | 追加             | ✅ 是         | ✅ 是               | 存储所有消息         |
| `jump_to`             | 流程控制   | 每次覆盖（默认） | ❌ 否（临时） | ✅ 是               | 决定下一步跳转的节点 |
| `structured_response` | 结构化输出 | 每次覆盖         | ✅ 是         | ❌ 否（仅内部设置） | 返回解析后的结果对象 |
