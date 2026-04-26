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
# AgentState用于智能体 MessagesState用于工作流中
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```



### WorkFlow（工作流）

总结：Agent是一种特殊的节点（Node），而工作流是整个图的运行流程（Graph）。Agent负责“思考+行动”，工作流负责“编排+调度”

LangGraph 本身就是一个**专门用来构建 AI Agent 工作流**的框架——你把模型的“思考”、“调用工具”、“回复用户”等步骤串起来，形成一个自动化的智能体工作流。

* State状态

工作流中的状态：

1. 注意存放整个聊天历史记录，以及需求中需要存的一些东西
2. 整个状态在所有节点函数、所有的边函数，输入值都默认有这个状态
3. 所有的节点返回的内容，主要目的就是为了更新状态

```python
# Reducer本质上是一个纯函数
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyState(TypedDict):
    # 对 messages 字段使用 add_messages 这个内置 reducer
    messages: Annotated[list, add_messages]
    # 对 counter 字段不指定 reducer，将使用默认的"覆盖"行为
    counter: int
这个例子中，任何节点对 counter 的更新都会覆盖旧值，但对 messages 的更新则会由 add_messages 进行合并操作
add_messages意思：
追加：当新消息的 id 与现有消息列表中的任何 id 都不匹配时，新消息会被追加到列表末尾（这看起来像追加）。
替换：如果新消息的 id 与列表中某条消息的 id 相同，则会用新消息替换那条旧消息（而不是追加）。
删除：通过传入一个 RemoveMessage 对象（其中包含要删除的消息的 id），可以从列表中删除对应的消息。
```

| 字段                  | 作用       | 更新方式         | 是否持久化   | 是否可输入         | 典型用途             |
| :-------------------- | :--------- | :--------------- | :----------- | :----------------- | :------------------- |
| `messages`            | 对话历史   | 追加             | ✅ 是         | ✅ 是               | 存储所有消息         |
| `jump_to`             | 流程控制   | 每次覆盖（默认） | ❌ 否（临时） | ✅ 是               | 决定下一步跳转的节点 |
| `structured_response` | 结构化输出 | 每次覆盖         | ✅ 是         | ❌ 否（仅内部设置） | 返回解析后的结果对象 |

* 节点Node

节点通常是Python函数，第一个位置参数是状态（可选），第二个位置参数config（可选）

```python
def my_other_node(state:State):
  return state
def my_node(state:State,config:RunnableConfig):
  return {"results":f"Hello,{state['input']}"}

start节点：start节点是一个特殊节点，代表将用户输入发送到图的节点。
end节点：是一个特殊节点，代表一个终止的节点。
```

* 边Edge

边定义了逻辑如何路由以及图如何决定停止。这是代理工作方式以及不同节点之间如何通信的重要组成部分。

1. 普通边：直接从一个节点到下一个节点
2. 条件边：调用函数以及确定接下来要转向哪些节点

#### 新旧版本对照

| 旧版 (0.x)                                      | 新版 (1.0+)                                         |
| :---------------------------------------------- | :-------------------------------------------------- |
| 手动定义 `StateGraph(State)`                    | 无需显式状态类，用普通函数参数传递                  |
| `add_node`, `add_edge`, `add_conditional_edges` | `@task` 定义任务，`@entrypoint` 编排逻辑            |
| 节点函数接收 `state` 并返回 `dict`              | 任务函数接收明确的参数，返回具体值                  |
| 条件边通过路由函数实现循环                      | 直接在 `@entrypoint` 中用 `while` 或 `for` 控制循环 |
| 显式编译 `graph = builder.compile()`            | 装饰器自动包装，返回可调用对象                      |
| 调用 `graph.invoke(input)`                      | 直接调用 `entrypoint.invoke(input)`                 |

| 场景             | 推荐使用       |
| ---------------- | -------------- |
| 简单线性工作流   | Functional API |
| 需要复杂条件分支 | 传统 API       |
| 需要精细控制状态 | 传统 API       |
| 快速原型开发     | Functional API |



1. MCP结合工作流

方法1 MCP->智能体中->工作流中的一个节点

方法2 自定义工具调用

| 功能          | BasicToolNode(自定义)          | ToolNode(官方)                           |
| ------------- | ------------------------------ | ---------------------------------------- |
| 工具调用逻辑  | 需手动解析tool_calls并执行工具 | 自动解析tool_calls,支持同步/异步工具调用 |
| 状态管理      | 需要手动封装ToolMessage        | 自动将结果封装为Tool Message并更新状态   |
| 错误处理      | 需手动捕获异常                 | 内置工具名称校验和异常处理机制           |
| Langgraph集成 | 需手动配置节点和边             | 深度集成StateGraph，支持tools_condition  |
| 性能优化      | 依赖手动实现的并发逻辑         | 内置并发调度和资源管理                   |

#### 人机协作

1. 使用interrupt暂停（中断）

   * 指定一个检查点器，用于在每一步之后保存图状态

   * 在适当位置调用interrupt()，第二种：转入interrupt_before[节点名称]

   * 使用线程ID运行图，直到触发interrupt

   * 使用invoke/ainvoke/stream/astream

     a. 恢复执行：interrupt_before--->agent.astream(None,config,stream_mode='values')

     b. 恢复执行：interrupt()--->agent.astream(Command(resume={"answer":user_input}),config, stream_mode='values')

2. 















































