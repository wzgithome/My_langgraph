"""
    短期存储：线程级存储（会话级）

    长期存储：跨线程存储（跨会话）

    设置长期存储的前提是要设置短期存储
"""

# 长期存储示例
import os
import dotenv
from langchain.agents import create_agent
from langchain_core.stores import InMemoryStore
# 已弃用
# from langchain_community.tools import TavilySearchResults
# 新版本
from langchain_tavily import TavilySearch

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from tools.tools_demo6 import runnable_tool

# ================== 1. 环境配置 ==================
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2", "")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL2", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# ================== 2. 初始化模型和工具 ==================
model = ChatOpenAI(model='MiniMax-M2.5', temperature=0.6)

# 获取前三条网页记录
search = TavilySearch(max_results=3)

tools = [search, runnable_tool]

# 短期记忆， 保存到内存中
'''
在部署到 LangGraph API 时，不能自己传入 checkpointer，
因为 API 平台会自动管理持久化（使用 Postgres 等数据库），
你传的自定义 checkpointer 会被忽略，甚至直接导致加载失败。
'''
# checkpointer=InMemorySaver()     # 短期存储，保存在内存中
# store=InMemoryStore() # 长期存储


DB_URL = 'postgresql://esired:root@127.0.0.1:5432/langgraph_db'

with (
        # 引入长期存储
        PostgresStore.from_conn_string(DB_URL)as store ,
        PostgresSaver.from_conn_string(DB_URL)as checkpointer
) :
    # 首次使用存储时，需要调用，会执行建表，后面就不需要了
    # checkpointer.setup()
    # 首次使用和上方的一样
    store.setup()
    agent = create_agent(
        model=model,
        system_prompt='你是一个智能助手',
        checkpointer=checkpointer,
        store=store,
        tools=tools
    )
    # 会话a
    # config_a = {
    #     "configurable": {
    #         "user_id":"alice",
    #         "thread_id": "thread_A"
    #     }
    # }

    config_b = {
        "configurable": {
            "user_id": "alice",
            "thread_id": "thread_B"
        }
    }

    # 从短期存储中，返回所有当前会话的上下文
    # rest = list(agent.get_state(config))
    # print(rest)

    # 从长期存储中，返回所有当前会话的上下文
    # rest1=list(agent.get_state_history(config))
    # print(rest1)

    res1=agent.invoke(
        {"messages":[{"role":"user","content":"给我一个关于相声的报幕词"}]},config=config_b
    )
    print(res1['messages'][-1].content)

    res2 = agent.invoke(
        {"messages": [{"role": "user", "content": "再给我一个关于流行歌曲的"}]}, config=config_b
    )
    print(res2['messages'][-1].content)
