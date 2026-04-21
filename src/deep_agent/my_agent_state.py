
"""
    短期存储：线程级存储（会话级）

    长期存储：跨线程存储（跨会话）


"""

import os
import dotenv
from langchain.agents import create_agent
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from tools.tools_demo6 import runnable_tool

# ================== 1. 环境配置 ==================
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2", "")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL2", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")



# ================== 2. 初始化模型和工具 ==================
model = ChatOpenAI(model='MiniMax-M2.5', temperature=0.6)


# 获取前三条网页记录
search = TavilySearchResults(max_results=3)

tools=[search,runnable_tool]

# 短期记忆， 保存到内存中
'''
在部署到 LangGraph API 时，不能自己传入 checkpointer，
因为 API 平台会自动管理持久化（使用 Postgres 等数据库），
你传的自定义 checkpointer 会被忽略，甚至直接导致加载失败。
'''
# checkpointer=InMemorySaver()

DB_URL='postgresql://postgres:root@localhost:5432/langgraph_db'

with PostgresSaver.from_conn_string(DB_URL) as checkpointer:

    agent = create_agent(
        model=model,
        system_prompt='你是一个智能助手',
        checkpointer=checkpointer,
        tools=tools
    )
    config={
        "configurable":{
            "thread_id":"user_1"
        }
    }

    # 从短期存储中，返回所有当前会话的上下文
    rest=list(agent.get_state(config))
    print(rest)

    res1=agent.invoke(
        {"messages":[{"role":"user","content":"北京天气怎么样"}]},config=config
    )
    print(res1['messages'][-1].content)

    res2 = agent.invoke(
        {"messages": [{"role": "user", "content": "上海呢"}]}, config=config
    )
    print(res2['messages'][-1].content)

