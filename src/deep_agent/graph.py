from typing import List

from deep_agent.my_state import CustomState
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import os
import dotenv
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from tools.tools_demo8 import get_user_info
from tools.tools_demo3 import calculate
from tools.tools_demo6 import runnable_tool
from tools.tools_demo7 import MySearchTool
from tools.tools_demo9 import *
from langchain_community.tools import TavilySearchResults

# ================== 1. 环境配置 ==================
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2", "")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL2", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# 创建网络搜索的工具
# my_search_tool=MySearchTool()

# ================== 2. 初始化模型和工具 ==================
model = ChatOpenAI(model='MiniMax-M2.5', temperature=0.6)
# 获取前三条网页记录
search = TavilySearchResults(max_results=3)
tools = [calculate, runnable_tool, search]  # 如需其他工具，加入列表


# ================== 3. 动态提示词生成函数 ==================
# def make_system_prompt(config: RunnableConfig) -> List[SystemMessage]:
#     """根据 config 中的 user_name 动态生成系统消息"""
#     user_name = config.get("configurable", {}).get("user_name", "zs")
#     print("make_system_prompt", user_name)
#     content = f"你是一个智能助手，你的名字是 {user_name}。"
#     return [SystemMessage(content=content)]


class DynamicSystemPromptMiddleware(AgentMiddleware):
    """在 Agent 每次调用模型前，动态注入系统提示词。"""
    '''
        节点式钩子：
            before_agent
            before_model
            after_model
            after_agent
    '''

    # ✅ 关键修改：在参数中显式声明 config
    def before_agent(self, state, config: RunnableConfig) -> dict | None:
        # 现在可以安全地从 config 参数中获取配置了
        user_name = config.get("configurable", {}).get("user_name", "访客")
        role = config.get("configurable", {}).get("role", "通用助手")
        print("DynamicSystemPromptMiddleware", user_name, role)

        dynamic_prompt = SystemMessage(
            content=f"你是一位{role}，名叫{user_name}。请提供专业帮助。"
        )
        # 将消息插入到列表开头
        return {"messages": [dynamic_prompt] + state["messages"]}


# create_agent 不支持动态 prompt 函数
agent = create_agent(
    model=model,
    tools=[calculate, runnable_tool,
           search, get_user_info,
           get_user_info_by_name,greet_user],
    # system_prompt='你是一个智能助手',
    middleware=[DynamicSystemPromptMiddleware()],
    state_schema=CustomState
)

# 调用同步方法
# agent.invoke()

# 调用异步方法
# agent.ainvoke()
