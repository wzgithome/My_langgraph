from langchain.agents import AgentState
from langgraph.graph import MessagesState


# 已经废弃
# from langgraph.prebuilt.chat_agent_executor import AgentState

# 自定义的智能体的状态类
class CustomState(AgentState):
    user_name: str








