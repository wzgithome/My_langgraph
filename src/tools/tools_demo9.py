from typing import Annotated

from deep_agent.my_state import CustomState
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

# 把username修改到state中
@tool
def get_user_info_by_name(tool_call_id:Annotated[str,InjectedToolCallId],
        config:RunnableConfig) -> Command:
    """获取当前用户的username，以便生成祝福语句"""
    user_name=config.get("configurable").get("user_name",'zs')
    print(f"获取用户{user_name}的所有信息")
    return Command(
        update={
            "user_name":user_name,    # 更新状态中的用户名
            "messages":[
                ToolMessage(
                    content=f"获取用户{user_name}的所有信息成功",
                    tool_call_id=tool_call_id   # 指定工具调用的id(前一条消息的ID)
                )
            ]
        }
    )

@tool
def greet_user(state:Annotated[CustomState,InjectedState]) -> None:
    """获取用户的user_name之后，生成祝福语句"""
    user_name=state.get("user_name")
    return f"{user_name}，欢迎使用深度智能助手"


