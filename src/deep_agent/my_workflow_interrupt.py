import asyncio
import json
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
# 新版本
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

from deep_agent.env_utils import ZHIPU_API_KEY
from deep_agent.my_llm import model

"""
    人工干预
"""

# 智谱MCP服务端的连接
zhipu_mcp_server_config={
    "url": "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization="+ZHIPU_API_KEY,
    'transport': 'sse'
}

bing_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/b0f5de625ded4f/sse",
    'transport': 'sse'
}
my12306_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/7de29b75b86e40/sse",
    'transport': 'sse'
}

chart_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/a22a82a319c649/sse",
    'transport': 'sse'
}

mcp_client = MultiServerMCPClient({
    'chart_mcp': chart_mcp_server_config,
    'my12306_mcp': my12306_mcp_server_config,
    # 'zhipu_mcp':zhipu_mcp_server_config
    # 'bing_mcp': bing_mcp_server_config
})


class State(MessagesState):
    pass


async def create_graph():
    tools = await mcp_client.get_tools()  # 30个以上的工具，全部来自MCP服务端

    build = StateGraph(State)
    llm_with_tools = model.bind_tools(tools)

    # 注意ainvoke是异步执行
    async def chatbot(state: State):
        return {'messages': [await llm_with_tools.ainvoke(state['messages'])]}

    build.add_node('chatbot', chatbot)

    # tool_node = BasicToolsNode(tools)
    tool_node = ToolNode(tools)
    # 节点名称tools必须
    build.add_node('tools', tool_node)

    build.add_conditional_edges(
        "chatbot", tools_condition,
    )
    build.add_edge("tools", "chatbot")
    build.add_edge(START, "chatbot")
    # interrupt_before在工具执行之前中断
    graph = build.compile(interrupt_before=['tools'])
    return graph




async def run_graph():
    graph=await create_graph()

    config={
        "configurable":{
            "thread_id":"zs1122"
        }
    }
    async def execute_graph(user_input:str)->str:
        """执行工作流的函数"""
        pass


    while True:
        user_input=input('用户：')
        res=await execute_graph(user_input)


























