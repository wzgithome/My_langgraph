import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.func import task, entrypoint
from langgraph.prebuilt import ToolNode
from deep_agent.env_utils import ZHIPU_API_KEY
from deep_agent.my_llm import model
from langchain_mcp_adapters.client import MultiServerMCPClient


"""
    MCP配置使用最新版本的代码
"""
# ---------- MCP 配置 ----------
zhipu_mcp_server_config = {
    "url": "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=" + ZHIPU_API_KEY,
    "transport": "sse"
}
bing_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/b0f5de625ded4f/sse",
    "transport": "sse"
}
my12306_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/7de29b75b86e40/sse",
    "transport": "sse"
}
chart_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/a22a82a319c649/sse",
    "transport": "sse"
}

mcp_client = MultiServerMCPClient({
    "chart_mcp": chart_mcp_server_config,
    "my12306_mcp": my12306_mcp_server_config,
    # "zhipu_mcp": zhipu_mcp_server_config,
    # "bing_mcp": bing_mcp_server_config
})

# 全局缓存 tools 和 tool_node（异步初始化一次）
_tools_cache: Optional[List] = None
_tool_node_cache: Optional[ToolNode] = None


async def get_tools_and_node():
    global _tools_cache, _tool_node_cache
    if _tools_cache is None:
        tools = await mcp_client.get_tools()
        _tools_cache = tools
        _tool_node_cache = ToolNode(tools)
    return _tools_cache, _tool_node_cache


# ---------- 任务定义 ----------
@task
async def call_model(messages: List, tools: List) -> AIMessage:
    """调用 LLM，绑定工具"""
    llm_with_tools = model.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)
    return response


@task
async def execute_tools(ai_message: AIMessage, tool_node: ToolNode) -> List[ToolMessage]:
    """执行工具调用"""
    # ToolNode 的 invoke 方法是同步的，但内部会执行工具，可能包含异步
    # 注意：ToolNode.invoke 可以直接接收一个 AIMessage，返回 ToolMessage 列表
    result = tool_node.invoke({"messages": [ai_message]})
    return result["messages"]


# ---------- 入口工作流 ----------
@entrypoint()
async def agent_workflow(user_input: str) -> Dict[str, Any]:
    """MCP 智能体工作流：支持工具调用，自动循环直到没有工具调用"""
    # 1. 获取 MCP 工具和 ToolNode（首次调用时初始化）
    tools, tool_node = await get_tools_and_node()

    # 2. 初始化消息历史
    messages = [HumanMessage(content=user_input)]
    final_response = None

    # 3. 循环：调用模型 -> 如果有工具调用 -> 执行工具 -> 继续
    while True:
        ai_msg = await call_model(messages, tools).result()
        messages.append(ai_msg)

        # 检查是否有工具调用
        if not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
            final_response = ai_msg
            break

        # 执行工具
        tool_messages = await execute_tools(ai_msg, tool_node).result()
        messages.extend(tool_messages)

    # 4. 返回最终结果
    return {
        "messages": messages,
        "final_response": final_response.content if final_response else "",
    }


# ---------- 导出 graph（供 langgraph CLI 使用）----------
graph = agent_workflow