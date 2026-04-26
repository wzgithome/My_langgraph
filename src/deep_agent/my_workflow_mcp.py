import asyncio
import json
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
# 新版本
from langchain_tavily import TavilySearch

from deep_agent.env_utils import ZHIPU_API_KEY
from deep_agent.my_llm import model

"""
    MCP查询12306车票以及生成图表（使用自定义BasicToolNode）
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


# 自定义工具调用
class BasicToolsNode:
    """
        异步工具节点，用于并发执行AIMessage中请求的工具调用

        功能：
            1.接收工具列表并建立名称索引
            2.并发执行消息中的工具调用请求
            3.自动处理同步/异步工具适配
    """

    def __init__(self, tools: list):
        """初始化工具节点
            Args：
                tools：工具列表，每个工具需包含name属性
        """
        self.tools_by_name = {tool.name: tool for tool in tools}  # 所有工具名字的集合

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        """异步调用入口
            Args：
                state：输入字典，需要包含"messages"
            Returns：
                包含ToolMessage列表的字典
            Raises：
                ValueError：当输入无效时抛出
            """
        # 输入验证
        if not (messages := state.get('messages')):
            raise ValueError('输入数据中未找到消息内容')
        messages: AIMessage = messages[-1]  # 取最新消息：AIMessage

        # 并发执行工具调用
        outputs = await self._execute_tool_calls(messages.tool_calls)
        return {"messages": outputs}

    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """执行实际工具调用
        Args：
            tool_calls:工具调用请求
        Returns：
            ToolMessage结果列表
        """

        async def _invoke_tool(tool_call: Dict) -> ToolMessage:
            """
                执行单个工具调用
                Args：
                    tool_call:工具调用请求字典，需包含name/args/id字段
                Returns：
                    封装的ToolMessage
                Raise：
                    KeyError：工具未注册时抛出
                    RuntimeError：工具调用失败时抛出
            """
            try:
                # 异步调用工具
                tool = self.tools_by_name.get(tool_call['name'])  # 验证 工具是否在之前的 工具集合中

                if not tool:
                    raise KeyError(f"未注册的工具：{tool_call['name']}")
                if hasattr(tool, 'ainvoke'):  # 优先使用异步方法
                    tool_result = await tool.ainvoke(tool_call["args"])
                else:  # 同步工具通过线程池转异步
                    loop = asyncio.get_running_loop()
                    tool_result = await loop.run_in_executor(
                        None,  # 使用默认线程池
                        tool.invoke,  # 同步调用方法
                        tool_call["args"]  # 参数
                    )
                # 构造ToolMessage
                return ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            except Exception as e:
                raise RuntimeError(f"工具调用失败：{tool_call['name']}") from e

        try:
            """
                并发执行所有工具调用
                asyncio.gather()是Python异步编程中用于并发调度多个协程的核心函数，其核心行为包括：
                并发执行：所有传入的协程会被同时调度到事件循环中，通过非阻塞I/O实现并行处理
                结果收集：按输入顺序返回所有协程的结果（或异常），与任务完成顺序无关
                异常处理：默认情况下，任一任务失败会立即取消其他任务抛出异常；若设置return_exceptions=True，则异常会作为结果抛出
            """
            return await asyncio.gather(*[_invoke_tool(tool_call) for tool_call in tool_calls])


        except Exception as e:
            raise RuntimeError(f"并发执行工具时发生错误") from e


# ========== 任务1 ==========


class State(MessagesState):
    pass


def route_tools_func(state: State):
    """
        动态路由函数，如果从大模型输出后的AIMessage中包含有工具调用的请求（指令），
        就进入到tools节点，否则结束
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get('messages', []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge:{state}")
    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


async def create_graph():
    tools = await mcp_client.get_tools()  # 30个以上的工具，全部来自MCP服务端

    build = StateGraph(State)
    llm_with_tools = model.bind_tools(tools)

    # 注意ainvoke是异步执行
    async def chatbot(state: State):
        return {'messages': [await llm_with_tools.ainvoke(state['messages'])]}

    build.add_node('chatbot', chatbot)

    tool_node = BasicToolsNode(tools)
    build.add_node('tools', tool_node)

    build.add_conditional_edges(
        "chatbot", route_tools_func,
        {
            "tools": "tools", END: END
        }
    )
    build.add_edge("tools", "chatbot")
    build.add_edge(START, "chatbot")
    graph = build.compile()
    return graph


agent = asyncio.run(create_graph())
