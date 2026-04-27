import asyncio
import json
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
# 新版本
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from xarray.core.utils import alias_message

from deep_agent.env_utils import ZHIPU_API_KEY
from deep_agent.my_llm import model

"""
    人工干预中断，使用interrupt_before方法
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
    "url": "https://mcp.api-inference.modelscope.net/12f0774efe824f/sse",
    'transport': 'sse'
}

chart_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/80585a32fd6349/sse",
    'transport': 'sse'
}

mcp_client = MultiServerMCPClient({
    'chart_mcp': chart_mcp_server_config,
    'my12306_mcp': my12306_mcp_server_config,
    'zhipu_mcp':zhipu_mcp_server_config,
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
    memory=MemorySaver()
    graph = build.compile(checkpointer=memory,interrupt_before=['tools'])
    return graph




async def run_graph():
    graph=await create_graph()

    config={
        "configurable":{
            "thread_id":"zs1122"
        }
    }

    # 人工介入给予的回答
    def get_answer(tool_message,user_answer):
        """让人工介入，并且给一个问题的答案"""
        tool_name=tool_message.tool_calls[0]['name']

        answer=(
            f"人工强制终止了工具：{tool_name}的执行，拒绝的理由是：{user_answer}"
        )

        new_message=[
            ToolMessage(content=answer,tool_call_id=tool_message.tool_calls[0]['id']),
            AIMessage(content=answer)
        ]
        graph.update_state(
            config=config,
            values={'messages':new_message}
        )


    def print_message(event,result):
        """格式化输出消息"""
        messages=event.get('messages')
        if messages:
            if isinstance(messages,list):
                message=messages[-1]  # 如果消息是列表，则取最后一个
            if message.__class__.__name__=='AIMessage':
                if message.content:
                    result=message.content
            msg_repr=message.pretty_repr(html=True)
            if len(msg_repr)>1500:
                msg_repr=msg_repr[:1500]+"...(已截断)"
            print(msg_repr)   # 输出消息的表示形式

        return result

    async def execute_graph(user_input:str)->str:
        """执行工作流的函数"""
        result='' # AI助手的最后一条消息
        if user_input.strip().lower()!='y': #正常的用户提问
            current_state=graph.get_state(config)
            if current_state.next:  # 如果有下一步，则当前工作流处在中段中
                # 状态中存储的最后一条message
                tools_script_message=current_state.values['messages'][-1]
                get_answer(tools_script_message,user_input)
                message=graph.get_state(config).values['messages'][-1]
                result=message.content
                return result
            else:
                async for chunk in graph.astream(
                        {'messages':('user',user_input)},config,stream_mode='values'):
                    result=print_message(chunk,result)

        # 工具中断了，用户输入了y
        else: # 用户输入了Y想继续工具的调用
            async for chunk in graph.astream(None,config,stream_mode='values'):
                result=print_message(chunk,result)

        current_state=graph.get_state(config)
        if current_state.next:
            ai_message=current_state.values['messages'][-1]
            tool_name=ai_message.tool_calls[0]['name']

            result=(f'AI助手马上根据你要求，执行{tool_name}工具。您是否批准继续执行？输入y继续；'
                    f'否则，请说明您的理由。\n')

        return result



    while True:
        user_input=input('用户：')
        res=await execute_graph(user_input)
        print('AI:',res)


if __name__ == '__main__':
    asyncio.run(run_graph())























