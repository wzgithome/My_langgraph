import asyncio
import json
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from watchfiles import awatch

my12306_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/7de29b75b86e40/sse",
    'transport': 'sse'
}

chart_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/a22a82a319c649/sse",
    'transport': 'sse'
}

mcp_client=MultiServerMCPClient({
        'chart_mcp':chart_mcp_server_config,
        'my12306_mcp':my12306_mcp_server_config
    }
)

# 自定义工具调用
class BasicToolsNode:
    """
        异步工具节点，用于并发执行AIMessage中请求的工具调用

        功能：
            1.接收工具列表并建立名称索引
            2.并发执行消息中的工具调用请求
            3.自动处理同步/异步工具适配
    """
    def __init__(self,tools:list):
        """初始化工具节点
            Args：
                tools：工具列表，每个工具需包含name属性
        """
        self.tools_by_name={tool.name:tool for tool in tools} #所有工具名字的集合

    async def __call__(self, state:Dict[str,Any])->Dict[str,List[ToolMessage]]:
        """异步调用入口
            Args：
                state：输入字典，需要包含"messages"
            Returns：
                包含ToolMessage列表的字典
            Raises：
                ValueError：当输入无效时抛出
            """
        # 输入验证
        if not (messages:=state.get('messages')):
            raise ValueError('输入数据中未找到消息内容')
        messages: AIMessage=messages[-1] # 取最新消息：AIMessage

        # 并发执行工具调用
        outputs=await self._execute_tool_calls(messages.tool_calls)
        return {"messages":outputs}


    async def _execute_tool_calls(self,tool_calls:List[Dict])->List[ToolMessage]:
        """执行实际工具调用
        Args：
            tool_calls:工具调用请求
        Returns：
            ToolMessage结果列表
        """


        async def _invoke_tool(tool_call:Dict)->ToolMessage:
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
                tool=self.tools_by_name.get(tool_call['name']) # 验证 工具是否在之前的 工具集合中

                if not tool:
                    raise KeyError(f"未注册的工具：{tool_call['name']}")
                if hasattr(tool,'ainvoke'): # 优先使用异步方法
                    tool_result=await tool.ainvoke(tool_call["args"])
                else: # 同步工具通过线程池转异步
                    loop=asyncio.get_running_loop()
                    tool_result=await loop.run_in_executor(
                        None, # 使用默认线程池
                        tool.invoke, # 同步调用方法
                        tool_call["args"] # 参数
                    )
                # 构造ToolMessage
                return ToolMessage(
                    content=json.dumps(tool_result,ensure_ascii=False),
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

