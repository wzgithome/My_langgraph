
from langgraph_sdk import get_sync_client
import asyncio
# 同步请求

client=get_sync_client(url="http://localhost:2024")


for chunk in client.runs.stream(
        None,
        "agent",
        input={
            "messages":[{"role":"user","content":"用户的年龄是多少？"}]
        },
        stream_mode="messages-tuple"
        # stream_mode="messages"
    ):
        # print(f' event of type:{chunk.event}...')
        # print(chunk.data)
        if isinstance(chunk.data,list) and 'type' in chunk.data[0] and chunk.data[0]['type']=='AIMessageChunk':
            print(chunk.data[0]['content'],end='|',flush=True)






