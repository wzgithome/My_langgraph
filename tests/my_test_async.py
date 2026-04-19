
from langgraph_sdk import get_client
import asyncio

# 异步请求
client=get_client(url="http://localhost:2024")

async def main():
    async for chunk in  client.runs.stream(
        None,
        "agent",
        input={
            "messages":[{"role":"user","content":"给当前用户生成一个祝福词"}]
        },
        config={"configurable":{"user_name":"托尼史塔可"}}
    ):
        print(f' event of type:{chunk.event}...')
        print(chunk.data)
        print('\n\n')

if __name__ == '__main__':
    asyncio.run(main())



