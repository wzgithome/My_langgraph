from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
import dotenv


def get_weather(city:str)->str:
    """根据城市查询天气"""
    return f'今天{city}是晴天，气温在28度'

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL2")
model = ChatOpenAI(
    model='MiniMax-M2.5',
    temperature=0.6
)

agent=create_agent(
    model=model,
    tools=[get_weather],
    system_prompt='你是一个天气助手'
)

