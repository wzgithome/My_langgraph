# 工具Runnable对象创建工具

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import dotenv
import os

from pydantic import BaseModel, Field

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL2")

model = ChatOpenAI(
    model='MiniMax-M2.5',
    temperature=0.6
)
prompt=PromptTemplate.from_template("帮我生成一个简单的，关于{topic}的报幕词。要求：1、内容搞笑一点 2、输出内容才有{language}")

chain=prompt|model|StrOutputParser()

class ToolArgs(BaseModel):
    topic:str=Field(...,description='主题')
    language:str=Field(...,description='语言')

runnable_tool=chain.as_tool(
    name='generate_news_title',
    description='这个一个专门生成报幕词的工具',
    args_schema=ToolArgs
)

# print(runnable_tool.args_schema.model_json_schema())






