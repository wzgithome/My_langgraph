# 网络搜索的工具
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.tools import TavilySearchResults
import os
import dotenv
dotenv.load_dotenv()

# 创建工具的方式

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
class SearchArgs(BaseModel):
    query:str=Field(description='需要进行网络搜索的信息')

class MySearchTool(BaseTool):
    name: str = "search_tool"
    description: str = "用于互联网搜索的工具"
    args_schema: type[BaseModel] = SearchArgs
    def _run(self, query: str) -> str:
        """使用网络搜索"""
        try:
            print(f"正在使用网络搜索，查询：{query}")
            search=TavilySearchResults()

            print('查询结果',search)
        except Exception as e:
            print(f"网络搜索发生错误：{e}")
            return '没有搜索到任何内容'


# my_tool=MySearchTool()
# print(my_tool.name)
# print(my_tool.description)
# print(my_tool.args_schema.model_json_schema())

















