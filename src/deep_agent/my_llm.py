import os
import dotenv
from langchain_openai import ChatOpenAI

# 定义的模型类
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL2")


# ================== 2. 初始化模型和工具 ==================
model = ChatOpenAI(
    model='qwen3.6-flash',
    temperature=0.6,
    # 开启深度思考
    # extra_body={"stream": True},
    # 启用联网搜索功能
    # extra_body={"enable_search": True},
)


