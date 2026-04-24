from typing import TypedDict, Literal

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph

from my_llm import model
from pydantic import Field
from pydantic.v1 import BaseModel


# 自定义状态
class State(TypedDict):
    joke: str         # 生成的冷笑话内容
    topic: str          # 用户指定的主题
    feedback: str       # 改进内容
    funny_or_not: str   # 幽默评级
# 结构化输出模型类（用于LLM评估反馈）
class Feedback(BaseModel):
    """使用此工具结构化你的响应"""
    grade:Literal["funny","not funny"]=(
        Field(description='判断笑话是否幽默',examples=['funny','not funny']))
    feedback: str=Field(
        description='若不幽默，提供改进建议',
        examples='可以加入双关语或意外结局'
    )


# 节点函数1
def generator_func(state:State):
    """由大模型生成一个冷笑话的节点"""
    # 这段代码是一个条件表达式（三元运算符），如果 state.get("feedback", None) 为真（即不是 None、空字符串、0等假值），则使用 A；否则使用 B。
    prompt=(
        f"根据反馈改进笑话：{state['feedback']}\n主题：{state['topic']}"
        if state.get("feedback",None)
        else f"创作一个关于{state['topic']}的笑话"
    )
    # 第一种写法
    # resp=model.invoke(prompt)
    # return {'joke':resp.content}

    # 第二种写法
    chain=model | StrOutputParser
    resp=chain.invoke(prompt)
    return {'joke':resp}

# 节点函数2
def avaluator_func(state:State):
    """由大模型评估冷笑话的幽默评级节点"""


    # 第二种写法
    chain=model | StrOutputParser
    resp=chain.invoke(prompt)
    return {'joke':resp}



# 构建一个工作流

builder=StateGraph(State)
builder.add_node('generator',generator_func)
builder.add_node('avaluator',avaluator_func)


