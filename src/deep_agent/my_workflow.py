from typing import TypedDict, Literal

from langchain_core.output_parsers import StrOutputParser
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from deep_agent.my_llm import model
from pydantic import Field
from pydantic import BaseModel


"""
    旧版本的工作流
"""

# 自定义状态
class State(TypedDict):
    joke: str  # 生成的冷笑话内容
    topic: str  # 用户指定的主题
    feedback: str  # 改进内容
    funny_or_not: str  # 幽默评级


# 结构化输出模型类（用于LLM评估反馈）
class Feedback(BaseModel):
    """使用此工具结构化你的响应"""
    grade: Literal["funny", "not funny"] = Field(
        description='判断笑话是否幽默',
        examples=['funny', 'not funny'])
    feedback: str = Field(
        description='若不幽默，提供改进建议',example='可以加入双关语或意外结局'
    )


"""
节点函数必须返回一个字典，而不能返回元组、列表或其他类型。
其根本原因在于：LangGraph 的状态更新机制是基于“字段名”的合并（merge）。
"""


# 节点函数1
def generator_func(state: State):
    """由大模型生成一个冷笑话的节点"""
    # 这段代码是一个条件表达式（三元运算符），如果 state.get("feedback", None) 为真（即不是 None、空字符串、0等假值），则使用 A；否则使用 B。
    prompt = (
        f"根据反馈改进笑话：{state['feedback']}\n主题：{state['topic']}"
        if state.get("feedback", None)
        else f"创作一个关于{state['topic']}的笑话"
    )
    # 第一种写法
    # resp=model.invoke(prompt)
    # return {'joke':resp.content}

    # 第二种写法
    chain = model | StrOutputParser()
    resp = chain.invoke(prompt)
    return {'joke': resp}


# 节点函数2
def avaluator_func(state: State):
    """由大模型评估冷笑话的幽默评级节点"""
    prompt = (
        f"评估此笑话的幽默程度：\n{state['joke']}\n"
        "注意：幽默应包含意外性或巧妙的措辞"
    )
    # 第一种：支持OpenAI
    # chain=model.with_structured_output(Feedback)
    # resp=chain.invoke(prompt)
    # return {
    #     'feedback': resp.feedback,
    #     'funny_or_not': resp.grade
    # }

    # 第二种：支持Qwen
    result = model.bind_tools([Feedback])
    resp = result.invoke(prompt)
    res = resp.tool_calls[-1]['args']
    return {
        'feedback': res['feedback'],
        'funny_or_not': res['grade']
    }


# 条件边的路由函数
def route_func(state: State) -> str:
    """动态路由决策函数"""
    # 方法1:直接返回
    # return END if state.get('funny_or_not') == "funny" else 'generator'

    # 方法2:使用映射返回
    return 'Accepted' if state.get('funny_or_not') == 'funny' else 'Rejected+Feedback'



# 构建一个工作流
builder = StateGraph(State)
builder.add_node('generator', generator_func)
builder.add_node('avaluator', avaluator_func)

builder.add_edge(START, 'generator')
builder.add_edge('generator', 'avaluator')
# 方法一：直接返回
# builder.add_conditional_edges(
#     'avaluator',
#     route_func
# )

# 方法二：映射返回（更加直观）
builder.add_conditional_edges(
    'avaluator',route_func,
    {
        'Accepted':END,
        'Rejected+Feedback':'generator'
    }
)
graph=builder.compile()


