from typing import TypedDict, Literal

from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import task, entrypoint
from pydantic import BaseModel, Field

from deep_agent.my_llm import model

"""
        新版本工作流，增加短期记忆
"""


# ========== 结构化输出模型（与之前相同） ==========
class Feedback(BaseModel):
    """评估笑话的反馈"""
    grade: Literal["funny", "not funny"] = Field(
        description="判断笑话是否幽默",
        examples=["funny", "not funny"]
    )
    feedback: str = Field(
        description="若不幽默，提供改进建议",
        example="可以加入双关语或意外结局"
    )


# ========== 任务1：生成笑话 ==========
@task
def generate_joke(topic: str, previous_feedback: str | None = None) -> str:
    """根据主题和上轮反馈生成一笑话"""
    if previous_feedback:
        prompt = f"根据反馈改进笑话：{previous_feedback}\n主题：{topic}"
    else:
        prompt = f"创作一个关于{topic}的笑话"

    chain = model | StrOutputParser()
    joke = chain.invoke(prompt)
    return joke


# ========== 任务2：评估笑话 ==========
@task
def evaluate_joke(joke: str) -> tuple[str, str]:
    """评估笑话，返回 (grade, feedback)"""
    prompt = (
        f"评估此笑话的幽默程度：\n{joke}\n"
        "注意：幽默应包含意外性或巧妙的措辞"
    )

    # 若模型支持 with_structured_output，优先使用
    try:
        structured_chain = model.with_structured_output(Feedback)
        result = structured_chain.invoke(prompt)
        return result.grade, result.feedback
    except Exception:
        # 备选：使用 bind_tools 手动解析（兼容不支持 structured_output 的模型）
        bound = model.bind_tools([Feedback])
        resp = bound.invoke(prompt)
        tool_calls = getattr(resp, "tool_calls", [])
        if tool_calls:
            args = tool_calls[-1]["args"]
            return args["grade"], args["feedback"]
        # 降级处理：默认认为是 not funny 并给出通用反馈
        return "not funny", "无法评估，请重新生成"


# ========== 入口工作流 ==========
# memory_checkpointer=MemorySaver()

@entrypoint()
def joke_workflow(topic: str) -> dict:
    """工作流入口：反复生成并评估，直到笑话被评为 funny"""
    feedback = None
    while True:
        # 生成笑话
        joke = generate_joke(topic, feedback).result()
        # 评估笑话
        grade, new_feedback = evaluate_joke(joke).result()

        if grade == "funny":
            return {
                "joke": joke,
                "topic": topic,
                "feedback": new_feedback,
                "funny_or_not": grade
            }
        else:
            # 不幽默，用反馈继续循环
            feedback = new_feedback
# ========== 关键：导出 graph 变量供 langgraph CLI 使用 ==========
graph = joke_workflow
