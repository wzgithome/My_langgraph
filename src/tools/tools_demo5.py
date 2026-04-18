from typing import Annotated

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field


'''
    通过注解的形式,推荐使用（谷歌风格）
'''



def calculate5(a:float,b:float,operation:str)->float:
    """
       计算两个数字的运算结果。

       Args:
           a: 第一个需要输入的数字
           b: 第二个需要输入的数字
           operation: 运算符号，只能是 add, sub, mul, div, pow（乘方）

       Returns:
           返回两个输入数字的运算结果。
       """

    print(f"调用calculate工具，第一个数字:{a},第二个数字：{b},运算符号：{operation}")

    result = 0.0

    match operation:
        case "add":
            result = a+b

        case "sub":
            result = a-b

        case "mul":
            result = a*b

        case "div":
            if b!=0:
                result=a/b
            else:
                raise ValueError('除数不能为零')

    return result


async def calculate6(a:float,b:float,operation:str)->float:
    """

       Args:
           a: 第一个需要输入的数字
           b: 第二个需要输入的数字
           operation: 运算符号，只能是 add, sub, mul, div, pow（乘方）

       Returns:
           返回两个输入数字的运算结果。
       """

    print(f"调用calculate工具，第一个数字:{a},第二个数字：{b},运算符号：{operation}")

    result = 0.0

    match operation:
        case "add":
            result = a+b

        case "sub":
            result = a-b

        case "mul":
            result = a*b

        case "div":
            if b!=0:
                result=a/b
            else:
                raise ValueError('除数不能为零')

    return result
calculate=StructuredTool.from_function(
    name="calculate",
    description="计算两个数字的运算结果",
    # 创建同步工具
    func=calculate5,
    args_schema=BaseModel,
    return_direct=False,
    # coroutine创建异步工具
    coroutine=calculate6,
    # args_schema=CalculateInput,
    # kwargs_schema=CalculateInput,
    # kwargs_
)





