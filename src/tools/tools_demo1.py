from langchain_core.tools import tool


@tool
def calculate(a:float,b:float,operation:str)->float:
    """工具函数：计算两个数字的运算结果"""

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

print(calculate.name)
print(calculate.args)
print(calculate.description)
# 如果使用回调处理程序则为必须。它可用于为预期参数提供更多信息或验证
print(calculate.args_schema.model_json_schema())





