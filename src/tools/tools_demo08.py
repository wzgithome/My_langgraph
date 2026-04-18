from langchain_core.runnables import RunnableConfig


def get_user_info_by_name(config:RunnableConfig) -> dict[str, int | str]:
    """获取用户的所有信息，包括：性别，年龄等"""
    user_name=config.get("configurable").get("user_name",'zs')
    print(f"获取用户{user_name}的所有信息")
    return {"userName":user_name,"age":18,"sex":"male"}


