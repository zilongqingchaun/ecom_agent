#在这个框架下面实现agent调用，并返回格式化输出。



from app.settings.settings import settings
def main():
    #从settings导入配置
    if settings.model_mode == "multiple":
        from app.agent.ecom_multiple_agent import ecom_multiple_agent
        agent = ecom_multiple_agent(model=settings.model_name,temperature=settings.model_temperature)
        #在这里调用agent的相关方法
        pass
    else:
        #如果是多agent模式，则导入ecom_multiple_agent，否则导入ecom_single_agent
        from app.agent.ecom_single_agent import ecom_single_agent
        agent = ecom_single_agent(model=settings.model_name,temperature=settings.model_temperature)
        #在这里调用agent的相关方法
        pass
    #返回的应该是一个openai对象，然后这里导入输入
    #写一个循环，开始对话,目前支持退出和继续对话
    while True:
        input_text = input("请输入您的问题：").strip()

        if not input_text:
            print("输入不能为空，请重新输入。")
            continue

        if input_text.lower() in ["exit", "quit"]:
            print("退出对话。")
            break
        #假设经过上述两步得到了输入context
        try:
            response=agent.chat(input_text)
            print("Agent的回答：", response)
        except Exception as e:
            print("调用Agent时发生错误：", str(e))





if __name__ == "__main__":
    main()