#在这个框架下面实现agent调用，并返回格式化输出。



import sys

from app.settings.settings import settings


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if settings.model_mode == "multiple":
        from app.agent.ecom_multiple_agent import ecom_multiple_agent
        agent = ecom_multiple_agent(model=settings.model_name,temperature=settings.model_temperature)
        pass
    else:
        from app.agent.ecom_single_agent import ecom_single_agent
        agent = ecom_single_agent(model=settings.model_name,temperature=settings.model_temperature)
        pass
    try:
        while True:
            input_text = input("请输入您的问题：").strip()

            if not input_text:
                print("输入不能为空，请重新输入。")
                continue

            if input_text.lower() in ["exit", "quit"]:
                print("退出对话。")
                break
            try:
                response=agent.chat(input_text)
                print("Agent的回答：", response)
            except Exception as e:
                print("调用Agent时发生错误：", str(e))
    except (KeyboardInterrupt, EOFError):
        print("\n退出对话。")
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                print("保存长期记忆失败：", str(exc))


if __name__ == "__main__":
    main()
