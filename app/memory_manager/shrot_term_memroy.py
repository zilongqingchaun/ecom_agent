#从最近的几条对话里面抽取关键信息。返回list[str],主要是偏好，关注商品，身份信息和情绪等具体参考prompt

from app.settings.settings import settings
from openai import OpenAI
from app.prompt.memory import STM_EXTRACTION_PROMPT
class ShortTermMemory:
    def __init__(self):
        #提取的关键事实
        self.short_term_facts=[]
        self.model=settings.model_name
        self.client=OpenAI(api_key=settings.model_api_key, base_url=settings.base_url)
    def load_facts(self,recent_raw_history:list[dict]):
        #这里传入的就是最近几条对话
        #从最近的几条对话里面抽取关键信息。返回list[str],主要是偏好，关注商品，身份信息和情绪等具体参考prompt
        for recent_message in recent_raw_history:
            if recent_message["role"]=="user":
                #调用模型进行抽取
                extracted_facts = self.extract_facts(recent_message["content"])
                facts = [fact.strip() for fact in extracted_facts.splitlines() if fact.strip()]
                self.short_term_facts.extend(fact for fact in facts if fact != "无新信息")
    def extract_facts(self,message:str)->str:
        # 这里可以调用模型来提取关键信息, 添加提示词
        response = self.client.chat.completions.create(model=self.model,
                                                        messages=[{"role": "system", "content": STM_EXTRACTION_PROMPT.format(existing_facts="\n".join(self.short_term_facts))},
                                                                  {"role": "user", "content": message}],    
                                                    temperature=settings.memory_temperature,
                                                    )
        content = response.choices[0].message.content or ""

        return content
