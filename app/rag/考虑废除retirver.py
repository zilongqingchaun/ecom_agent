from app.rag.backends.chroma_db import ecom_chroma_backend
#感觉不是很需要这个retriver，有点多余了，老项目用是需要同时管理手写余弦和chroma数据库，我应该不需要
#定义retriver，管理后端(检查是否有数据，从而检测是否划分chunk和进行嵌入)，进行检索并返回检索结果
class retriver:
    def __init__(self):
        self.backend=None
        self.chroma_backend=ecom_chroma_backend
        
    def load_chroma(self,collection_name:str):
        #加载chroma数据库
        pass

    def search(self,query:str):
        #在chroma数据库中进行检索 并返回检索结果

        pass
    
        


