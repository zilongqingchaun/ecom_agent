#切分
#固定切分chunk的数据格式，chunk类内部包含chunk_id、docs、section、content四个属性
#分别代表id，文档名，章节名，内容
from dataclasses import dataclass
from app.settings.settings import settings

@dataclass
class Chunk:
    chunk_id: str
    docs:str
    section:str
    content:str

#根据文件划分chunk，暂时只实现按h2划分
#1、根据知识库路径，得到所有的md文本。
#2、对每个md文本，按h2划分为多个chunk，每个chunk包含文档名、章节名、内容。

class Make_Chunk:
    def __init__(self):
        self.rag_path=settings.knowledge_rag_path

    def chunk_by_h2(self)->list[Chunk]:
        md_files=self.md_text()
        chunks=[]
        chunk_id=0
        for md in md_files:
            section = None
            text=[]
            for line in md.get("text", "").splitlines():
                if not line.strip():
                    continue

                if line.startswith("## "):
                    # 新二级标题出现时，先保存上一节的正文。
                    if section is not None and text:
                        chunks.append(self.make_chunk(
                            str(chunk_id), md.get("docs", ""), section, "\n".join(text)
                        ))
                        chunk_id += 1
                    text=[]
                    section=line.strip()
                    continue

                # 忽略文件标题及第一个二级标题之前的内容。
                if section is not None:
                    text.append(line.strip())

            # 保存文件中的最后一节。
            if section is not None and text:
                chunks.append(self.make_chunk(
                    str(chunk_id), md.get("docs", ""), section, "\n".join(text)
                ))
                chunk_id += 1

        return chunks

    def make_chunk(self,chunk_id: str, docs: str, section: str, content: str)->Chunk:
        #暂不实现，返回list[chunk]，每个chunk是每个小节的信息。
        
        return Chunk(chunk_id=chunk_id, docs=docs, section=section, content=content)





    def md_text(self)->list[dict[str,str]]:
        #返回rag_path下所有的md文件内容
        md_files=[]
        for file in self.rag_path.glob("**/*.md"):
            docs=file.stem
            text=file.read_text(encoding="utf-8")
            md_files.append({"docs": docs, "text": text})

        return md_files



make_chunk=Make_Chunk()
