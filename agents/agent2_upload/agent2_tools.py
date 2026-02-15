"""
文档处理和检索工具模块
包含PDF文档加载、向量化、存储和混合检索功能
"""

import hashlib
import os
import re
import sqlite3
import sys
from typing import List, Tuple

import PyPDF2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sklearn.feature_extraction.text import TfidfVectorizer

# 导入web_print模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.web_print import print_my_content

load_dotenv()

# 创建 OpenAI 客户端（兼容DashScope API）
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
if TONGYI_API_KEY:
    openai_client = OpenAI(
        api_key=TONGYI_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
else:
    print("⚠️ 警告: 未找到 TONGYI_API_KEY 环境变量")
    openai_client = None


class DocumentProcessor:
    """文档处理工具：加载PDF、分块、向量化、存储"""

    def __init__(self,
                 chunk_size: int = 600,
                 chunk_overlap: int = 60,
                 milvus_db_path: str = os.getenv("AGENT2_DOC_MILVUS_PATH"),
                 sqlite_db_path: str = os.getenv("AGENT2_DOC_SQL_PATH")
                 ):
        """
        初始化文档处理器

        Args:
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 文本块重叠大小
            milvus_db_path: Milvus数据库路径
            sqlite_db_path: SQLite数据库路径
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.milvus_db_path = milvus_db_path
        self.sqlite_db_path = sqlite_db_path

    def load_pdf(self, pdf_path: str) -> str:
        """
        加载PDF文档并提取文本

        Args:
            pdf_path: PDF文件路径

        Returns:
            提取的文本内容
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        将文本按固定长度分块

        Args:
            text: 原始文本

        Returns:
            文本块列表
        """
        chunks = []
        text = re.sub(r'\s+', ' ', text).strip()

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def get_embedding(self, text: str) -> List[float]:
        """
        使用OpenAI兼容API的文本嵌入模型获取文本向量

        Args:
            text: 输入文本

        Returns:
            文本向量
        """
        if openai_client is None:
            raise Exception("OpenAI客户端未初始化，请检查TONGYI_API_KEY环境变量")
        
        response = openai_client.embeddings.create(
            model=os.getenv("AGENT2_EMBEDDING_MODEL", "text-embedding-v3"),
            input=text,
            dimensions=int(os.getenv("AGENT2_EMBEDDING_DIM", "1024")),
            encoding_format="float"
        )
        
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if openai_client is None:
            raise Exception("OpenAI客户端未初始化，请检查TONGYI_API_KEY环境变量")
        
        # 如果只有一个文本，直接调用单个API
        if len(texts) == 1:
            return [self.get_embedding(texts[0])]
        
        all_embeddings = []
        batch_size = 10  # 每批最多10个文本
        
        # 分批处理文本
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 批量API调用
            response = openai_client.embeddings.create(
                model=os.getenv("AGENT2_EMBEDDING_MODEL", "text-embedding-v3"),
                input=batch_texts,
                dimensions=int(os.getenv("AGENT2_EMBEDDING_DIM", "1024")),
                encoding_format="float"
            )
            
            # 收集本批次的向量
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # 打印进度信息
            if len(texts) > batch_size:
                progress = min(i + batch_size, len(texts))
                # print_my_content(f"向量化进度: {progress}/{len(texts)}", "default_session", "info")
        
        return all_embeddings

    def _get_collection_name(self, doc_name: str, thread_id: str = "default_session") -> str:
        """根据文档名称和会话ID生成collection名称"""
        # 使用hash确保名称合法
        combined = f"{doc_name}_{thread_id}"
        hash_name = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"doc_{hash_name}"

    def store_to_milvus(self, chunks: List[str], doc_name: str, thread_id: str = "default_session"):
        """
        将文本块向量化后存储到Milvus（支持会话隔离）

        Args:
            chunks: 文本块列表
            doc_name: 文档名称
            thread_id: 会话ID
        """
        collection_name = self._get_collection_name(doc_name, thread_id)

        # 初始化Milvus客户端
        client = MilvusClient(self.milvus_db_path)

        # 如果collection已存在，先删除
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)

        # 创建collection
        embedding_dim = int(os.getenv("AGENT2_EMBEDDING_DIM", "1024"))
        client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,  # 从环境变量获取向量维度
            metric_type="COSINE"
        )

        # 批量获取向量
        print_my_content(f"批量向量化 {len(chunks)} 个文本块...", thread_id, "info")
        embeddings = self.get_embeddings_batch(chunks)

        # 准备数据
        data = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append({
                "id": idx,
                "vector": embedding,
                "text": chunk,
                "thread_id": thread_id
            })

        # 插入数据
        client.insert(collection_name=collection_name, data=data)
        print_my_content(f"成功将 {len(chunks)} 个文本块存储到Milvus (会话: {thread_id})", thread_id, "success")

    def store_to_sqlite(self, chunks: List[str], doc_name: str, thread_id: str = "default_session"):
        """
        将文本块存储到SQLite（支持会话隔离）

        Args:
            chunks: 文本块列表
            doc_name: 文档名称
            thread_id: 会话ID
        """
        table_name = self._get_collection_name(doc_name, thread_id)

        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        # 创建表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                thread_id TEXT NOT NULL
            )
        """)

        # 清空旧数据
        cursor.execute(f"DELETE FROM {table_name} WHERE thread_id = ?", (thread_id,))

        # 插入数据
        for idx, chunk in enumerate(chunks):
            cursor.execute(f"INSERT INTO {table_name} (id, text, thread_id) VALUES (?, ?, ?)",
                           (idx, chunk, thread_id))

        conn.commit()
        conn.close()
        print_my_content(f"成功将 {len(chunks)} 个文本块存储到SQLite (会话: {thread_id})", thread_id, "success")

    def process_document(self, pdf_path: str, thread_id: str = "default_session") -> str:
        """
        处理PDF文档：加载、分块、向量化、存储（支持会话隔离）

        Args:
            pdf_path: PDF文件路径
            thread_id: 会话ID

        Returns:
            处理状态信息
        """
        # 提取文档名称
        doc_name = os.path.basename(pdf_path).replace('.pdf', '')

        # 加载PDF
        print_my_content(f"正在加载PDF: {os.path.basename(pdf_path)}", thread_id, "info")
        text = self.load_pdf(pdf_path)

        # 分块
        print_my_content("正在分块文本...", thread_id, "info")
        chunks = self.chunk_text(text)

        # 存储到Milvus
        print_my_content("正在向量化并存储到Milvus...", thread_id, "info")
        self.store_to_milvus(chunks, doc_name, thread_id)

        # 存储到SQLite
        print_my_content("正在存储到SQLite...", thread_id, "info")
        self.store_to_sqlite(chunks, doc_name, thread_id)

        print_my_content(f"文档处理完成！共处理 {len(chunks)} 个文本块 (会话: {thread_id})", thread_id, "success")
        return f"文档处理完成！共处理 {len(chunks)} 个文本块"


class DocumentRetriever:
    """文档检索工具：混合检索（向量+关键词）+ RRF融合"""

    def __init__(self,
                 top_k: int = 10,
                 milvus_db_path: str = os.getenv("AGENT2_DOC_MILVUS_PATH"),
                 sqlite_db_path: str = os.getenv("AGENT2_DOC_SQL_PATH")):
        """
        初始化检索器

        Args:
            top_k: 返回Top K个结果
            milvus_db_path: Milvus数据库路径
            sqlite_db_path: SQLite数据库路径
        """
        self.top_k = top_k
        self.milvus_db_path = milvus_db_path
        self.sqlite_db_path = sqlite_db_path

    def get_embedding(self, text: str) -> List[float]:
        """获取文本向量"""
        if openai_client is None:
            raise Exception("OpenAI客户端未初始化，请检查TONGYI_API_KEY环境变量")
        
        response = openai_client.embeddings.create(
            model=os.getenv("AGENT2_EMBEDDING_MODEL", "text-embedding-v3"),
            input=text,
            dimensions=int(os.getenv("AGENT2_EMBEDDING_DIM", "1024")),
            encoding_format="float"
        )
        
        return response.data[0].embedding

    def _get_collection_name(self, doc_name: str, thread_id: str = "default_session") -> str:
        """根据文档名称和会话ID生成collection名称"""
        combined = f"{doc_name}_{thread_id}"
        hash_name = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"doc_{hash_name}"

    def vector_search(self, query: str, doc_name: str, thread_id: str = "default_session") -> List[Tuple[int, str, float]]:
        """
        向量检索（支持会话隔离）

        Args:
            query: 查询文本
            doc_name: 文档名称
            thread_id: 会话ID

        Returns:
            [(chunk_id, chunk_text, score), ...]
        """
        collection_name = self._get_collection_name(doc_name, thread_id)

        # 获取查询向量
        query_vector = self.get_embedding(query)

        # 向量检索
        client = MilvusClient(self.milvus_db_path)

        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=self.top_k,
            output_fields=["id", "text", "thread_id"]
        )

        # 格式化结果
        retrieved = []
        for hit in results[0]:
            retrieved.append((
                hit['id'],
                hit['entity']['text'],
                hit['distance']
            ))

        return retrieved

    def keyword_search(self, query: str, doc_name: str, thread_id: str = "default_session") -> List[Tuple[int, str, float]]:
        """
        关键词检索（基于TF-IDF，支持会话隔离）

        Args:
            query: 查询文本
            doc_name: 文档名称
            thread_id: 会话ID

        Returns:
            [(chunk_id, chunk_text, score), ...]
        """
        table_name = self._get_collection_name(doc_name, thread_id)

        # 从SQLite获取所有文本块
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, text FROM {table_name} WHERE thread_id = ?", (thread_id,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        chunk_ids = [row[0] for row in rows]
        texts = [row[1] for row in rows]

        # 计算TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts + [query])

        # 计算相似度
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]

        similarities = (doc_vecs * query_vec.T).toarray().flatten()

        # 获取Top K
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        retrieved = []
        for idx in top_indices:
            retrieved.append((
                chunk_ids[idx],
                texts[idx],
                float(similarities[idx])
            ))

        return retrieved

    def rrf_fusion(self,
                   vector_results: List[Tuple[int, str, float]],
                   keyword_results: List[Tuple[int, str, float]],
                   k: int = 100) -> List[str]:
        """
        RRF（Reciprocal Rank Fusion）融合算法

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            k: RRF参数

        Returns:
            融合后的文本块列表
        """
        # 计算RRF分数
        rrf_scores = {}

        # 向量检索结果
        for rank, (chunk_id, text, score) in enumerate(vector_results, 1):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {'text': text, 'score': 0}
            rrf_scores[chunk_id]['score'] += 1 / (k + rank)

        # 关键词检索结果
        for rank, (chunk_id, text, score) in enumerate(keyword_results, 1):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {'text': text, 'score': 0}
            rrf_scores[chunk_id]['score'] += 1 / (k + rank)

        # 按分数排序
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # 返回Top K文本
        return [item[1]['text'] for item in sorted_results[:self.top_k]]

    def retrieve(self, query: str, doc_name: str, thread_id: str = "default_session") -> List[str]:
        """
        混合检索：向量检索 + 关键词检索 + RRF融合（支持会话隔离）

        Args:
            query: 用户查询
            doc_name: 文档名称
            thread_id: 会话ID

        Returns:
            检索到的相关文本块列表
        """
        # 向量检索
        print_my_content("执行向量检索...", thread_id, "info")
        vector_results = self.vector_search(query, doc_name, thread_id)

        # 关键词检索
        print_my_content("执行关键词检索...", thread_id, "info")
        keyword_results = self.keyword_search(query, doc_name, thread_id)

        # RRF融合
        print_my_content("融合检索结果...", thread_id, "info")
        final_results = self.rrf_fusion(vector_results, keyword_results)

        if final_results:
            print_my_content(f"检索完成，找到 {len(final_results)} 个相关片段 (会话: {thread_id})", thread_id, "success")
        else:
            print_my_content("未找到相关片段", thread_id, "warning")

        return final_results


# 工具函数，供Agent调用
def process_pdf_document(pdf_path: str, thread_id: str = "default_session") -> str:
    """
    处理PDF文档的工具函数（支持会话隔离）

    Args:
        pdf_path: PDF文件路径
        thread_id: 会话ID

    Returns:
        处理结果信息
    """
    processor = DocumentProcessor()
    return processor.process_document(pdf_path, thread_id)


def retrieve_documents(query: str, doc_name: str, thread_id: str = "default_session", top_k: int = 10) -> List[str]:
    """
    检索相关文档的工具函数（支持会话隔离）

    Args:
        query: 用户查询
        doc_name: 文档名称
        thread_id: 会话ID
        top_k: 返回结果数量

    Returns:
        相关文本块列表
    """
    retriever = DocumentRetriever(top_k=top_k)
    return retriever.retrieve(query, doc_name, thread_id)
