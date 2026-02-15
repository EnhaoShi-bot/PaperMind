"""
长期记忆模块：持久化存储重要对话记录，支持跨会话记忆检索
短期记忆模块：维护当前会话的对话历史，支持多轮对话上下文
"""
import os
import sqlite3
import sys
from datetime import datetime
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv
from pymilvus import MilvusClient

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

class ShortMemory:
    """短期记忆：会话级别的对话历史管理"""

    def __init__(self, max_turns: int = 10):
        """
        初始化短期记忆

        Args:
            max_turns: 最大保存的对话轮次
        """
        self.max_turns = max_turns
        self.memory: List[Dict] = []

    def add_conversation(self, user_query: str, assistant_response: str,
                         metadata: Optional[Dict] = None):
        """
        添加一轮对话到记忆

        Args:
            user_query: 用户查询
            assistant_response: 助手回答
            metadata: 额外的元数据（可选）
        """
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user': user_query,
            'assistant': assistant_response,
            'metadata': metadata or {}
        }

        self.memory.append(conversation)

        # 超过最大轮次，移除最早的记录
        if len(self.memory) > self.max_turns:
            self.memory.pop(0)

    def get_recent_conversations(self, n: int = 5) -> List[Dict]:
        """
        获取最近N轮对话

        Args:
            n: 获取的对话轮次

        Returns:
            最近N轮对话列表
        """
        return self.memory[-n:] if len(self.memory) >= n else self.memory

    def get_context_for_llm(self, n: int = 5) -> str:
        """
        获取格式化的对话上下文，用于提供给LLM

        Args:
            n: 获取最近N轮对话

        Returns:
            格式化的对话上下文字符串
        """
        recent = self.get_recent_conversations(n)

        if not recent:
            return ""

        context_parts = ["以下是最近的对话历史：\n"]

        for conv in recent:
            context_parts.append(f"用户: {conv['user']}")
            context_parts.append(f"助手: {conv['assistant']}\n")

        return "\n".join(context_parts)


class LongMemory:
    """长期记忆：持久化存储重要对话（支持会话隔离）"""

    def __init__(self,
                 sqlite_db_path: str = os.getenv("AGENT2_LONG_MEMORY_SQL_PATH"),
                 milvus_db_path: str = os.getenv("AGENT2_LONG_MEMORY_MILVUS_PATH")):
        """
        初始化长期记忆

        Args:
            sqlite_db_path: SQLite数据库路径
            milvus_db_path: Milvus数据库路径
        """
        self.sqlite_db_path = sqlite_db_path
        self.milvus_db_path = milvus_db_path
        self.base_collection_name = "long_memory"

    def _get_collection_name(self, thread_id: str = "default_session") -> str:
        """根据会话ID生成collection名称"""
        # 使用hash确保名称合法
        import hashlib
        hash_id = hashlib.md5(thread_id.encode()).hexdigest()[:8]
        return f"{self.base_collection_name}_{hash_id}"

    def _init_sqlite(self, thread_id: str = "default_session"):
        """初始化SQLite数据库（会话特定）"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        table_name = self._get_collection_name(thread_id)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                keywords TEXT,
                is_important BOOLEAN DEFAULT 1,
                metadata TEXT,
                thread_id TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _init_milvus(self, thread_id: str = "default_session"):
        """初始化Milvus数据库（会话特定）"""
        client = MilvusClient(self.milvus_db_path)
        collection_name = self._get_collection_name(thread_id)

        # 如果collection不存在，创建它
        if not client.has_collection(collection_name):
            embedding_dim = int(os.getenv("AGENT2_EMBEDDING_DIM", "1024"))
            client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dim,  # 从环境变量获取向量维度
                metric_type="COSINE"
            )

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

    def add_memory(self,
                   user_query: str,
                   assistant_response: str,
                   thread_id: str = "default_session",
                   keywords: Optional[str] = None,
                   is_important: bool = True,
                   metadata: Optional[str] = None):
        """
        添加记忆到长期存储（支持会话隔离）

        Args:
            user_query: 用户查询
            assistant_response: 助手回答
            thread_id: 会话ID
            keywords: 关键词（可选）
            is_important: 是否标记为重要
            metadata: 额外的元数据（可选）
        """
        # 初始化数据库（如果不存在）
        self._init_sqlite(thread_id)
        self._init_milvus(thread_id)
        
        # 合并用户查询和助手回答作为记忆内容
        content = f"user: {user_query}\n assistant: {assistant_response}"
        timestamp = datetime.now().isoformat()

        # 存储到SQLite
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        table_name = self._get_collection_name(thread_id)
        cursor.execute(f"""
            INSERT INTO {table_name} (content, timestamp, keywords, is_important, metadata, thread_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (content, timestamp, keywords, is_important, metadata, thread_id))

        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # 存储到Milvus（向量化）
        embedding = self.get_embedding(content)
        collection_name = self._get_collection_name(thread_id)

        client = MilvusClient(self.milvus_db_path)
        client.insert(
            collection_name=collection_name,
            data=[{
                "id": memory_id,
                "vector": embedding,
                "content": content,
                "thread_id": thread_id
            }]
        )

        print_my_content(f"成功添加长期记忆 (ID: {memory_id}, 会话: {thread_id})", thread_id, "success")
        return memory_id

    def search_by_keywords(self, keywords: str, thread_id: str = "default_session", limit: int = 5) -> List[Dict]:
        """
        基于关键词搜索记忆（支持会话隔离）

        Args:
            keywords: 关键词
            thread_id: 会话ID
            limit: 返回结果数量

        Returns:
            匹配的记忆列表
        """
        # 初始化数据库（如果不存在）
        self._init_sqlite(thread_id)
        
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        table_name = self._get_collection_name(thread_id)
        # 使用LIKE进行模糊匹配
        cursor.execute(f"""
            SELECT id, content, timestamp, keywords, is_important
            FROM {table_name}
            WHERE (keywords LIKE ? OR content LIKE ?) AND thread_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{keywords}%", f"%{keywords}%", thread_id, limit))

        rows = cursor.fetchall()
        conn.close()

        memories = []
        for row in rows:
            memories.append({
                'id': row[0],
                'content': row[1],
                'timestamp': row[2],
                'keywords': row[3],
                'is_important': bool(row[4])
            })

        return memories

    def search_by_similarity(self, query: str, thread_id: str = "default_session", limit: int = 5) -> List[Dict]:
        """
        基于向量相似度搜索记忆（支持会话隔离）

        Args:
            query: 查询文本
            thread_id: 会话ID
            limit: 返回结果数量

        Returns:
            相似的记忆列表
        """
        # 初始化数据库（如果不存在）
        self._init_milvus(thread_id)
        
        # 获取查询向量
        query_vector = self.get_embedding(query)
        collection_name = self._get_collection_name(thread_id)

        # 向量检索
        client = MilvusClient(self.milvus_db_path)

        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["id", "content", "thread_id"]
        )

        if not results or not results[0]:
            return []

        # 从SQLite获取完整信息
        memory_ids = [hit['id'] for hit in results[0]]

        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        table_name = self._get_collection_name(thread_id)
        placeholders = ','.join(['?'] * len(memory_ids))
        cursor.execute(f"""
            SELECT id, content, timestamp, keywords, is_important
            FROM {table_name}
            WHERE id IN ({placeholders}) AND thread_id = ?
        """, memory_ids + [thread_id])

        rows = cursor.fetchall()
        conn.close()

        # 构建结果
        memories = []
        for row in rows:
            memories.append({
                'id': row[0],
                'content': row[1],
                'timestamp': row[2],
                'keywords': row[3],
                'is_important': bool(row[4])
            })

        return memories

    def get_context_for_query(self, query: str, thread_id: str = "default_session", limit: int = 3) -> str:
        """
        获取与查询相关的长期记忆上下文（支持会话隔离）

        Args:
            query: 用户查询
            thread_id: 会话ID
            limit: 返回结果数量

        Returns:
            格式化的记忆上下文
        """
        memories = self.search_by_similarity(query, thread_id, limit)

        if not memories:
            return ""

        context_parts = ["the following are relevant historical memories：\n"]

        for mem in memories:
            context_parts.append(f"[{mem['timestamp']}]")
            context_parts.append(mem['content'])
            context_parts.append("")

        return "\n".join(context_parts)
