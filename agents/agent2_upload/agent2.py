"""智能体主逻辑模块实现plan-and-execute策略的RAG智能体"""
import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from .agent2_memory import LongMemory, ShortMemory
from .agent2_plan_and_execute import PlanAgent, ExecuteAgent
from .agent2_tools import process_pdf_document, retrieve_documents

# 导入web_print模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.web_print import print_my_content

load_dotenv()


class RAGAgent:
    """整合后的 RAG 智能体"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TONGYI_API_KEY')
        self.plan_agent = PlanAgent(self.api_key)
        self.execute_agent = ExecuteAgent(self.api_key)
        self.short_memory = ShortMemory(max_turns=10)
        self.long_memory = LongMemory()
        self.current_doc_name = None

    def ensure_document(self, pdf_path: str, thread_id: str = "default_session"):
        """确保文档已处理，如果路径变化则重新加载（支持会话隔离）"""
        doc_name = os.path.basename(pdf_path).replace('.pdf', '')
        if self.current_doc_name != doc_name:
            print_my_content(f"检测到新文档，正在处理: {os.path.basename(pdf_path)}", thread_id, "info")
            process_pdf_document(pdf_path, thread_id)
            self.current_doc_name = doc_name
        return self.current_doc_name


def init_agent2():
    """初始化并返回 agent2 实例 (RAGAgent)"""
    return RAGAgent()


def invoke_agent2(agent_instance: RAGAgent, user_query: str, pdf_path: str, thread_id: str = "default_session", save_to_long_memory: bool = False):
    """使用持久化的 RAGAgent 实例进行对话"""
    # 打印开始处理消息
    print_my_content(f"开始处理用户查询: {user_query[:50]}...", thread_id, "info")
    print_my_content(f"PDF文档: {os.path.basename(pdf_path)}", thread_id, "info")
    
    # 1. 确保文档已加载（使用会话ID作为分区）
    try:
        doc_name = agent_instance.ensure_document(pdf_path, thread_id)
        print_my_content(f"文档处理完成: {doc_name} (会话: {thread_id})", thread_id, "success")
    except Exception as e:
        print_my_content(f"文档处理失败: {str(e)}", thread_id, "error")
        return f"文档处理失败: {str(e)}", pdf_path

    # 2. 获取记忆上下文（使用会话ID作为分区）
    context = ""
    short_ctx = agent_instance.short_memory.get_context_for_llm(n=3)
    long_ctx = agent_instance.long_memory.get_context_for_query(user_query, thread_id, limit=2)
    if short_ctx: 
        context += f"【short_memory】\n{short_ctx}\n"
        print_my_content("获取短期记忆上下文", thread_id, "info")
    if long_ctx: 
        context += f"【long_memory】\n{long_ctx}\n"
        print_my_content("获取长期记忆上下文", thread_id, "info")

    # 3. 规划与检索（使用会话ID作为分区）
    print_my_content("正在制定执行计划...", thread_id, "info")
    plan_raw = agent_instance.plan_agent.create_plan(user_query, context)
    try:
        plan = json.loads(plan_raw)
        print_my_content("执行计划生成完成", thread_id, "success")
    except:
        plan = {"need_retrieval": True, "need_context": True}
        print_my_content("使用默认执行计划", thread_id, "warning")

    retrieved_docs = None
    if plan.get("need_retrieval", True):
        print_my_content("正在检索相关文档...", thread_id, "info")
        retrieved_docs = retrieve_documents(user_query, doc_name, thread_id, top_k=10)
        if retrieved_docs:
            print_my_content(f"检索到 {len(retrieved_docs)} 个相关文档片段", thread_id, "success")
        else:
            print_my_content("未检索到相关文档", thread_id, "warning")

    # 4. 生成回答
    print_my_content("正在生成回答...", thread_id, "info")
    answer = agent_instance.execute_agent.execute(
        user_query=user_query,
        plan=plan_raw,
        retrieved_docs=retrieved_docs,
        context=context if plan.get("need_context", True) else ""
    )
    print_my_content("回答生成完成", thread_id, "success")
    print_my_content(f"回答长度: {len(answer)} 字符", thread_id, "info")

    # 5. 更新记忆（使用会话ID作为分区）
    agent_instance.short_memory.add_conversation(user_query, answer)
    print_my_content("更新短期记忆", thread_id, "info")
    
    if save_to_long_memory or "[remember this]" in user_query:
        agent_instance.long_memory.add_memory(user_query, answer, thread_id, is_important=True)
        print_my_content("更新长期记忆", thread_id, "info")

    return answer, pdf_path
