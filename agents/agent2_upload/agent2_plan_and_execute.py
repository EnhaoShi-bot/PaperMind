import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

# 导入web_print模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.web_print import print_my_content

load_dotenv()


class PlanAgent:
    """Plan智能体：制定执行计划"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TONGYI_API_KEY')
        self.model = os.getenv('AGENT2_PLAN_MODEL')
        self.llm = ChatTongyi(
            model=self.model,
            api_key=self.api_key,
            model_kwargs={"enable_thinking": True}
        )

    def create_plan(self, user_query: str, context: str = "", thread_id: str = "default_session") -> str:
        system_prompt = """
        You are an intelligent planning assistant. Your task is to analyze the user's query and formulate a clear execution plan.
        Note that if a user uploads a PDF document and the document is mentioned in the user's question, or if the user's question may use this document, it should be retrieved to facilitate the answer to the user's question.
        Similarly, if a user's question may need to be answered based on previous conversation history, you should consider using contextual information.
        Please analyze the user's question and determine the steps that need to be performed:
        1. Is it necessary to process a new PDF document?
        2. Is it necessary to retrieve content from documents?
        3. Is it necessary to refer to conversation history or long-term memory?
        4. How should the response be structured?

        Please output the plan in a concise JSON format containing the following fields:
        {
          "need_document_processing": true/false,
          "need_retrieval": true/false,
          "need_context": true/false,
          "steps": ["Step 1", "Step 2", ...]
        }"""

        user_message = f"""
        User Query: {user_query}
        Context Information:
        {context if context else "No context provided."}
        
        Please create a clear execution plan in JSON format. The plan should include the following fields:
        "need_document_processing": true/false,
        "need_retrieval": true/false,
        "need_context": true/false,
        "steps": ["Step 1", "Step 2", ...]
        """

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            response = self.llm.invoke(messages)
            plan = response.content
            return plan
        except Exception as e:
            print_my_content(f"Plan生成失败: {e}", thread_id, "error")
            return json.dumps({
                "need_document_processing": False,
                "need_retrieval": True,
                "need_context": True,
                "steps": ["retrieve relevant documents", "answer based on retrieval results"]
            })


class ExecuteAgent:
    """Execute智能体：执行RAG流程并生成答案"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TONGYI_API_KEY')
        self.model = os.getenv('AGENT2_EXECUTE_MODEL')
        self.llm = ChatTongyi(
            model=self.model,
            api_key=self.api_key,
            model_kwargs={"enable_thinking": False}
        )

    def execute(self, user_query: str, plan: str, retrieved_docs: List[str] = None, context: str = "", thread_id: str = "default_session") -> str:
        system_prompt = """
        You are a professional document-based Q&A assistant. Your role is to answer users’ questions accurately and thoroughly by leveraging the content of their uploaded documents along with relevant contextual information.
        
        Guidelines:  
        1. Prioritize information from the provided document(s) when formulating your response. If the document content is incomplete or insufficient, you may supplement your answer with reliable contextual knowledge.  
        2. Ensure all responses are factually accurate, objective, and grounded in the available information.  
        3. Reference the conversation history as needed to maintain coherence and contextual consistency in your answers.
        """

        user_message_parts = [f"User Query: {user_query}\n"]
        if retrieved_docs:
            user_message_parts.append("user uploaded document content:")
            for i, doc in enumerate(retrieved_docs, 1):
                user_message_parts.append(f"\n[Document Fragment {i}]\n{doc}\n")
        if context:
            user_message_parts.append(f"\nContextual Information:\n{context}")
        user_message = "\n".join(user_message_parts)

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            response = self.llm.invoke(messages)
            answer = response.content
            return answer
        except Exception as e:
            return f"抱歉，生成答案时出现异常：{str(e)}"
