import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from .agent3_tools import search_arxiv_papers, download_arxiv_paper, search_pdf_tool
import json

# 导入web_print模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.web_print import print_my_content

load_dotenv()


class PaperRecommendationAgent:
    def __init__(self):
        self.llm = ChatTongyi(
            model=os.getenv("AGENT3_MODEL"),
            api_key=os.getenv("TONGYI_API_KEY"),
            model_kwargs={"enable_thinking": True}
        )
        self.tools = [
            search_arxiv_papers,
            download_arxiv_paper,
            search_pdf_tool
        ]
        self.prompt = """
            You are a helpful research assistant specialized in helping users discover, retrieve, and understand academic papers from arXiv.
            You have access to the following tools:

            {tools}

            Your capabilities:
            1. Search arXiv papers based on user's research interests
            2. Download papers that users want to read
            3. Search and analyze content within downloaded PDF papers
            4. Provide summaries and answer questions about papers

            Important guidelines:
            - Always prioritize relevance, recency, and clarity when recommending papers.
            - When recommending papers, include: title, authors, publication date, key contributions, and arXiv ID.
            - When analyzing a paper, first ensure it is downloaded; then use the PDF search tool to locate relevant sections.
            - Base your answers on the content of the papers.

            【Important Formatting Guidelines】
            - **Minimize complex math notation.** Describe equations in plain English when possible. Use `$equation$` for inline math and `$$equation$$` for block-level math only when necessary.
            """

        # 创建 agent
        self.agent = create_agent(
            self.llm,
            self.tools,
            system_prompt=self.prompt,
            state_schema=AgentState,
            checkpointer=InMemorySaver(),
        )


def init_agent3():
    """初始化并返回 agent3 实例"""
    return PaperRecommendationAgent()


async def invoke_agent3(agent_instance: PaperRecommendationAgent, query: str, thread_id: str = "default_session"):
    """使用持久化的实例进行多轮对话（支持会话隔离）"""
    print_my_content(f"开始处理用户查询: {query[:50]}...", thread_id, "info")
    print_my_content("初始化agent3（arXiv论文检索与问答）...", thread_id, "info")

    config = {"configurable": {"thread_id": thread_id}}
    response_text = ""
    pdf_path = None
    tool_calls_count = 0

    async for chunk in agent_instance.agent.astream({"messages": [HumanMessage(content=query)]}, config):
        match chunk:
            case {"model": _}:
                model_messages = chunk['model']['messages']
                if model_messages:
                    last_message = model_messages[0]
                    if hasattr(last_message, 'content'):
                        response_text = last_message.content
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        tool_calls_count += len(last_message.tool_calls)
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get('name', '未知工具')
                            print_my_content(f"调用工具: {tool_name}", thread_id, "info")
            case {"tools": _}:
                tool_messages = chunk['tools']['messages']
                if tool_messages:
                    content = tool_messages[0].content
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            if 'status' in data and data['status'] == 'success':
                                if 'file_path' in data:
                                    pdf_path = os.path.abspath(data['file_path'])
                                    print_my_content(f"论文下载成功: {os.path.basename(pdf_path)} (会话: {thread_id})", thread_id, "success")
                                elif 'count' in data:
                                    print_my_content(f"找到 {data['count']} 篇相关论文 (会话: {thread_id})", thread_id, "success")
                                elif 'results_count' in data:
                                    print_my_content(f"检索到 {data['results_count']} 个相关片段 (会话: {thread_id})", thread_id, "success")
                    except:
                        pass

    if response_text:
        print_my_content("回答生成完成", thread_id, "success")
        print_my_content(f"回答长度: {len(response_text)} 字符", thread_id, "info")

    if tool_calls_count > 0:
        print_my_content(f"总共调用 {tool_calls_count} 次工具", thread_id, "info")

    return response_text, pdf_path
