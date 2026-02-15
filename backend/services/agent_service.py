import os
import sys
import asyncio

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.get_responses import get_agent_response, reset_agent

class AgentService:
    def __init__(self):
        self.agent_cache = {
            "agent1": None,
            "agent2": None,
            "agent3": None
        }
    
    async def get_response(self, user_input: str, mode: str = "existing", 
                     pdf_path: str = None, thread_id: str = "default_session",
                     api_key: str = None):
        """
        统一获取Agent回复的接口
        
        Args:
            user_input: 用户输入
            mode: 问答模式 (existing/upload/research)
            pdf_path: PDF文件路径
            thread_id: 会话ID
            api_key: 用户API-KEY，如果为None则使用.env中的
            
        Returns:
            tuple: (response, new_pdf_path)
        """
        return await get_agent_response(
            user_input=user_input,
            mode=mode,
            pdf_path=pdf_path,
            thread_id=thread_id,
            api_key=api_key
        )
    
    def reset_agent(self, mode: str = None):
        """
        重置agent实例
        
        Args:
            mode: 要重置的agent模式，如果为None则重置所有agent
        """
        reset_agent(mode)
