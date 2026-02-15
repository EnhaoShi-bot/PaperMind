"""
Web打印模块 - 用于在前端显示中间过程
"""
import json
from typing import Dict, List, Optional
from datetime import datetime

# 全局存储中间过程消息
_process_messages: Dict[str, List[Dict]] = {}
_max_messages_per_session = 100


def print_my_content(content: str, session_id: str = "default_session", level: str = "info"):
    """
    打印内容到前端显示
    
    Args:
        content: 要显示的内容
        session_id: 会话ID，用于区分不同用户的会话
        level: 消息级别 (info, warning, error, success)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = {
        "timestamp": timestamp,
        "content": content,
        "level": level
    }
    
    # 初始化会话消息列表
    if session_id not in _process_messages:
        _process_messages[session_id] = []
    
    # 添加消息
    _process_messages[session_id].append(message)
    
    # 限制消息数量
    if len(_process_messages[session_id]) > _max_messages_per_session:
        _process_messages[session_id] = _process_messages[session_id][- _max_messages_per_session:]
    
    # 打印到控制台（用于调试）
    level_colors = {
        "info": "\033[94m",      # 蓝色
        "warning": "\033[93m",   # 黄色
        "error": "\033[91m",     # 红色
        "success": "\033[92m"    # 绿色
    }
    reset_color = "\033[0m"
    
    color = level_colors.get(level, "\033[94m")
    print(f"{color}[{timestamp}] [{level.upper()}] {content}{reset_color}")


def get_process_messages(session_id: str = "default_session", clear: bool = False) -> List[Dict]:
    """
    获取指定会话的中间过程消息
    
    Args:
        session_id: 会话ID
        clear: 是否在获取后清空消息
        
    Returns:
        消息列表
    """
    messages = _process_messages.get(session_id, [])
    
    if clear:
        _process_messages[session_id] = []
    
    return messages


def clear_process_messages(session_id: str = "default_session"):
    """
    清空指定会话的中间过程消息
    
    Args:
        session_id: 会话ID
    """
    if session_id in _process_messages:
        _process_messages[session_id] = []