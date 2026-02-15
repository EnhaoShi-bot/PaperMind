import os
import asyncio
import re
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from agents.agent1_existing.agent1 import init_agent1, invoke_agent1
from agents.agent2_upload.agent2 import init_agent2, invoke_agent2
from agents.agent3_arxiv.agent3 import init_agent3, invoke_agent3

load_dotenv()
# 实例缓存，用于存储已初始化的 agent
# 使用 session_id 作为 key 实现会话隔离
_agent_cache = {}  # 格式: {"session_id_mode": agent_instance}

# 翻译器缓存
_translator_cache = {
    "zh_to_en": None,
    "en_to_zh": None
}


def get_translator(translator_type, api_key=None):
    """获取翻译器实例"""
    # 使用api_key作为缓存键的一部分
    cache_key = f"{translator_type}_{api_key if api_key else 'default'}"
    
    if cache_key not in _translator_cache:
        if translator_type == "zh_to_en":
            model_name = os.getenv("TRANS_ZH_TO_EN")
            _translator_cache[cache_key] = ChatTongyi(
                model=model_name,
                api_key=api_key if api_key else os.getenv("TONGYI_API_KEY"),
                model_kwargs={"enable_thinking": False}
            )
        elif translator_type == "en_to_zh":
            model_name = os.getenv("TARNS_EN_TO_ZH")
            _translator_cache[cache_key] = ChatTongyi(
                model=model_name,
                api_key=api_key if api_key else os.getenv("TONGYI_API_KEY"),
                model_kwargs={"enable_thinking": False}
            )
        else:
            raise ValueError(f"未知的翻译器类型: {translator_type}")
    
    return _translator_cache[cache_key]


def is_chinese(text):
    """判断文本是否包含中文字符"""
    # 简单的判断：如果包含中文字符，则认为是中文
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))


async def translate_zh_to_en(text, api_key=None):
    """
    中翻英翻译器
    如果输入是中文，翻译成英文；如果已经是英文，保持不变
    """
    if not text or not is_chinese(text):
        return text

    translator = get_translator("zh_to_en", api_key)
    system_prompt = "你是一名专业翻译助手，仅将用户提供的中文内容准确、简洁地翻译为英文。请严格保留专业术语的准确性，不添加解释、评论或额外内容，也不回答问题，仅输出翻译后的英文文本。"
    human_prompt = f"请将以下中文内容翻译成英文，并仅输出翻译后的文本：{text}"

    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = await translator.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"中翻英翻译错误: {e}")
        return text


async def translate_en_to_zh(text, api_key=None):
    """
    英翻中翻译器
    将英文翻译成中文，同时保持markdown语法结构
    """
    if not text:
        return text

    translator = get_translator("en_to_zh", api_key)
    system_prompt = """
        你是一位专业的翻译助手，请将用户提供的英文内容准确翻译为中文。  
        在翻译过程中，请严格遵守以下规则：
        
        1. **保留所有 Markdown 语法结构不变**，包括但不限于：
           - 标题（如 `# 标题`）
           - 粗体（`**粗体**`）
           - 斜体（`*斜体*`）
           - 行内代码（`` `code` ``）和代码块
           - 链接（`[链接文本](url)`）
           - 表格、列表等格式
        
        2. **仅翻译 Markdown 标记内的文本内容**，不得修改、删除或添加任何 Markdown 语法符号。
        
        3. **确保专业术语翻译准确**，符合行业通用译法；若术语有多种译法，请优先采用最常见或上下文最贴切的版本。
        
        请输出符合上述要求的完整翻译结果。
        """
    human_prompt = f"请将以下英文内容翻译成中文，并仅输出翻译后的文本，并保持Markdown语法结构不变：{text}"

    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = await translator.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"英翻中翻译错误: {e}")
        return text


def get_agent_instance(mode, thread_id="default_session"):
    """获取 agent 实例（支持会话隔离）"""
    # 创建会话特定的缓存键
    cache_key = f"{thread_id}_{mode}"
    
    # 定义 mode 到初始化函数的映射
    agent_map = {
        'existing': init_agent1,
        'upload': init_agent2,
        'research': init_agent3
    }

    init_func = agent_map.get(mode)
    if not init_func:
        return None
    
    # 如果缓存中没有该会话的agent，则初始化
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = init_func()
    
    return _agent_cache.get(cache_key)


def reset_agent(mode=None, thread_id="default_session"):
    """
    重置 agent 实例缓存
    
    Args:
        mode: 要重置的 agent 模式，如果为 None 则重置该会话的所有 agent
        thread_id: 会话ID
    """
    if mode is None:
        # 重置该会话的所有 agent
        keys_to_remove = [key for key in _agent_cache.keys() if key.startswith(f"{thread_id}_")]
        for key in keys_to_remove:
            del _agent_cache[key]
    else:
        # 重置特定模式的 agent
        cache_key = f"{thread_id}_{mode}"
        if cache_key in _agent_cache:
            del _agent_cache[cache_key]


async def get_agent_response(user_input, mode="existing", pdf_path=None, thread_id="default_session", api_key=None):
    """
    统一获取 Agent 回复的接口（支持多轮对话）
    """
    instance = get_agent_instance(mode, thread_id)

    try:
        # 如果提供了api_key，临时设置环境变量
        original_api_key = None
        if api_key:
            original_api_key = os.environ.get("TONGYI_API_KEY")
            os.environ["TONGYI_API_KEY"] = api_key
            # 重置翻译器缓存，以便使用新的API-KEY
            global _translator_cache
            # 清除所有缓存，因为api_key改变了
            _translator_cache = {}
        
        # 步骤1: 对用户输入进行中翻英处理
        translated_input = await translate_zh_to_en(user_input, api_key)

        if mode == "existing":
            response, new_pdf_path = await invoke_agent1(instance, translated_input, thread_id)

        elif mode == "upload":
            # agent2 目前是同步调用逻辑，直接运行
            response, new_pdf_path = invoke_agent2(instance, translated_input, pdf_path, thread_id)

        elif mode == "research":
            response, new_pdf_path = await invoke_agent3(instance, translated_input, thread_id)

        else:
            return f"未知模式：{mode}", pdf_path

        # 步骤2: 对agent响应进行英翻中处理
        translated_response = await translate_en_to_zh(response, api_key)

        # 路径有效性检查
        final_pdf_path = new_pdf_path if new_pdf_path and os.path.exists(new_pdf_path) else pdf_path
        
        # 恢复原始环境变量
        if api_key and original_api_key is not None:
            os.environ["TONGYI_API_KEY"] = original_api_key
        elif api_key:
            del os.environ["TONGYI_API_KEY"]
            
        return translated_response, final_pdf_path

    except Exception as e:
        import traceback
        # 确保恢复环境变量
        if api_key and original_api_key is not None:
            os.environ["TONGYI_API_KEY"] = original_api_key
        elif api_key:
            del os.environ["TONGYI_API_KEY"]
        return f"Agent 运行错误: {str(e)}\n{traceback.format_exc()}", pdf_path


if __name__ == "__main__":
    async def main():
        print("\n=================== 测试 existing 模式 ===================")
        response, final_pdf_path = await get_agent_response("transformer的多头注意力机制是怎么实现的？",
                                                            mode="existing")
        print(response)
        print(final_pdf_path)

        print("\n=================== 测试 upload 模式 ===================")
        response, final_pdf_path = await get_agent_response("这篇论文主要讲了什么", mode="upload",
                                                            pdf_path="/home/seh/app/PaperMind/assets/awesome_papers/lora.pdf")
        print(response)
        print(final_pdf_path)

        print("\n=================== 测试 research 模式 ===================")
        response, final_pdf_path = await get_agent_response("搜索最新的强化学习论文，并下载其中一篇进行解读", mode="research")
        print(response)
        print(final_pdf_path)


    asyncio.run(main())
