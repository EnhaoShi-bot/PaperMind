import json
import os
# 导入web_print模块
import sys
from pathlib import Path
from typing import List, Dict, Any, Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .agent1_retrieval_tool import Qwen25VLEmbedding, MultimodalRetriever, create_retrieval_tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.web_print import print_my_content

load_dotenv()


def get_project_root() -> Path:
    """
    获取项目根目录（包含 .env 文件的目录）
    返回 Path 对象
    """
    # 从当前文件向上查找 .env 文件
    current = Path(__file__).absolute()
    while current != current.parent:
        if (current / '.env').exists():
            return current
        current = current.parent
    # 如果没找到，返回当前工作目录
    return Path.cwd()


# 初始化LLM
llm = ChatTongyi(
    model=os.getenv("AGENT1_MODEL"),
    api_key=os.getenv("TONGYI_API_KEY"),
    model_kwargs={"enable_thinking": False}
)

# 创建检索器（保留用于其他用途）
retriever = MultimodalRetriever(
    embedding_model=Qwen25VLEmbedding(
        api_key=os.getenv("TONGYI_API_KEY"),
        model=os.getenv("AGENT1_EMBEDDING_MODEL")
    ),
    is_print=False
)

# ==================== 工具函数 ====================
# 使用工厂函数创建检索工具
retrieve_tool = create_retrieval_tool(
    api_key=os.getenv("TONGYI_API_KEY"),
    is_print=False
)

# 绑定工具到LLM
tools = [retrieve_tool]
llm_with_tools = llm.bind_tools(tools)


# ==================== 状态定义 ====================
class AgentState(TypedDict):
    """Agent的状态定义"""
    messages: Annotated[List, add_messages]
    retrieved_images: List[Dict[str, Any]]  # 存储检索到的图片信息


# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """
You are an expert AI research assistant specialized in interpreting academic literature. Your goal is to help users understand, analyze, and extract insights from AI-related papers using retrieved text excerpts and visual materials.

## Capabilities
- Retrieve relevant content from academic databases based on user queries
- Incorporate useful images into responses to aid comprehension (optional but recommended when relevant)
- Deliver professional, accurate, and comprehensive answers

## Response Guidelines
1. **Base answers on retrieved materials.** Prioritize evidence from search results.
2. **Reference images** using the exact format: `(images)[path]`  
   Example: `(images)[./images/transformer_architecture.jpg]`
3. **Explain technical concepts step-by-step** to ensure clarity.
4. **Maintain a professional yet accessible tone.** Make advanced material understandable without oversimplifying.
5. **Minimize complex math notation.** Describe equations in plain English when possible. Use `$equation$` for inline math and `$$equation$$` for block-level math only when necessary.

## Operational Constraints
- **Max 3 retrieval attempts.** If initial results are insufficient, refine keywords and retry (≤3 times total).
- **Transparency about speculation.** If unable to answer after max attempts, you may offer reasoned speculation but clearly state: (a) that it's speculative, and (b) how many retrieval rounds were attempted.
- **English keywords only** for all database searches, as the literature is exclusively in English.

Now respond to the user's query following the above guidelines.
"""

SYSTEM_PROMPT_ZH = """
你是一位专注于解读人工智能学术论文的专家助手。你的任务是帮助用户理解、分析并从AI相关学术文献中提取洞见，利用检索到的文本片段和视觉材料进行辅助。

## 核心能力
- 根据用户需求检索学术数据库，利用获取的文本与图像回答问题
- 适时插入有价值的图像辅助理解（非强制，无关图像可忽略）
- 以专业视角确保回答的准确性与完整性

## 回答规范
1. **基于检索材料作答**，优先使用搜索结果中的证据
2. **图像引用格式**：使用 `(images)[图片路径]`  
   示例：`(images)[./images/transformer_architecture.jpg]`
3. **技术内容分步解释**，确保逻辑清晰易懂
4. **专业且易懂的表达风格**，在不简化本质的前提下降低理解门槛
5. **避免复杂数学符号**，尽量用自然语言描述公式。必要时使用 `$equation$` 表示行内公式，`$$equation$$` 表示块级公式

## 执行约束
- **最多3次检索**。若结果不理想可优化关键词重试，但总次数不超过3轮
- **推测需声明**。若达到上限仍无法回答，可进行合理推测，但必须明确告知用户：（a）此为推测内容；（b）已尝试的检索轮次
- **检索关键词必须使用英文**，因为所有文献均为英文

现在，请遵循以上指南回答用户问题。
"""


# ==================== 节点函数 ====================
# agent节点
def call_model(state: AgentState) -> AgentState:
    """调用模型生成响应"""
    messages = state["messages"]

    # 打印处理状态
    if len(messages) > 0 and isinstance(messages[-1], HumanMessage):
        user_query = messages[-1].content
        if isinstance(user_query, str):
            print_my_content(f"正在处理用户查询: {user_query[:10]}...", "default_session", "info")
        else:
            print_my_content("正在处理用户查询...", "default_session", "info")

    # 如果是第一次调用，添加系统消息
    response = llm_with_tools.invoke(messages)

    # 检查是否有工具调用
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tool_call['name'] for tool_call in response.tool_calls]
        print_my_content(f"模型决定调用工具: {', '.join(tool_names)}", "default_session", "info")
    else:
        print_my_content("模型生成回答中...", "default_session", "info")

    return {
        "messages": [response],
        "retrieved_images": state.get("retrieved_images", [])
    }


# process_tool节点
def process_tool_output(state: AgentState) -> AgentState:
    """
    处理工具输出，提取图片信息
    这个节点在工具执行后运行，检查是否有图片需要添加到上下文
    """
    messages = state["messages"]
    retrieved_images = state.get("retrieved_images", [])
    # 查找最后一个ToolMessage
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            try:
                tool_result = json.loads(message.content)
                # 如果有图片路径，直接存储路径信息
                image_paths = tool_result.get("image_paths", [])

                # 打印工具执行结果
                if image_paths:
                    print_my_content(f"检索工具返回了 {len(image_paths)} 张相关图片", "default_session", "success")
                    for i, img_path in enumerate(image_paths[:3]):  # 只显示前3张
                        print_my_content(f"图片 {i + 1}: {os.path.basename(img_path)}", "default_session", "info")
                    if len(image_paths) > 3:
                        print_my_content(f"... 还有 {len(image_paths) - 3} 张图片", "default_session", "info")
                else:
                    print_my_content("检索工具未找到相关图片", "default_session", "warning")

                if image_paths:
                    project_root = get_project_root()

                    for img_path in image_paths:
                        original_path = img_path  # 保存原始路径用于调试

                        # 将路径转换为绝对路径
                        # 如果已经是绝对路径，直接使用
                        if os.path.isabs(img_path):
                            abs_path = Path(img_path)
                        else:
                            # 相对路径：假设相对于项目根目录
                            abs_path = (project_root / img_path).resolve()

                        # 如果文件不存在，尝试在 assets/paper_images 的子目录中查找文件名
                        if not abs_path.exists():
                            filename = abs_path.name
                            # 在项目根目录下的 assets/paper_images 中搜索
                            possible_dirs = [
                                project_root / "assets" / "paper_images" / "transformer",
                                project_root / "assets" / "paper_images" / "lora",
                                project_root / "assets" / "paper_images" / "dpo",
                            ]
                            found = False
                            for dir_path in possible_dirs:
                                candidate = dir_path / filename
                                if candidate.exists():
                                    abs_path = candidate
                                    found = True
                                    break
                            if not found:
                                # 跳过不存在的图片
                                continue

                        retrieved_images.append({
                            "path": str(abs_path),
                            "original_path": original_path  # 保存原始路径用于调试
                        })

                # 更新ToolMessage的内容，使其更友好
                text_content = tool_result.get("text", "")
                updated_content = f"""【Retrieval Tool Results】

【Relevant Text Excerpts】
{text_content}

【Retrieved Images】
Retrieved {len(image_paths)} relevant images.
Image path list:
{chr(10).join(f"{i + 1}. {path}" for i, path in enumerate(image_paths)) if image_paths else "None"}
"""

                # 创建新的ToolMessage替换原来的
                new_messages = []
                for msg in messages:
                    if msg == message:
                        new_messages.append(ToolMessage(
                            content=updated_content,
                            tool_call_id=message.tool_call_id
                        ))
                    else:
                        new_messages.append(msg)

                return {
                    "messages": {"__overwrite__": new_messages},
                    "retrieved_images": retrieved_images
                }
            except json.JSONDecodeError:
                pass
    return {
        "messages": messages,
        "retrieved_images": retrieved_images
    }


# add_images节点
def add_images_to_context(state: AgentState) -> AgentState:
    """
    将图片路径添加到对话上下文中
    这个节点在工具处理后、模型再次调用前运行
    """
    messages = state["messages"]
    retrieved_images = state.get("retrieved_images", [])

    # 如果有图片且还没添加过
    if retrieved_images:
        # 构建包含图片的消息内容 - 使用通义千问的格式
        image_message_content = [
            {
                "type": "text",
                "text": "You are provided with image search results below. Analyze them carefully and cite relevant findings in your response when necessary."
            }
        ]

        project_root = get_project_root()

        # 添加所有图片 - 使用图片路径而不是base64编码
        for i, img_info in enumerate(retrieved_images):
            img_path = img_info["path"]
            original_path = img_info.get("original_path", img_path)

            # 路径已经是绝对路径（由 process_tool_output 处理过）
            abs_path = Path(img_path)
            if not abs_path.exists():
                # 如果文件不存在，尝试在 assets/paper_images 的子目录中查找文件名
                filename = abs_path.name
                possible_dirs = [
                    project_root / "assets" / "paper_images" / "transformer",
                    project_root / "assets" / "paper_images" / "lora",
                    project_root / "assets" / "paper_images" / "dpo",
                ]
                found = False
                for dir_path in possible_dirs:
                    candidate = dir_path / filename
                    if candidate.exists():
                        abs_path = candidate
                        found = True
                        break
                if not found:
                    # 跳过不存在的图片
                    continue

            try:
                file_url = f"file://{abs_path}"

                # 添加图片
                image_message_content.append({
                    "type": "image",
                    "image": file_url
                })
                # 在每张图片后添加路径说明
                image_message_content.append({
                    "type": "text",
                    "text": f"[Image {i + 1}] Path: {img_path} (原始路径: {original_path})"
                })
            except Exception as e:
                # 如果无法创建文件URL，跳过这张图片
                print_my_content(f"警告: 无法处理图片 {img_path}: {e}")
                continue

        # 将图片作为HumanMessage添加
        new_message = HumanMessage(content=image_message_content)
        return {
            "messages": [new_message],
            "retrieved_images": []
        }
    return {
        "messages": [],
        "retrieved_images": retrieved_images
    }


# 条件边，判断是否结束
def should_continue(state: AgentState) -> Literal["tools", "add_images", "end"]:
    """
    决定下一步的路由逻辑
    """
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息有工具调用，执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则结束
    return "end"


# ==================== 主函数 ====================
def init_agent1():
    """初始化并返回 agent1 实例"""
    tool_node = ToolNode(tools)
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_tool", process_tool_output)
    workflow.add_node("add_images", add_images_to_context)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "process_tool")
    workflow.add_edge("process_tool", "add_images")
    workflow.add_edge("add_images", "agent")

    # 关键：添加内存保存器以支持多轮对话
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


async def invoke_agent1(agent_instance, query: str, thread_id: str = "default_session"):
    """使用持久化的 agent 实例进行对话（支持会话隔离）"""
    config = {"configurable": {"thread_id": thread_id}}

    # 打印开始处理消息
    print_my_content(f"开始处理用户查询: {query[:100]}...", thread_id, "info")
    print_my_content("初始化agent1（现有知识库问答）...", thread_id, "info")

    # 首次调用时需要注入 SystemMessage
    # 检查该 thread_id 是否已有历史记录
    state = await agent_instance.aget_state(config)

    if not state.values.get("messages"):
        initial_input = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)],
            "retrieved_images": []
        }
        print_my_content("创建新的对话会话...", thread_id, "info")
    else:
        initial_input = {"messages": [HumanMessage(content=query)]}
        print_my_content("继续现有对话会话...", thread_id, "info")

    response_text = ""
    last_partition_name = None

    async for chunk in agent_instance.astream(initial_input, config=config, stream_mode="updates"):
        if "agent" in chunk:
            msg = chunk["agent"]["messages"][0]
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    args = tool_call.get("args", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    if "partition_name" in args:
                        last_partition_name = args["partition_name"]
                        print_my_content(f"检索分区: {last_partition_name} (会话: {thread_id})", thread_id, "info")
            else:
                # 提取最终回答文本
                if isinstance(msg.content, list):
                    response_text = "".join([item.get('text', '') for item in msg.content if isinstance(item, dict)])
                else:
                    response_text = msg.content

                # 打印回答生成完成
                if response_text:
                    print_my_content("回答生成完成", thread_id, "success")
                    print_my_content(f"回答长度: {len(response_text)} 字符 (会话: {thread_id})", thread_id, "info")

    pdf_mapping = {
        "transformer": "/home/seh/app/PaperMind/assets/awesome_papers/transformer.pdf",
        "lora": "/home/seh/app/PaperMind/assets/awesome_papers/lora.pdf",
        "dpo": "/home/seh/app/PaperMind/assets/awesome_papers/dpo.pdf"
    }
    pdf_path = pdf_mapping.get(last_partition_name, None)

    if pdf_path:
        print_my_content(f"关联PDF文档: {os.path.basename(pdf_path)} (会话: {thread_id})", thread_id, "success")
    else:
        print_my_content("未找到关联的PDF文档", thread_id, "warning")

    return response_text, pdf_path
