import hashlib
import os
import sys
import warnings
from typing import Dict, Any, List

import arxiv
import requests
from openai import OpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

# 导入web_print模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.web_print import print_my_content

# ==================== 初始化配置 ====================
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
load_dotenv()

# 全局配置
MILVUS_DB_PATH = os.getenv("AGENT3_DOC_DB_PATH")
EMBEDDING_MODEL = os.getenv("AGENT3_EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_DIM = int(os.getenv("AGENT3_EMBEDDING_DIM", "1024"))

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

# 创建 arxiv 客户端
arxiv_client = arxiv.Client()


# ==================== 辅助函数 ====================

def get_collection_name(file_path: str, thread_id: str = "default_session") -> str:
    """
    根据文件路径和会话ID生成唯一的collection名称（支持会话隔离）

    Args:
        file_path: PDF文件路径
        thread_id: 会话ID

    Returns:
        唯一的collection名称
    """
    # 获取文件的规范化路径
    abs_path = os.path.abspath(file_path)

    # 使用MD5哈希生成唯一标识
    # 包含文件路径、文件名和会话ID
    hash_input = f"{abs_path}_{thread_id}"
    collection_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

    # 使用文件名（不含扩展名）作为前缀
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]

    # 清理文件名中的特殊字符，只保留字母数字和下划线
    safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in file_name_without_ext)

    # 限制长度，避免名称过长
    safe_name = safe_name[:30]

    return f"paper_{safe_name}_{collection_hash}"


def get_file_signature(file_path: str) -> Dict[str, Any]:
    """
    获取文件签名（用于检测文件是否变更）

    Args:
        file_path: 文件路径

    Returns:
        包含文件签名信息的字典
    """
    try:
        stat_info = os.stat(file_path)
        return {
            "size": stat_info.st_size,
            "mtime": stat_info.st_mtime,
            "path": os.path.abspath(file_path)
        }
    except:
        return {}


def collection_exists(client: MilvusClient, collection_name: str) -> bool:
    """
    检查集合是否存在

    Args:
        client: Milvus客户端
        collection_name: 集合名称

    Returns:
        集合是否存在
    """
    try:
        return client.has_collection(collection_name)
    except:
        return False


# ==================== 嵌入函数 ====================

def get_embedding(text: str) -> List[float]:
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
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIM,
        encoding_format="float"
    )
    
    return response.data[0].embedding


def embed_query(query: str) -> List[float]:
    """
    向量化查询文本（兼容原接口）
    
    Args:
        query: 查询文本
        
    Returns:
        查询向量
    """
    return get_embedding(query)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    批量向量化文档文本（兼容原接口）
    
    Args:
        texts: 文本列表
        
    Returns:
        向量列表
    """
    if openai_client is None:
        raise Exception("OpenAI客户端未初始化，请检查TONGYI_API_KEY环境变量")
    
    # 如果只有一个文本，直接调用单个API
    if len(texts) == 1:
        return [get_embedding(texts[0])]
    
    all_embeddings = []
    batch_size = 10  # 每批最多10个文本
    
    # 分批处理文本
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 批量API调用
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts,
            dimensions=EMBEDDING_DIM,
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


# ==================== arXiv 工具 ====================

@tool
def search_arxiv_papers(query: str, max_results: int = 10, thread_id: str = "default_session") -> Dict[str, Any]:
    """
    Search for academic papers on arXiv (supports session isolation).

    Args:
        query: Search keywords, e.g., 'reinforcement learning', 'transformer neural network'.
        max_results: Maximum number of results to return. Default is 10.
        thread_id: Session ID for isolation.

    Returns:
        A dictionary containing the status, number of results, and list of papers with details.
    """
    try:
        print_my_content(f"正在搜索arXiv论文: {query}", thread_id, "info")

        # 使用 arxiv API 搜索
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for i, paper in enumerate(arxiv_client.results(search), 1):
            paper_id = paper.entry_id.split('/')[-1]
            paper_info = {
                "index": i,
                "paper_id": paper_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published_date": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "arxiv_url": paper.entry_id
            }
            papers.append(paper_info)

        if not papers:
            print_my_content(f"未找到相关论文: {query}", thread_id, "warning")
            return {
                "status": "success",
                "message": f"No papers found for query: '{query}'",
                "count": 0,
                "papers": []
            }

        print_my_content(f"找到 {len(papers)} 篇相关论文 (会话: {thread_id})", thread_id, "success")
        return {
            "status": "success",
            "message": f"Found {len(papers)} papers for query: '{query}'",
            "count": len(papers),
            "papers": papers
        }

    except Exception as e:
        print_my_content(f"arXiv搜索失败: {str(e)}", thread_id, "error")
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "count": 0,
            "papers": []
        }


@tool
def download_arxiv_paper(paper_id: str, save_path: str = "", thread_id: str = "default_session") -> Dict[str, Any]:
    """
    Download a PDF file from arXiv by paper ID (supports session isolation).

    Args:
        paper_id: arXiv paper ID, e.g., '2401.12345', '1706.03762', or full URL.
        save_path: Path to save the PDF file. If empty string, uses format '/home/seh/app/PaperMind/agents/agent3_arxiv/files/arxiv_{paper_id}.pdf'.
        thread_id: Session ID for isolation.

    Returns:
        A dictionary containing the status, file path, and paper metadata.
    """
    try:
        print_my_content(f"正在下载arXiv论文: {paper_id}", thread_id, "info")

        # 清理 paper_id,移除可能的前缀
        paper_id = paper_id.replace("https://arxiv.org/abs/", "")
        paper_id = paper_id.replace("http://arxiv.org/abs/", "")
        paper_id = paper_id.replace("https://arxiv.org/pdf/", "")
        paper_id = paper_id.replace("http://arxiv.org/pdf/", "")
        paper_id = paper_id.replace(".pdf", "")
        paper_id = paper_id.strip()

        # 使用 arxiv API 获取论文信息
        search = arxiv.Search(id_list=[paper_id])
        paper = next(arxiv_client.results(search))

        # 构建 PDF URL
        pdf_url = paper.pdf_url

        # 下载 PDF（设置1分钟超时）
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        # 确定保存路径
        if not save_path or save_path.strip() == "":
            save_path = f"/home/seh/app/PaperMind/agents/agent3_arxiv/files/arxiv_{paper_id.replace('/', '_').replace('v', '_v')}.pdf"

        # 保存 PDF
        with open(save_path, 'wb') as f:
            f.write(response.content)

        file_size_kb = len(response.content) / 1024
        file_size_mb = file_size_kb / 1024

        print_my_content(f"论文下载成功: {paper.title} (会话: {thread_id})", thread_id, "success")
        print_my_content(f"文件大小: {file_size_mb:.2f} MB", thread_id, "info")

        return {
            "status": "success",
            "message": "Paper downloaded successfully",
            "paper_id": paper_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "published_date": paper.published.strftime("%Y-%m-%d"),
            "file_path": save_path,
            "file_size_kb": round(file_size_kb, 2),
            "file_size_mb": round(file_size_mb, 2),
            "pdf_url": pdf_url
        }

    except StopIteration:
        print_my_content(f"论文未找到: {paper_id}", thread_id, "error")
        return {
            "status": "error",
            "message": f"Paper not found: {paper_id}",
            "paper_id": paper_id,
            "file_path": None
        }
    except requests.exceptions.RequestException as e:
        print_my_content(f"论文下载失败: {str(e)}", thread_id, "error")
        return {
            "status": "error",
            "message": f"Download failed: {str(e)}",
            "paper_id": paper_id,
            "file_path": None
        }
    except Exception as e:
        print_my_content(f"下载过程出错: {str(e)}", thread_id, "error")
        return {
            "status": "error",
            "message": f"Error occurred: {str(e)}",
            "paper_id": paper_id,
            "file_path": None
        }


# ==================== PDF 搜索工具 ====================

@tool
def search_pdf_tool(file_path: str, query: str, top_k: int = 7, thread_id: str = "default_session") -> Dict[str, Any]:
    """
    Search for relevant content from a single PDF file (supports session isolation).

    Args:
    file_path: Path to the PDF file.
    query: Search query/question.
    top_k: Number of most relevant results to return. Default is 7.
    thread_id: Session ID for isolation.

    Returns:
    A dictionary containing the status and relevant document content.
    """

    try:
        print_my_content(f"正在搜索PDF文件: {os.path.basename(file_path)}", thread_id, "info")
        print_my_content(f"搜索查询: {query}", thread_id, "info")

        # 1. 验证文件路径
        if not os.path.exists(file_path):
            print_my_content(f"文件不存在: {file_path}", thread_id, "error")
            return {
                "status": "error",
                "message": f"File does not exist: {file_path}"
            }

        if not file_path.lower().endswith('.pdf'):
            print_my_content(f"不是PDF文件: {file_path}", thread_id, "error")
            return {
                "status": "error",
                "message": f"Not a PDF file: {file_path}"
            }

        # 2. 生成collection名称并初始化客户端（使用会话ID）
        collection_name = get_collection_name(file_path, thread_id)
        client = MilvusClient(MILVUS_DB_PATH)

        # 3. 检查是否已有缓存
        collection_already_exists = collection_exists(client, collection_name)

        if collection_already_exists:
            print_my_content("使用缓存向量数据库", thread_id, "info")
            # 直接使用现有集合进行搜索
            query_embedding = embed_query(query)

            search_results = client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "source_file", "page"]
            )

            # 格式化返回结果
            results = []
            for hits in search_results:
                for hit in hits:
                    results.append({
                        "content": hit["entity"]["text"],
                        "source_file": hit["entity"]["source_file"],
                        "page": hit["entity"]["page"],
                        "score": hit["distance"]
                    })

            print_my_content(f"从缓存检索到 {len(results)} 个相关片段 (会话: {thread_id})", thread_id, "success")
            return {
                "status": "success",
                "query": query,
                "collection_name": collection_name,
                "results_count": len(results),
                "results": results,
                "message": f"Successfully retrieved {len(results)} relevant results from cache",
                "cached": True
            }
        else:
            print_my_content("创建新的向量数据库", thread_id, "info")
            # 4. 加载并分割PDF文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "!", "?", ",", " ", ""]
            )

            try:
                print_my_content("加载PDF文档...", thread_id, "info")
                loader = PyPDFLoader(file_path)
                documents = loader.load()

                # 添加文件来源元数据
                for doc in documents:
                    doc.metadata["source_file"] = os.path.basename(file_path)

                all_documents = text_splitter.split_documents(documents)
                print_my_content(f"文档分割完成，共 {len(all_documents)} 个片段", thread_id, "info")

            except Exception as e:
                print_my_content(f"加载PDF失败: {str(e)}", thread_id, "error")
                return {
                    "status": "error",
                    "message": f"Failed to load file {file_path}: {str(e)}"
                }

            if not all_documents:
                print_my_content("无法从PDF提取内容", thread_id, "error")
                return {
                    "status": "error",
                    "message": "Unable to extract content from PDF file"
                }

            # 5. 批量向量化文档
            print_my_content("向量化文档片段...", thread_id, "info")
            texts = [doc.page_content for doc in all_documents]
            metadatas = [doc.metadata for doc in all_documents]

            # 批量生成embeddings
            doc_embeddings = embed_documents(texts)

            # 6. 创建MilvusLite集合
            # 检查并创建集合
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)

            # 创建集合schema
            client.create_collection(
                collection_name=collection_name,
                dimension=EMBEDDING_DIM,
                metric_type="COSINE",
                auto_id=True
            )

            # 7. 插入向量数据
            print_my_content("存储向量到数据库...", thread_id, "info")
            entities = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, doc_embeddings, metadatas)):
                entities.append({
                    "vector": embedding,
                    "text": text,
                    "source_file": metadata.get("source_file", "unknown"),
                    "page": metadata.get("page", 0),
                    "thread_id": thread_id
                })

            client.insert(
                collection_name=collection_name,
                data=entities
            )

            # 8. 向量化查询并搜索
            query_embedding = embed_query(query)

            search_results = client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "source_file", "page", "thread_id"]
            )

            # 9. 格式化返回结果
            results = []
            for hits in search_results:
                for hit in hits:
                    results.append({
                        "content": hit["entity"]["text"],
                        "source_file": hit["entity"]["source_file"],
                        "page": hit["entity"]["page"],
                        "score": hit["distance"]
                    })

            print_my_content(f"检索到 {len(results)} 个相关片段 (会话: {thread_id})", thread_id, "success")
            return {
                "status": "success",
                "query": query,
                "collection_name": collection_name,
                "results_count": len(results),
                "results": results,
                "message": f"Successfully retrieved {len(results)} relevant results",
                "cached": False
            }

    except Exception as e:
        print_my_content(f"PDF搜索过程出错: {str(e)}", thread_id, "error")
        return {
            "status": "error",
            "message": f"Error occurred during PDF search: {str(e)}"
        }
