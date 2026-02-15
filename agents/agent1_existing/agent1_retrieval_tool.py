import asyncio
import json
import os
import time
from http import HTTPStatus
from typing import List, Dict, Any, Tuple

import dashscope
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import MilvusClient

load_dotenv()


class Qwen25VLEmbedding:
    def __init__(self, api_key: str, model: str = os.getenv("AGENT1_EMBEDDING_MODEL"), is_print: str = True):
        """
        初始化Qwen2.5-VL嵌入模型
        """
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key
        self.embedding_dim = 1024
        self.is_print = is_print

    def _print(self, *args, **kwargs):
        if self.is_print:
            print(*args, **kwargs)  # 如果允许打印,就调用内置的 print

    async def embed_query(self, text: str) -> List[float]:
        """
        输入文本,输出文本的向量 - 异步版本
        """
        return await self._embed_text(text)

    async def _embed_text(self, text: str) -> List[float]:
        """
        文本向量化的实现过程,调用qwen api - 异步版本
        """
        try:
            # 使用 asyncio.to_thread 将同步的 SDK 调用转为异步
            resp = await asyncio.to_thread(
                dashscope.MultiModalEmbedding.call,
                model=self.model,
                input=[{'text': text}]
            )

            if resp.status_code == 200 and resp.output:
                return resp.output['embeddings'][0]['embedding']
            else:
                self._print(f"  警告: 文本嵌入失败: {resp.message}")
                return [0.0] * self.embedding_dim
        except Exception as e:
            self._print(f"  警告: 文本嵌入失败: {e}")
            return [0.0] * self.embedding_dim


class MultimodalRetriever:
    def __init__(
            self,
            embedding_model: Qwen25VLEmbedding,  # 多模态嵌入模型
            text_collection_name: str = "paper_text_collection",  # 数据库中文本集合的名称
            image_collection_name: str = "paper_image_collection",  # 数据库中图片集合的名称
            is_print: bool = True
    ):
        self.embedding_model = embedding_model
        self.text_collection_name = text_collection_name
        self.image_collection_name = image_collection_name
        self.qwen_api_key = os.getenv("TONGYI_API_KEY")
        self.is_print = is_print

        # 连接Milvus
        # Milvus Client 通常是同步的,我们在异步方法中使用 run_in_executor/to_thread 来避免阻塞
        self.milvus_client = MilvusClient(
            uri=os.getenv("ZILLIZ_ENDPOINT"),
            user=os.getenv("ZILLIZ_USER"),
            password=os.getenv("ZILLIZ_PASS")
        )

        # 为MQE和HyDE使用不同的模型
        self.llm_MQE = ChatTongyi(
            model=os.getenv("AGENT1_MQE_MODEL"),
            api_key=os.getenv("TONGYI_API_KEY"),
            model_kwargs={"enable_thinking": False}  # 👈 关闭 thinking
        )
        self.llm_HyDE = ChatTongyi(
            model=os.getenv("AGENT1_HYDE_MODEL"),
            api_key=os.getenv("TONGYI_API_KEY"),
            model_kwargs={"enable_thinking": False}  # 👈 关闭 thinking
        )

    def _print(self, *args, **kwargs):
        if self.is_print:
            print(*args, **kwargs)  # 如果允许打印,就调用内置的 print

    async def multi_query_expansion(self, query: str, num_queries: int = 3) -> List[str]:
        """
        多查询扩展(MQE) - 异步版本
        Args:
            query: 原始查询
            num_queries: 生成的查询数量
        Returns:
            扩展后的查询列表(包含原始查询)
        """
        prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant that helps users generate multiple search queries.
            For the given user question, generate {num} different but related search queries from various perspectives.
            These queries should help find relevant information.

            Original question: {question}

            Return only the list of queries, one per line, without numbering or any other formatting."""
        )

        messages = prompt.format_messages(question=query, num=num_queries)
        # 使用 ainvoke 进行异步调用
        response = await self.llm_MQE.ainvoke(messages)
        # 解析生成的查询
        expanded_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        # 确保包含原始查询
        all_queries = [query] + expanded_queries[:num_queries]
        return all_queries

    async def hypothetical_document_embedding(self, query: str) -> str:
        """
        假设文档嵌入(HyDE) - 异步版本
        Args:
            query: 用户查询
        Returns:
            生成的假设文档
        """
        prompt = ChatPromptTemplate.from_template(
            """Based on the following question, generate a hypothetical, detailed answer document.
            This document should contain technical details and relevant information that could answer the question.

            Question: {question}

            Please generate a professional, detailed answer (100-200 words):"""
        )
        messages = prompt.format_messages(question=query)
        # 使用 ainvoke 进行异步调用
        response = await self.llm_HyDE.ainvoke(messages)
        return response.content.strip()

    async def search_collection(
            self,
            query_embedding: List[float],
            collection_name: str,
            partition_name: str,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        在指定集合和分区中搜索 - 异步版本
        Args:
            query_embedding: 查询向量
            collection_name: 集合名称
            partition_name: 分区名称
            top_k: 返回结果数量
        Returns:
            搜索结果列表
        """
        if collection_name == self.text_collection_name:
            vector_field = "text_embedding"
        else:
            vector_field = "image_embedding"

        # 使用 asyncio.to_thread 将同步的 Milvus 搜索转为异步
        results = await asyncio.to_thread(
            self.milvus_client.search,
            collection_name=collection_name,
            data=[query_embedding],
            anns_field=vector_field,
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            partition_names=[partition_name],
            output_fields=["*"]
        )
        return results[0] if results else []

    async def _process_single_query_async(self, q: str, partition_name: str, top_k: int) -> Tuple[List, List]:
        """
        内部辅助函数:处理单个查询字符串的完整检索流程(Embedding -> Milvus Search)
        """
        # 1. 生成向量 (现在 embed_query 已经是异步的)
        query_embedding = await self.embedding_model.embed_query(q)

        # 2. 并行搜索 Text 和 Image 集合 (search_collection 现在也是异步的)
        text_results, image_results = await asyncio.gather(
            self.search_collection(
                query_embedding=query_embedding,
                collection_name=self.text_collection_name,
                partition_name=partition_name,
                top_k=top_k
            ),
            self.search_collection(
                query_embedding=query_embedding,
                collection_name=self.image_collection_name,
                partition_name=partition_name,
                top_k=top_k
            )
        )

        return text_results, image_results

    async def retrieve(
            self,
            query: str,
            partition_name: str,
            use_mqe: bool = True,
            use_hyde: bool = True,
            top_k: int = 10,
            image_top_k: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        执行检索 - 异步版本
        Args:
            query: 用户查询
            partition_name: 分区名称 (dahua, hik, uniview)
            use_mqe: 是否使用多查询扩展
            use_hyde: 是否使用假设文档嵌入
            top_k: 每个查询返回的文本结果数量
            image_top_k: 每个查询返回的图片结果数量，如果为None则使用top_k的一半
        Returns:
            包含文本和图片检索结果的字典
        """
        # 设置图片检索数量，默认为文本的一半
        if image_top_k is None:
            image_top_k = max(1, top_k // 2)
        all_queries = []

        self._print("\n" + "=" * 40)
        self._print("第一步:生成查询拓展(MQE)和假设文档嵌入(HyDE)")
        self._print("=" * 40)

        # 使用 asyncio.gather 并行执行 MQE 和 HyDE
        tasks = []

        # 1. 多查询扩展任务
        if use_mqe:
            self._print("异步调用LLM:执行多查询扩展...")
            tasks.append(self.multi_query_expansion(query, num_queries=5))
        else:
            # 如果不使用MQE,我们需要保持列表结构对齐,这里手动添加原始查询
            all_queries.append(query)
            tasks.append(asyncio.sleep(0))  # 占位符,保持索引一致性逻辑简单化

        # 2. 假设文档嵌入任务
        if use_hyde:
            self._print("异步调用LLM :生成假设文档...")
            tasks.append(self.hypothetical_document_embedding(query))

        # 执行第一步的所有异步任务
        if tasks:
            results = await asyncio.gather(*tasks)

            # 处理结果
            idx = 0
            if use_mqe:
                all_queries.extend(results[idx])
                idx += 1
            else:
                idx += 1  # 跳过占位符

            if use_hyde:
                all_queries.append(results[idx])

        # 3. 对所有查询进行检索
        text_results_all = []  # 在文本集合中的查询结果
        image_results_all = []  # 在图片集合中的查询结果

        self._print("\n" + "=" * 40)
        self._print("第二步:查询所有生成的字段")
        self._print("=" * 40)

        # 批量处理查询嵌入以减少API调用次数
        query_embeddings = await asyncio.gather(*[self.embedding_model.embed_query(q) for q in all_queries])

        # 并行执行所有查询的检索操作
        search_tasks = []
        for i, q in enumerate(all_queries):
            self._print(f"并发执行:查询检索: {q[:10]}...")
            query_embedding = query_embeddings[i]

            # 直接并行搜索两个集合，使用不同的top_k
            text_search_task = self.search_collection(
                query_embedding=query_embedding,
                collection_name=self.text_collection_name,
                partition_name=partition_name,
                top_k=top_k
            )
            image_search_task = self.search_collection(
                query_embedding=query_embedding,
                collection_name=self.image_collection_name,
                partition_name=partition_name,
                top_k=image_top_k
            )

            search_tasks.append(asyncio.gather(text_search_task, image_search_task))

        # 并发执行所有查询的 检索
        if search_tasks:
            search_results_list = await asyncio.gather(*search_tasks)

            for t_res, i_res in search_results_list:
                text_results_all.extend(t_res)
                image_results_all.extend(i_res)

        self._print("\n" + "=" * 40)
        self._print("第三步:查询结果去重排序与截断")
        self._print("=" * 40)

        # 4. 去重并按分数排序
        # 注意:此处维持同步执行。
        # 原因:去重排序是纯内存CPU操作(字典查找和列表排序),数据量级较小(<1000条)。
        # 在Python GIL限制下,使用async/await或多线程处理CPU密集型任务并不会带来性能提升,
        # 反而可能因上下文切换增加开销。
        text_results_unique = self._deduplicate_results(text_results_all, "chunk_id")
        self._print(f"1.对文本检索结果进行去重、排序 (CPU密集型任务,保持同步)")
        image_results_unique = self._deduplicate_results(image_results_all, "image_id")
        self._print(f"2.对图片检索结果进行去重、排序 (CPU密集型任务,保持同步)")
        self._print(f"3.统一截断,文本保留得分最高的{top_k * 2}条记录,图片保留得分最高的{image_top_k * 2}条记录")

        return {  # 由于生成了 n 个相关的查询字段(MQE和HyDE),所以会查询 n * top_k个文本和 n * image_top_k个图片出来
            "text_results": text_results_unique[:top_k * 2],
            "image_results": image_results_unique[:image_top_k * 2]
        }

    def _deduplicate_results(
            self,
            results: List[Dict[str, Any]],
            id_field: str
    ) -> List[Dict[str, Any]]:
        """
        去重并保留最高分数的结果
        Args:
            results: 原始结果列表
            id_field: 用于去重的ID字段
        Returns:
            去重后的结果列表
        """
        seen = {}
        for result in results:
            result_id = result['entity'].get(id_field)
            score = result.get('distance', 0)
            if result_id not in seen or score > seen[result_id].get('distance', 0):
                seen[result_id] = result
        # 按分数排序
        unique_results = list(seen.values())
        unique_results.sort(key=lambda x: x.get('distance', 0), reverse=True)
        return unique_results

    async def rerank(
            self,
            query: str,
            documents: List[str],
            top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        使用阿里云百炼的Qwen3-Rerank进行重排序 - 异步版本
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_n: 返回前N个结果
        Returns:
            重排序后的结果列表,格式: [{'index': int, 'relevance_score': float}, ...]
        """
        try:
            # 使用 asyncio.to_thread 将同步的 API 调用转为异步
            resp = await asyncio.to_thread(
                dashscope.TextReRank.call,
                model=os.getenv("AGENT1_RERANK_MODEL"),
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=True
            )

            if resp.status_code == HTTPStatus.OK:
                # 解析返回结果
                results = []
                for item in resp.output.results:
                    results.append({
                        'index': item.index,
                        'relevance_score': item.relevance_score
                    })
                return results
            else:
                self._print(f"重排序失败: {resp.code} - {resp.message}")
                return []
        except Exception as e:
            self._print(f"重排序失败: {e}")
            return []

    async def retrieve_and_rerank(
            self,
            query: str,
            partition_name: str,
            use_mqe: bool = True,
            use_hyde: bool = True,
            top_k: int = 10,
            rerank_top_n: int = 5,
            image_top_k: int = None
    ) -> Dict[str, Any]:
        """
        执行检索和重排序的完整流程 - 异步版本
        Args:
            query: 用户查询
            partition_name: 分区名称
            use_mqe: 是否使用多查询扩展
            use_hyde: 是否使用假设文档嵌入
            top_k: 初始文本检索数量
            rerank_top_n: 重排序后返回数量
            image_top_k: 初始图片检索数量，如果为None则使用top_k的一半
        Returns:
            包含重排序后的文本和图片结果
        """
        # 1. 检索 (Await 异步检索方法)
        results = await self.retrieve(
            query=query,
            partition_name=partition_name,
            use_mqe=use_mqe,
            use_hyde=use_hyde,
            top_k=top_k,
            image_top_k=image_top_k
        )

        self._print("\n" + "=" * 40)
        self._print("第四步:对查询结果进行重排序")
        self._print("=" * 40)

        # 准备文本重排序数据
        text_docs = []
        text_metadata = []
        for result in results['text_results']:
            entity = result['entity']
            text_docs.append(entity.get('text', ''))
            text_metadata.append({
                'chunk_id': entity.get('chunk_id'),
                'page_numbers': entity.get('page_numbers'),
                'has_images': entity.get('has_images'),
                'image_paths': entity.get('image_paths'),
                'metadata': entity.get('metadata'),
                'original_score': result.get('distance', 0)
            })

        # 准备图片重排序数据
        image_docs = []
        image_metadata = []
        for result in results['image_results']:
            entity = result['entity']
            # 使用文本上下文作为重排序的文本
            image_docs.append(entity.get('text_context', ''))
            image_metadata.append({
                'image_id': entity.get('image_id'),
                'image_path': entity.get('image_path'),  # 本地路径
                'page_numbers': entity.get('page_numbers'),
                'text_context': entity.get('text_context'),
                'metadata': entity.get('metadata'),
                'original_score': result.get('distance', 0)
            })

        # 异步并发执行重排序 (rerank 现在也是异步的)
        self._print(f"并发执行:对文本和图片的检索结果进行重排序...")

        # 定义任务
        rerank_text_task = self.rerank(query, text_docs, top_n=rerank_top_n) if text_docs else asyncio.sleep(0)
        rerank_image_task = self.rerank(query, image_docs, top_n=rerank_top_n) if image_docs else asyncio.sleep(0)

        # 等待结果
        rerank_text_results, rerank_image_results = await asyncio.gather(rerank_text_task, rerank_image_task)

        # 处理文本结果
        reranked_text = []
        if text_docs and isinstance(rerank_text_results, list):
            for rr in rerank_text_results:
                idx = rr['index']
                reranked_text.append({
                    'text': text_docs[idx],
                    'rerank_score': rr['relevance_score'],
                    **text_metadata[idx]
                })

        # 处理图片结果
        reranked_images = []
        if image_docs and isinstance(rerank_image_results, list):
            for rr in rerank_image_results:
                idx = rr['index']
                reranked_images.append({
                    'rerank_score': rr['relevance_score'],
                    **image_metadata[idx]
                })

        self._print("\n" + "=" * 40)
        self._print("第五步:完成检索过程")
        self._print("=" * 40)
        self._print(f"文本结果: {len(reranked_text)}个\n图片结果: {len(reranked_images)}个")

        # 收集需要的属性
        text_set = set()
        image_path_set = set()

        # 处理文本结果
        for item in reranked_text:
            text = item.get('text')
            if text and isinstance(text, str):
                text_set.add(text.strip())

            # 如果 has_images 为 true,收集 ['metadata']['image_path']
            if item.get('has_images'):
                metadata_dict = json.loads(item.get('metadata'))
                if isinstance(metadata_dict, dict):
                    image_path = metadata_dict.get('image_paths')[0]
                    if image_path and isinstance(image_path, str):
                        image_path_set.add(image_path.strip())

        # 处理图片结果
        for item in reranked_images:
            text_context = item.get('text_context')
            if text_context and isinstance(text_context, str):
                text_set.add(text_context.strip())

            # 收集 ['image_path']
            image_path = item.get('image_path')
            if image_path and isinstance(image_path, str):
                image_path_set.add(image_path.strip())

        return {
            'text': list(text_set),
            'image_path': list(image_path_set)
        }


# ==================== 创建检索工具并返回 ====================
def create_retrieval_tool(api_key: str = None, is_print: bool = False):
    """
    创建检索工具的工厂函数
    Args:
        api_key: 通义千问API密钥，如果为None则从环境变量获取
        is_print: 是否打印调试信息
    Returns:
        配置好的检索工具函数
    """
    import json
    from langchain_core.tools import tool

    if api_key is None:
        api_key = os.getenv("TONGYI_API_KEY")

    # 创建检索器实例
    retriever = MultimodalRetriever(
        embedding_model=Qwen25VLEmbedding(
            api_key=api_key,
            model=os.getenv("AGENT1_EMBEDDING_MODEL")
        ),
        is_print=is_print
    )

    @tool
    async def retrieve_tool(
            query: str,
            partition_name: str,
            use_mqe: bool = True,
            use_hyde: bool = True,
            top_k: int = 5,
            rerank_top_n: int = 3
    ) -> str:
        """
        Perform multimodal retrieval using the retriever.
        First retrieves candidate documents from the specified partition using the user query.
        Optionally enables MQE (Multi-Query Expansion) and HyDE (Hypothetical Document Embeddings) for improved recall.
        Then uses a reranker to reorder and return the top `rerank_top_n` most relevant document contents.

        Args:
            query (str): Retrieval query text. Should be clear and specific for optimal results.
            partition_name (str): Data partition name. Must be one of: "transformer", "lora", or "dpo".
            use_mqe (bool): Whether to enable Multi-Query Expansion. Default is True.
                            Enabling this improves recall but increases retrieval time. It is recommended to disable this for simple queries to save latency.
            use_hyde (bool): Whether to enable Hypothetical Document Embeddings. Default is True.
                            Enabling this improves recall but increases retrieval time. It is recommended to disable this for simple queries to save latency.
            top_k (int): Initial retrieval candidate document count. Default is 5.
                         Increasing this value can help when results are not ideal, but it will increase retrieval time.
            rerank_top_n (int): Number of documents to return after reranking. Default is 3.
                                A value between 3 and 5 is recommended to balance precision and speed.

        Returns:
            str: JSON-formatted string containing text content and image paths.
        """
        try:
            result = await retriever.retrieve_and_rerank(
                query=query,
                partition_name=partition_name,
                use_mqe=use_mqe,
                use_hyde=use_hyde,
                top_k=top_k,
                rerank_top_n=rerank_top_n
            )
            # Return JSON format for easy parsing
            return json.dumps({
                "text": result.get("text", "No relevant text found"),
                "image_paths": result.get("image_path", [])
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "error": f"Retrieval failed: {str(e)}",
                "text": "",
                "image_paths": []
            }, ensure_ascii=False)

    return retrieve_tool


if __name__ == "__main__":
    async def main():
        # 创建检索器
        retriever = MultimodalRetriever(
            embedding_model=Qwen25VLEmbedding(
                api_key=os.getenv("TONGYI_API_KEY"),
                model=os.getenv("AGENT1_EMBEDDING_MODEL")
            ),
            is_print=False
        )

        start_time = time.time()

        # 执行检索和重排序 (异步等待)
        results = await retriever.retrieve_and_rerank(
            query="How to calculate the multi-head attention mechanism in Transformer",
            partition_name="transformer",
            use_mqe=True,
            use_hyde=True,
            top_k=5,
            rerank_top_n=3
        )

        end_time = time.time()
        print(f"\n总耗时: {end_time - start_time:.2f} 秒")

        # 打印结果 - 更新为新的返回结构
        print("\n" * 2 + "=" * 40)
        print("收集的文本内容")
        print("=" * 40)
        if results['text']:
            for i, text in enumerate(results['text'], 1):
                clean_text = text.replace('\n', ' ').strip()
                print(f"📊 {i}. {clean_text[:50]}..." if len(clean_text) > 50 else f"📊 {i}. {clean_text}")
        else:
            print("没有收集到文本内容")

        print("\n" * 2 + "=" * 40)
        print("收集的图片路径")
        print("=" * 40)
        if results['image_path']:
            for i, image_path in enumerate(results['image_path'], 1):
                print(f"🖼️ {i}. {image_path}")
        else:
            print("没有收集到图片路径")


    # 运行异步主循环
    asyncio.run(main())
