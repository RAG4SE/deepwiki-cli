"""
对比异步协程 vs 线程池执行器的并发处理方式
"""

import openai
import os
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# 配置
API_KEY = os.environ["DASHSCOPE_API_KEY"]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

questions = [
    "how are you",
    "what are you doing",
    "tell me a joke",
    "what's the weather like",
]

# ===========================================
# 方式1: 异步协程 (当前使用的方式)
# ===========================================


async def async_approach():
    """使用asyncio协程的并发处理"""
    print("🔄 方式1: 异步协程 (asyncio)")
    print(f"📊 主线程ID: {threading.get_ident()}")

    async_client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    async def ask_question_async(question: str, index: int) -> Dict[str, Any]:
        thread_id = threading.get_ident()
        print(f"  🚀 协程 {index+1} 开始 (线程 {thread_id}): {question}")

        try:
            api_kwargs = {
                "messages": [{"role": "user", "content": question}],
                "model": "qwen3-8b",
                "extra_body": {"enable_thinking": False},
            }

            response = await async_client.chat.completions.create(**api_kwargs)
            answer = response.choices[0].message.content

            print(f"  ✅ 协程 {index+1} 完成 (线程 {thread_id}): {answer[:50] + '...'}")
            return {
                "index": index + 1,
                "question": question,
                "answer": answer[:50] + "...",
            }

        except Exception as e:
            print(f"  ❌ 协程 {index+1} 失败: {e}")
            return {"index": index + 1, "question": question, "error": str(e)}

    start_time = time.time()

    # 并行执行所有协程
    tasks = [ask_question_async(q, i) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)

    end_time = time.time()

    print(f"  ⏱️  协程方式耗时: {end_time - start_time:.2f}秒")
    return results, end_time - start_time


# ===========================================
# 方式2: 线程池执行器
# ===========================================


def thread_pool_approach():
    """使用ThreadPoolExecutor的并发处理"""
    print("\n🧵 方式2: 线程池执行器 (ThreadPoolExecutor)")
    print(f"📊 主线程ID: {threading.get_ident()}")

    # 注意：对于同步OpenAI客户端，每个线程需要独立的客户端实例
    def ask_question_sync(question: str, index: int) -> Dict[str, Any]:
        thread_id = threading.get_ident()
        print(f"  🚀 线程 {index+1} 开始 (线程 {thread_id}): {question}")

        # 每个线程创建独立的同步客户端
        sync_client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

        try:
            api_kwargs = {
                "messages": [{"role": "user", "content": question}],
                "model": "qwen3-8b",
                "extra_body": {"enable_thinking": False},
            }

            response = sync_client.chat.completions.create(**api_kwargs)
            answer = response.choices[0].message.content

            print(f"  ✅ 线程 {index+1} 完成 (线程 {thread_id}): {answer[:50] + '...'}")
            return {
                "index": index + 1,
                "question": question,
                "answer": answer[:50] + "...",
            }

        except Exception as e:
            print(f"  ❌ 线程 {index+1} 失败: {e}")
            return {"index": index + 1, "question": question, "error": str(e)}

    start_time = time.time()

    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(ask_question_sync, q, i): i for i, q in enumerate(questions)
        }

        results = [None] * len(questions)

        # 收集结果
        for future in as_completed(future_to_index):
            result = future.result()
            results[result["index"] - 1] = result

    end_time = time.time()

    print(f"  ⏱️  线程池方式耗时: {end_time - start_time:.2f}秒")
    return results, end_time - start_time


# ===========================================
# 主函数对比两种方式
# ===========================================


async def main():
    """对比两种并发处理方式"""
    print("🔬 并发处理方式对比测试")
    print("=" * 80)

    # 测试异步协程
    async_results, async_time = await async_approach()

    # 测试线程池
    thread_results, thread_time = thread_pool_approach()

    # 对比结果
    print(f"\n📈 性能对比:")
    print(f"  🔄 异步协程耗时: {async_time:.2f}秒")
    print(f"  🧵 线程池耗时: {thread_time:.2f}秒")
    print(f"  📊 性能差异: {abs(async_time - thread_time):.2f}秒")

    if async_time < thread_time:
        print(
            f"  🏆 异步协程快 {((thread_time - async_time) / thread_time * 100):.1f}%"
        )
    else:
        print(f"  🏆 线程池快 {((async_time - thread_time) / async_time * 100):.1f}%")


# ===========================================
# 理论对比说明
# ===========================================


def print_theoretical_comparison():
    """打印理论对比"""
    print("\n" + "=" * 80)
    print("📚 理论对比: 异步协程 vs 线程池")
    print("=" * 80)

    comparison = [
        ("🔄 并发模型", "单线程事件循环，协程切换", "多线程，抢占式调度"),
        ("💾 内存开销", "轻量级协程 (~KB级)", "重量级线程 (~MB级)"),
        ("🔀 上下文切换", "用户态切换，开销极小", "内核态切换，开销较大"),
        ("🐍 GIL影响", "单线程运行，不受GIL限制", "受Python GIL限制"),
        ("🎯 适用场景", "I/O密集型任务（网络请求）", "CPU密集型或阻塞I/O"),
        ("📈 扩展性", "可轻松处理数万并发", "受线程数量限制"),
        ("🐛 调试难度", "相对简单，单线程执行", "复杂，需考虑线程安全"),
        ("🔧 资源管理", "自动管理，无需手动清理", "需要管理线程池大小"),
    ]

    for aspect, async_feature, thread_feature in comparison:
        print(f"{aspect}")
        print(f"  🔄 异步协程: {async_feature}")
        print(f"  🧵 线程池:   {thread_feature}")
        print()


if __name__ == "__main__":
    print_theoretical_comparison()
    asyncio.run(main())
