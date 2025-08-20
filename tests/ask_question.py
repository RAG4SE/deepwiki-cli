import openai
import os
import asyncio
import time

# 创建异步客户端
async_client = openai.AsyncOpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义两个独立的问题
questions = ["how are you", "what are you doing"]


async def ask_question(question, question_index):
    """异步处理单个问题"""
    try:
        print(f"🚀 开始处理问题 {question_index + 1}: {question}")

        api_kwargs = {
            "messages": [{"role": "user", "content": question}],
            "model": "qwen3-8b",
            "extra_body": {"enable_thinking": False},
        }

        response = await async_client.chat.completions.create(**api_kwargs)
        answer = response.choices[0].message.content

        print(f"✅ 问题 {question_index + 1} 完成")
        return {"question": question, "answer": answer, "index": question_index + 1}

    except Exception as e:
        print(f"❌ 问题 {question_index + 1} 失败: {e}")
        return {"question": question, "error": str(e), "index": question_index + 1}


async def main():
    """并行处理所有问题"""
    print("📝 开始并行处理两个问题...")
    start_time = time.time()

    # 并行执行所有问题
    tasks = [ask_question(q, i) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)

    end_time = time.time()

    # 格式化显示结果
    print(f"\n🎉 所有问题处理完成，耗时: {end_time - start_time:.2f}秒")
    print("=" * 60)

    for result in results:
        print(f"\n📋 问题 {result['index']}: {result['question']}")
        if "answer" in result:
            print(f"💬 回答: {result['answer']}")
        else:
            print(f"❌ 错误: {result['error']}")
        print("-" * 40)


# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())
