from openai import OpenAI

# 1. 你的中转平台 base_url，按平台文档填
# ⚠️ 只写到 /v1，千万不要写到 /chat/completions
BASE_URL = "https://api.bianxie.ai/v1"

# 2. 你的中转平台发给你的 key（就是 sk-vLFNYoL... 那个）
API_KEY = "sk-vLFNYoLiyacWZVaUMvg9BINr2Ei7e7r4tbpeLSuMsxhFaM0m"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,   # 注意是 base_url，不是 api_base
)

print("BASE_URL =", BASE_URL)
print("KEY HEAD =", API_KEY[:10], "...")

try:
    resp = client.chat.completions.create(
        model="chatgpt-4o-latest",        # 平台允许你用的那个模型
        messages=[{"role": "user", "content": "说一句hello就可以"}],
        max_tokens=10,
    )
    print("RESULT:", resp.choices[0].message.content)
except Exception as e:
    print("API 错误：", e)
