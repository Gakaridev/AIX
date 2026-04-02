from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://huggingface.co/mradermacher/TinyLlama2-jp-122M-GGUF/resolve/main/TinyLlama2-jp-122M.Q8_0.gguf"
MODEL_PATH = "model.gguf"

if not os.path.exists(MODEL_PATH):
    print("モデルをインストール中...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("インストール完了！")

llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=1)

# --- ここがメモリー機能！ ---
# 直近の会話を保存するリスト（スリープするとリセットされます）
chat_history = []

@app.get("/")
def index():
    return {"status": "Gakari AI is Ready!"}

@app.post("/chat")
def chat(data: dict):
    user_input = data.get("text", "")
    
    # 1. システムプロンプト（憲法）
    system_rules = (
        "あなたはGakari AIです。新聞係が開発したテスト中のAIです。 "
        "人を不快にさせず、Gakari AI以外の名前は名乗りません。短く回答します。"
        "あなたは、newspaper-committee.vercel.appの情報しかわかりません"
    )
    
    # 2. 過去の履歴を結合（直近3往復分に絞ってメモリ節約）
    history_text = "\n".join(chat_history[-6:])
    
    # 3. AIに渡す最終的なプロンプト
    # 「ルール」→「履歴」→「今の質問」の順に並べるとAIが理解しやすいです
    full_prompt = f"システム: {system_rules}\n{history_text}\nユーザー: {user_input}\nシステム: "
    
    output = llm(
        full_prompt,
        max_tokens=128,
        stop=["ユーザー:", "システム:", "\n"],
        echo=False
    )
    
    answer = output["choices"][0]["text"].strip()
    
    # 4. 会話履歴を更新
    chat_history.append(f"ユーザー: {user_input}")
    chat_history.append(f"システム: {answer}")
    
    return {"answer": answer}
