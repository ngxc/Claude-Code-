import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import requests
import numpy as np
import torch
import hashlib
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from pynput import keyboard

# =========================
# 1. 配置参数
# =========================
API_KEY = ""
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

SHORT_TERM_TURNS = 5
MID_BLOCK_TURNS = 15
CHUNK_SIZE = 4
MAX_CHILDREN = 5
LONG_CHUNK_SIZE = CHUNK_SIZE * 3

RAG_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
TOP_K = 3
OUTPUT_FILE = "memory_tree_dialogue.json"

# =========================
# 2. 全局状态
# =========================
current_all_pairs = []
current_memory_obj = {
    "short_term_history": [],
    "mid_term_tree": None,
    "long_term_tree": None
}

block_cache = {}
global_node_counter = 0

# =========================
# 3. 初始化模型
# =========================
print("[系统] 正在初始化向量模型 (MiniLM)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
rag_model = SentenceTransformer(RAG_MODEL_NAME).to(device)


def call_api(messages, max_tokens=1000):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        r = requests.post(BASE_URL, json=payload, headers=headers, timeout=60)
        print("\n[DEBUG] status_code:", r.status_code)
        print("[DEBUG] response:", r.text[:500])
        data = r.json()
        if "choices" not in data:
            print("[ERROR] 返回异常:", data)
            return None
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("[ERROR] 请求异常:", str(e))
        return None

# =========================
# 4. hash函数
# =========================
def hash_pairs(pairs):
    text = json.dumps(pairs, ensure_ascii=False)
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# =========================
# 5. Node
# =========================
class Node:
    def __init__(self, summary="", children=None, messages=None, block_id=None):
        self.summary = summary
        self.children = children or []
        self.messages = messages or []
        self.block_id = block_id
        self.embedding = rag_model.encode(summary) if summary else None

    def to_dict(self):
        return {
            "block_id": self.block_id,
            "summary": self.summary,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "children": [c.to_dict() for c in self.children],
            "messages": self.messages
        }

# =========================
# 6. 总结函数
# =========================
def summarize_core(text, instruction):
    system_prompt = (
        "你是一个对话分析专家。\n"
        "1. 记录用户核心意图\n"
        "2. 记录话题转向\n"
        "3. 简洁但完整"
    )
    return call_api([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}:\n{text}"}
    ])

# =========================
# 7. 树构建
# =========================
def build_tree_recursive(pairs, level=1, current_chunk_size=CHUNK_SIZE):
    global global_node_counter
    if not pairs:
        return None

    # 叶子节点
    if len(pairs) <= CHUNK_SIZE:
        block_id = hash_pairs(pairs)
        if block_id in block_cache:
            print(f"[DEBUG] 复用叶子: {block_id[:6]}")
            return block_cache[block_id]

        global_node_counter += 1
        node_index = global_node_counter
        texts = [f"{m['role']}: {m['content']}" for p in pairs for m in p]
        content_text = "\n".join(texts)
        print(f"[DEBUG] 新建叶子: {block_id[:6]}")
        sum_text = summarize_core(content_text, f"块{node_index}总结")
        node = Node(
            summary=sum_text or "无摘要",
            messages=pairs,
            block_id=block_id
        )
        block_cache[block_id] = node
        return node

    # 父节点
    num_chunks = min(MAX_CHILDREN, (len(pairs) + current_chunk_size - 1) // current_chunk_size)
    chunk_size_for_split = (len(pairs) + num_chunks - 1) // num_chunks
    chunks = [pairs[i:i + chunk_size_for_split] for i in range(0, len(pairs), chunk_size_for_split)]

    children = []
    for chunk in chunks:
        child_node = build_tree_recursive(chunk, level + 1, current_chunk_size)
        if child_node:
            children.append(child_node)

    combined = "\n".join([c.summary for c in children])
    block_id = hash_text(combined)
    if block_id in block_cache:
        print(f"[DEBUG] 复用父节点: {block_id[:6]}")
        return block_cache[block_id]

    global_node_counter += 1
    node_index = global_node_counter
    print(f"[DEBUG] 新建父节点 L{level}-{node_index}")
    node_sum = summarize_core(combined, f"阶段{level}-{node_index}总结")
    node = Node(
        summary=node_sum or "无摘要",
        children=children,
        block_id=block_id
    )
    block_cache[block_id] = node
    return node

# =========================
# 8. RAG 检索
# =========================
def search_tree_recursive(node, query_emb, results):
    if not node:
        return
    if node.embedding is not None:
        dist = cdist([query_emb], [node.embedding], metric="cosine")[0][0]
        results.append((node, dist))
    for child in node.children:
        search_tree_recursive(child, query_emb, results)

# 获取叶子 messages
def get_leaf_messages(node):
    if not node.children:
        return node.messages
    msgs = []
    for c in node.children:
        msgs.extend(get_leaf_messages(c))
    return msgs

# =========================
# 9. 保存
# =========================
def save_state():
    data = {
        "all_pairs_raw": current_all_pairs,
        "mid_term_tree": current_memory_obj["mid_term_tree"].to_dict() if current_memory_obj["mid_term_tree"] else None,
        "long_term_tree": current_memory_obj["long_term_tree"].to_dict() if current_memory_obj["long_term_tree"] else None
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================
# 10. 更新结构
# =========================
def update_memory_structure():
    global current_memory_obj
    n = len(current_all_pairs)
    short_term = current_all_pairs[-SHORT_TERM_TURNS:] if n >= SHORT_TERM_TURNS else current_all_pairs
    mid_start = max(0, n - SHORT_TERM_TURNS - MID_BLOCK_TURNS)
    mid_end = n - SHORT_TERM_TURNS
    print(f"\n[DEBUG] 总轮数: {n}")
    print(f"[DEBUG] short={len(short_term)} mid={mid_end-mid_start} long={mid_start}")
    current_memory_obj["short_term_history"] = short_term
    mid_block = current_all_pairs[mid_start:mid_end]
    current_memory_obj["mid_term_tree"] = build_tree_recursive(mid_block, current_chunk_size=CHUNK_SIZE)
    long_block = current_all_pairs[:mid_start]
    current_memory_obj["long_term_tree"] = build_tree_recursive(long_block, current_chunk_size=LONG_CHUNK_SIZE)

# =========================
# 11. Esc监听
# =========================
def on_press(key):
    if key == keyboard.Key.esc:
        print("\n[退出] 保存中...")
        save_state()
        os._exit(0)

keyboard.Listener(on_press=on_press).start()

# =========================
# 12. 主循环
# =========================
def main():
    global current_all_pairs
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                current_all_pairs = data.get("all_pairs_raw", [])
                print(f"[加载] 历史 {len(current_all_pairs)} 轮")
        except Exception:
            print("[警告] JSON损坏，跳过加载")

    update_memory_structure()

    while True:
        user_input = input("\n用户: ").strip()
        if not user_input:
            continue

        query_emb = rag_model.encode(user_input)
        results = []
        search_tree_recursive(current_memory_obj["mid_term_tree"], query_emb, results)
        search_tree_recursive(current_memory_obj["long_term_tree"], query_emb, results)

        results.sort(key=lambda x: x[1])
        retrieved_pairs = []
        for node, dist in results[:TOP_K]:
            retrieved_pairs.extend(get_leaf_messages(node))

        messages = [{"role": "system", "content": "参考历史回答"}]

        if retrieved_pairs:
            messages.append({"role": "system", "content": "以下是相关历史对话"})
            for pair in retrieved_pairs:
                messages.extend(pair)

        for pair in current_memory_obj["short_term_history"]:
            messages.extend(pair)

        messages.append({"role": "user", "content": user_input})

        answer = call_api(messages)
        if answer:
            print("\nAI:", answer)
            pair = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer}
            ]
            current_all_pairs.append(pair)
            update_memory_structure()
            save_state()
        else:
            print(answer)
            print("API失败")

if __name__ == "__main__":
    main()