from cs336_basics.tokenizer import tokenizer


def print_vocab(vocab: dict[int, bytes], start = None, end = None, last_n = 50):
    """
    按 ID 升序打印词表内容
    - start / end: 打印指定 ID 范围（例如 31855 到 31894）
    - last_n: 如果不指定范围，默认打印最新 merge 出来的最后 N 个 Token
    """
    sorted_items = sorted(vocab.items(), key=lambda x: x[0])
    
    if start is not None or end is not None:
        start = start or 0
        end = end or (sorted_items[-1][0] + 1)
        items_to_print = [(k, v) for k, v in sorted_items if start <= k < end]
    else:
        items_to_print = sorted_items[-last_n:]

    for token_id, token_bytes in items_to_print:
        # 解码为 UTF-8 字符串
        token_str = token_bytes.decode("utf-8", errors="replace")
        # 将换行/制表等不可见字符转义，避免控制台排版乱掉
        token_str_visible = token_str.replace("\n", "\\n").replace("\t", "\\t")
        
        # 格式化对齐打印
        print(f"{token_id:<6} {token_str_visible}")

def load_sample_docs(file_path: str, num_docs: int = 10) -> list[str]:
    """
    从数据集中流式读取，按 <|endoftext|> 提取前 num_docs 篇完整文档
    无需一次性加载整份大文件
    """
    docs = []
    current_doc = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "<|endoftext|>" in line:
                parts = line.split("<|endoftext|>")
                # 拼接当前文档的剩余部分
                current_doc.append(parts[0])
                doc_str = "".join(current_doc).strip()
                if doc_str:
                    docs.append(doc_str)
                    if len(docs) >= num_docs:
                        break
                
                # 如果同一行有多个 <|endoftext|>，继续切分
                for p in parts[1:-1]:
                    if p.strip():
                        docs.append(p.strip())
                        if len(docs) >= num_docs:
                            break
                if len(docs) >= num_docs:
                    break
                current_doc = [parts[-1]]
            else:
                current_doc.append(line)
                
    return docs

def compute_compression_ratio(tok: tokenizer, docs: list[str]) -> float:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        # 计算原始 UTF-8 字节数与 encode 后的 Token 数
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tok.encode(doc))
    return total_bytes / total_tokens

import numpy as np

def tokenize_and_save_to_bin(
    tokenizer: tokenizer, 
    input_text_path: str, 
    output_bin_path: str, 
    chunk_size: int = 100_000
):
    """
    使用 encode_iterable 流式分词并写入二进制文件，内存占用近乎为 0
    """
    print(f"Tokenizing {input_text_path} -> {output_bin_path}...")
    
    with open(input_text_path, "r", encoding="utf-8") as f_in, open(output_bin_path, "wb") as f_out:
        token_stream = tokenizer.encode_iterable(f_in)
        
        buffer = []
        total_tokens = 0
        
        for token_id in token_stream:
            buffer.append(token_id)
            if len(buffer) >= chunk_size:
                # 转换为 uint16 数组写入二进制文件 (每个 token 仅占 2 字节)
                np.array(buffer, dtype=np.uint16).tofile(f_out)
                total_tokens += len(buffer)
                buffer = []
                
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_out)
            total_tokens += len(buffer)
            
    print(f"Done! Total tokens written: {total_tokens}")

if __name__ == "__main__":
    # 1. 路径配置（根据你的实际路径调整）
    OWT_VALID_PATH = "data/owt_valid.txt"
    TS_VALID_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

    # 2. 采样 10 篇文档
    print("Sampling 10 documents from datasets...")
    owt_docs = load_sample_docs(OWT_VALID_PATH, num_docs=10)
    ts_docs = load_sample_docs(TS_VALID_PATH, num_docs=10)

    # 3. 加载两个训练好的 Tokenizer
    print("Loading tokenizers...")
    ts_tokenizer = tokenizer.from_files(
        vocab_filepath="output/ts_vocab.json",
        merges_filepath="output/ts_merges.json",
        special_tokens=["<|endoftext|>"]
    )
    
    owt_tokenizer = tokenizer.from_files(
        vocab_filepath="output/owt_vocab.json",
        merges_filepath="output/owt_merges.json",
        special_tokens=["<|endoftext|>"]
    )

    # 打印尾 40 个
    print('\n--- Last 40 in TS vocab---')
    print_vocab(ts_tokenizer.vocab, last_n=40)
    print('\n--- Last 40 in OWT vocab---')
    print_vocab(owt_tokenizer.vocab, last_n=40)

    # 找到最长的 10 个 tokens
    longest_tokens = sorted(owt_tokenizer.vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print("\n--- Longest Tokens in OWT Vocab ---")
    for token_id, token_bytes in longest_tokens:
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"ID: {token_id:<6} | Byte Length: {len(token_bytes):<3} | Content: {repr(token_str)}")
    longest_tokens = sorted(ts_tokenizer.vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print("\n--- Longest Tokens in TS Vocab ---")
    for token_id, token_bytes in longest_tokens:
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"ID: {token_id:<6} | Byte Length: {len(token_bytes):<3} | Content: {repr(token_str)}")

    # 4. 执行问题 (a) 的测试
    cr_ts_on_ts = compute_compression_ratio(ts_tokenizer, ts_docs)
    cr_owt_on_owt = compute_compression_ratio(owt_tokenizer, owt_docs)

    print("\n--- 实验 (a) 结果 ---")
    print(f"TinyStories Tokenizer on TinyStories: {cr_ts_on_ts:.3f} bytes/token")
    print(f"OpenWebText Tokenizer on OpenWebText: {cr_owt_on_owt:.3f} bytes/token")

    # 5. 执行问题 (b) 的交叉测试
    cr_ts_on_owt = compute_compression_ratio(ts_tokenizer, owt_docs)

    print("\n--- 实验 (b) 结果 ---")
    print(f"TinyStories Tokenizer on OpenWebText (交叉测试): {cr_ts_on_owt:.3f} bytes/token")

    import time

    # 准备一段较大文本（例如 10MB 文本样本）
    with open("data/owt_valid.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()

    total_bytes = len(sample_text.encode("utf-8"))

    start_time = time.time()
    tokens = owt_tokenizer.encode(sample_text)
    elapsed_time = time.time() - start_time

    throughput_mb_s = (total_bytes / 1024 / 1024) / elapsed_time
    print(f"Throughput: {throughput_mb_s:.2f} MB/s (即 {throughput_mb_s * 1024:.2f} KB/s)")

    # 估算 825GB 耗时
    pile_size_mb = 825 * 1024
    total_seconds = pile_size_mb / throughput_mb_s
    print(f"Estimated time for 825GB Pile: {total_seconds / 3600 / 24:.2f} days (单核)")

    # 执行全量数据集的序列化
    # tokenize_and_save_to_bin(ts_tokenizer, "data/TinyStoriesV2-GPT4-train.txt", "output/ts_train.bin")
    # tokenize_and_save_to_bin(ts_tokenizer, "data/TinyStoriesV2-GPT4-valid.txt", "output/ts_val.bin")
    tokenize_and_save_to_bin(owt_tokenizer, "data/owt_train.txt", "output/owt_train.bin")
    tokenize_and_save_to_bin(owt_tokenizer, "data/owt_valid.txt", "output/owt_val.bin")