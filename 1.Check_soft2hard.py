from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
import os
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default = "Qwen/Qwen2.5-7B-Instruct", type = str, required=False)

    args = parser.parse_args()

    model_name = args.language_model
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map='auto').to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    soft_embedding = torch.load('1.soft_prompt/soft_length8_epoch9_reverse0_Qwen2.5-7B-Instruct_0.6792.pt', weights_only = True).to(model.device)

    ###################
    # hard_prompt = '[CONTEXT TRUNCATED]\n' * 100
    # hard_prompt_ids = tokenizer(hard_prompt, add_special_tokens=False, return_tensors='pt', max_length = 6, truncation = True)['input_ids'].to(model.device)
    # hard_prompt_embeddings = model.model.embed_tokens(hard_prompt_ids).squeeze(0)  # shape: (seq_len, embedding_size)
    # soft_embedding = hard_prompt_embeddings
    ###################

    # 获取词嵌入矩阵（vocab size x hidden size）
    embedding_matrix = model.get_input_embeddings().weight  # shape: [vocab_size, hidden_size]

    # soft_embedding shape: [soft_prompt_length, hidden_size]
    soft_prompt_vectors = soft_embedding

    if isinstance(soft_prompt_vectors, torch.nn.Parameter):
        soft_prompt_vectors = soft_prompt_vectors.data  # 转为 tensor

    soft_prompt_vectors = soft_prompt_vectors.to(model.device)

    # 归一化后计算余弦相似度
    embedding_matrix_norm = F.normalize(embedding_matrix, dim=1)  # [V, H]
    soft_vectors_norm = F.normalize(soft_prompt_vectors, dim=1)   # [S, H]

    # 相似度矩阵：[soft_len, vocab_size]
    similarity = torch.matmul(soft_vectors_norm, embedding_matrix_norm.T)  # [S, V]

    # 每个 soft token 找到最相近的 hard token 的索引
    nearest_token_ids = torch.argmax(similarity, dim=1)  # [S]

    # 将最近邻 hard token 的 ID 列表解码为完整文本
    pseudo_prompt = tokenizer.decode(nearest_token_ids.tolist(), clean_up_tokenization_spaces=True)

    print("伪文本提示（由 soft prompt 映射回来的）：")
    print(pseudo_prompt)