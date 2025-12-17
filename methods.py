from Utils.tool import Judge_path,extract_answer_GPT, calculate_fluctuation_metrics
from Utils.Node import expand
from search import TreeSearch
# from Cot_decoding.utils
from Cot_decoding.utils import  generate_branching_responses, save_to_json, initialize_model_and_tokenizer
from Cot_decoding.utils_data import setup_data_loader, extract_after
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import os

def find_delta(pred, id_prob: List[Tuple[int, str, float]], tokenizer):
    pred:List[int] = tokenizer.encode(pred, add_special_tokens=False)
    a = id_prob
    b = pred
    # 找到包含 b 中元素的 tuples
    matched_tuples = [t for t in a if t[0] in b]
    
    # 如果没有找到匹配的 tuples，返回 0 或者可以选择抛出异常
    if not matched_tuples:
        return -1
        
    # 计算第三个元素(索引2)的均值
    mean_value = sum(t[2] for t in matched_tuples) / len(matched_tuples)
    return mean_value


def estimate_acc(acc, num, total=1319):
    return (acc /num) * total

def get_leaves(searcher, min_step):
    """Extract leaf nodes from the search tree including last layer nodes
    
    Args:
        searcher: 搜索器实例
        min_step: 最小步数阈值
        
    Returns:
        leaves: 叶子节点列表 (包含最后一层的所有节点)
    """
    current_min_step = min_step
    
    while current_min_step >= 0:  # 设置最小为0,防止无限循环
        leaves = []
        max_step = max(searcher.all_nodes_by_level.keys())  # 获取最后一层的step
        
        for step, nodes in searcher.all_nodes_by_level.items():
            if step >= current_min_step:
                # 如果是最后一层,添加所有节点；否则只添加叶子节点
                if step == max_step:
                    leaves.extend(nodes)
                else:
                    leaves.extend([node for node in nodes if not node.extendable or node.is_leaf])
                
        if leaves:  # 如果找到了节点就返回
            if current_min_step < min_step:
                print(f"Reduced min_step from {min_step} to {current_min_step} to find leaves")
            return leaves
            
        current_min_step -= 1  # 如果没找到,减小步数阈值
        
    # 如果连step=0都没有找到节点,返回根节点
    return [searcher.root]

def greed_decoding(model, tokenizer, data, config, max_new_tokens = 200):
    x, y = data   
    prompt = "Question: " + x[0] + "\n" + "Answer:"
    digit_ans = y[0].strip()

    # digit_ans = extract_answer_from_output(example['answer'])
    inp = tokenizer(prompt, return_tensors = "pt").to(model.device)    
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, stop_strings = "Question:", tokenizer = tokenizer)
    out_text = tokenizer.decode(out[0], skip_special_tokens=True)
    gt, pred, full_text = Judge_path(out_text, model, tokenizer, digit_ans, config)
    return pred

from collections import Counter
def most_frequent_element(lst):
    # 使用 Counter 计算元素频率
    count = Counter(lst)
    # 获取出现次数最多的元素
    most_common_element = count.most_common(1)[0]
    return most_common_element


def self_consistency_decoding(model, tokenizer, data, config, num_sample = 10, max_new_tokens = 200):
        x, y = data   
        prompt = "Question: " + x[0] + "\n" + "Answer:"
        digit_ans = y[0].strip()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample=True, num_return_sequences = num_sample, temperature = 0.7)
        preds = []
        for i in range(num_sample):
            out_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            gt, pred, full_text = Judge_path(out_text, model, tokenizer, digit_ans, config)
            preds.append(float(pred))
        voted_pred = most_frequent_element(preds)[0]
        return voted_pred

def self_consistency_decoding_zscot(model, tokenizer, data, config, num_sample = 10, max_new_tokens = 200):
        x, y = data   
        prompt = "Question: " + x[0] + "\n" + "Answer:" + "Let's think step by step."
        digit_ans = y[0].strip()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens =  max_new_tokens , do_sample=True, num_return_sequences = num_sample, temperature = 0.7)
        preds = []
        for i in range(num_sample):
            out_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            gt, pred, full_text = Judge_path(out_text, model, tokenizer, digit_ans, config)
            preds.append(float(pred))
        voted_pred = most_frequent_element(preds)[0]
        return voted_pred



def zs_cot_prompting(model, tokenizer, data, config, max_new_tokens = 200):
    x, y = data   
    prompt = "Question: " + x[0] + "\n" + "Answer:" + "Let's think step by step."
    digit_ans = y[0].strip()

    # digit_ans = extract_answer_from_output(example['answer'])
    inp = tokenizer(prompt, return_tensors = "pt").to(model.device)    
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, stop_strings = "Question:", tokenizer = tokenizer)
    out_text = tokenizer.decode(out[0], skip_special_tokens=True)
    gt, pred, full_text = Judge_path(out_text, model, tokenizer, digit_ans, config)
    return pred

def tree_search_decoding(data, model, tokenizer, score_fn, config):
    """Process a single example from the dataset"""
    searcher = TreeSearch(
        model=model, tokenizer=tokenizer, expansion_func=expand,
        score_fun=score_fn, num_samples=config.num_samples, 
        temperature=config.temperature,
        score_layers = config.score_layers
    )

    x, y = data   
    prompt = "Question: " + x[0] + "\n" + "Answer:"
    digit_ans = y[0].strip()
    searcher.search(prompt, num_steps = config.MAX_LAYERS, sample_steps = config.sample_steps, 
                    max_length_node = config.NODE_LENGTH, top_k = config.top_k,
                    flatten = False, bsz = config.batch_size, chunk = config.chunk, rep = config.rep)
    
    leaves = get_leaves(searcher, config.min_step)
    
    for leaf in leaves:
        leaf.metrics = calculate_fluctuation_metrics(leaf.get_sequence_score())
        gt, pred, full_text = Judge_path(leaf.text, model, tokenizer, digit_ans, config)
        if pred == "":  # 如果没有找到答案,则使用原始输出
            pred = "999999"
        leaf.pred = pred
        leaf.full_text = full_text
        if config.use_gpt:
            leaf.pred_gpt = extract_answer_GPT(leaf.text) if config.use_gpt else None
    return leaves, searcher

def cot_decoding(model, tokenizer, data, config, max_new_tokens = 220):
    x, y = data   
    question = x[0]
    prompt = "Question: " + x[0] + "\n" + "Answer:"
    digit_ans = y[0].strip()
    gt = digit_ans
    responses_texts, id_probs = generate_branching_responses(model, tokenizer, prompt, num_branches = 10, max_length = max_new_tokens)
    
    candidates = []
    final_out = []
    for i, response in enumerate(responses_texts):
        gt, pred, full_text = Judge_path(response, model, tokenizer, gt, config)
        delta = find_delta(pred, id_probs[i], tokenizer)
        # pred = np.array(pred)
        candidates.append((pred, delta))
        dict_real, dict_fake = {}, {}
        dict_real['question'], dict_fake['question'] = question, question
        dict_real['label'], dict_fake['label'] = 1, 0
        ans_extracted = extract_after(prompt, response)
        if_true = np.array(gt) == np.array(pred)
        if if_true == True and pred in ans_extracted:
            dict_real['answer'] = ans_extracted 
            dict_real['qa'] = response
            dict_real['gt'] = gt
            # dict_real['id_prob'] = id_probs[i]
            final_out.append(dict_real)
            # print("True")
        else:
            dict_fake['answer'] = ans_extracted
            dict_fake['qa'] = response
            dict_fake['gt'] = gt
            # dict_fake['id_prob'] = id_probs[i]
            final_out.append(dict_fake)
            # print("False")
        # os.makedirs(config.pari_data_save_path, exist_ok=True)
    
    best_pred = max([(t[0], t[1]) for t in candidates if t[1] is not None], key=lambda x: x[1])[0]
    best_pred = np.array(best_pred)
    os.makedirs(os.path.dirname(config.pari_data_save_path), exist_ok=True)
    save_to_json(config.pari_data_save_path, final_out) 
    return best_pred