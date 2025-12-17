import torch
import torch.nn as nn
import re, json, os



ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def reshape_hidden_states(hidden_states_tuple):
    """
    Input: hidden_states_tuple: (tensor(bsz, length, hidden_size),....) len: layers_num
    Output: list[tensor(length, layers_num, hidden_size)] len: bsz
    """
    # 获取维度信息
    layers_num = len(hidden_states_tuple)
    bsz = hidden_states_tuple[0].size(0)
    length = hidden_states_tuple[0].size(1)
    hidden_size = hidden_states_tuple[0].size(2)
    
    # 将元组中的tensor堆叠成一个tensor
    # 从(layers_num, bsz, length, hidden_size)转换为(bsz, length, layers_num, hidden_size)
    stacked = torch.stack(hidden_states_tuple, dim=0)
    reshaped = stacked.permute(1, 2, 0, 3)
    
    # 分割成bsz个tensor
    return [reshaped[i] for i in range(bsz)]

class Node:
    def __init__(self, ids, parent = None, score = None, num_layers = 5, depth = 0):
        """
        ids: tensor(1, length) - 当前路径的token ids
        parent: Path - 父节点
        score: float - 当前路径的评分
        depth: int - 当前节点在树中的深度
        """
        self.ids = ids
        self.parent = parent
        self.score = score
        self.original_score = None
        self.depth = depth
        self.children = {}
        self.childs = []
        # for i in range(num_layers):
        #     self.children[i] = []
        # self.length = ids.size(1)
        self.last_sample_step = 0

    @property
    def length(self):
        return self.ids.size(1)

    def add_child(self, child):
        """添加子节点"""
        self.childs.append(child)
        
    def get_full_path(self):
        """获取从根节点到当前节点的完整路径"""
        if self.parent is None:
            return [self]
        return self.parent.get_full_path() + [self]
    
    def get_sequence_score(self):
        """获取从根节点到当前节点的平均分数"""
        if self.parent is None:
            return [self.score] if self.score is not None else [0]        
        scores = [p.score for p in self.get_full_path() if p.score is not None]
        # return sum(scores) / len(scores) if scores else 0
        return scores
    
    @property
    def is_leaf(self):
        """判断是否为叶子节点"""
        return len(self.childs) == 0
    
    def get_leaves(self):
        """获取以当前节点为根的所有叶子节点"""
        if self.is_leaf:
            return [self]
        
        leaves = []
        for child in self.childs:
            leaves.extend(child.get_leaves())
        return leaves

def generate_with_tree(
    model, 
    tokenizer, 
    inp, 
    scorer_fn,
    top_k = 10,
    top_n = 2,
    node_len = [5,15,15],
    max_layers = 8,
    unbreakable_tokens: list = []
):  
    
    def score_path(node):
        # 对路径进行打分
        extended_hidden_states = model(node, output_hidden_states = True).hidden_states[1: ]
        torch.cuda.empty_cache()
        extended_hidden_states  = extended_hidden_states
        extended_hidden_states = reshape_hidden_states(extended_hidden_states)  # list[tensor(length, layers_num, hidden_size)] len: bsz
        # extended_hidden_states = extended_hidden_states.to(torch.float32)
        extended_hidden_states = torch.stack(extended_hidden_states, dim = 0).to(torch.float32).to("cpu")
        torch.cuda.empty_cache()
        prob = scorer_fn(extended_hidden_states)
        return prob
        
    def extend_path(batch_ids, steps_to_extend):
        # batch_ids = node.ids
        extend_batch_ids = model.generate(batch_ids, do_sample = False, max_new_tokens = steps_to_extend-1, pad_token_id = tokenizer.eos_token_id) # tensor(bsz, length)
        return extend_batch_ids

    def score_process(score, layers_range: list = [i for i in range(17, 28)], use_last_token = True):  #score: tensor(bsz, length, num_layers)
        # 只保留指定layer的分数
        selected_scores = score[:, :, layers_range]  # (bsz, length, len(layer_range))
        
        # 计算每个样本的得分
        if use_last_token:
            # 使用最后一个token的分数, 在指定层上取平均
            sample_scores = selected_scores[:, -1, :].mean(dim = -1)  # (bsz,)
        else:
            # 使用所有token的平均分数
            sample_scores = selected_scores.mean(dim = [1, 2])  # (bsz,)
        return sample_scores 
    
    def expand_node(parent_node, top_n = 2, node_len = 10, unbreakable_tokens = []):
        """扩展单个节点"""
        # 获取当前节点的预测
        outputs = model(parent_node.ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_probs = nn.functional.softmax(next_token_logits, dim = -1)
        top1_id = torch.argmax(next_probs, dim = -1).unsqueeze(0)

        # 如果下一个 token 是运算符号, 则直接添加到路径上, 不进行分叉
        if top1_id.item() in unbreakable_tokens:
            print("Meet unbreakable token")
            extend_ids = torch.cat((parent_node.ids, top1_id), dim = 1)
            outputs = model(extend_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_probs = nn.functional.softmax(next_token_logits, dim = -1)
        topk_probs, topk_ids = torch.topk(next_probs, top_k, dim = -1)

        # 为每个top_k创建子节点
        new_ids = []
        for i in range(top_k):
            next_id = topk_ids[0][i].unsqueeze(0).unsqueeze(0)
            new_id = torch.cat((parent_node.ids, next_id), dim = 1)
            new_ids.append(new_id)
            # 创建子节点
            # child = Node(ids = new_ids, parent = node, depth = node.depth + 1)
            
        # 批量扩展和
        childrens = []
        batch_ids = torch.cat(new_ids, dim = 0)
        extended_ids = extend_path(batch_ids, node_len) # tensor(bsz, length)
        for i in range(len(extended_ids)):
            child = Node(ids = extended_ids[i].unsqueeze(0), parent = parent_node, depth = parent_node.depth + 1)
            childrens.append(child)

        return extended_ids
    
    def pick_top_n(extended_ids, parent_node, top_n = 2):
        children = []
        # 计算得分
        score = score_path(extended_ids) # renturn (batch_size, token_length, num_layers)
        # print(f"score: {score[1]}")
        score = score.cpu()
        torch.cuda.empty_cache()
        filtered_score = score_process(score)  # return (batch_size,)
        top_n_idx = torch.argsort(filtered_score, descending=True)[ :top_n]
        for i in range(len(extended_ids)):
            child = Node(ids = extended_ids[i].unsqueeze(0), parent = parent_node, depth = parent_node.depth + 1)
             # 保存原始分数
            child.score =  filtered_score[i].item()
            child.original_score = score[i] 
            children.append(child)
            
        # 按分数排序并只保留前n个最好的子节点
        children.sort(key = lambda x: x.score, reverse = True)
        # node.children = children[ :top_n]
        return children[ :top_n], top_n_idx, filtered_score 
    
    input_ids = inp["input_ids"].to(model.device)
    # 创建根节点
    root = Node(input_ids, parent = None, depth = 1, num_layers = max_layers)
    root.text = tokenizer.decode(root.ids[0], skip_special_tokens = True)
    # print(f"Root length: {root.length}")
    root.score = score_process(score_path(input_ids))[0].to("cpu").item()
    # 扩展树
    current_layer_nodes = [root]
    stop_layers = False
    for layer in range(max_layers):
        print(f"Expanding layer {layer + 1}\n")
        root.children[layer] = []
        next_layer_nodes = []

        for j, current_node in enumerate(current_layer_nodes):
            print(f"Expanding node {j + 1} at layer {layer + 1}\n")
            children_extendable = []
            extended_ids = expand_node(current_node, top_n = top_n[layer], node_len = node_len[layer], unbreakable_tokens = unbreakable_tokens)
            top_n_childrens, top_n_idx, filtered_score  = pick_top_n(extended_ids, current_node, top_n = top_n[layer])
            print(f"Filtered score: {filtered_score}")
            print(f"Top {top_n[layer]} nodes selected for parent {j + 1}\n, their index: {top_n_idx}, score: {filtered_score[top_n_idx]}")
            for i, child in enumerate(top_n_childrens):
                # print(f"Expanding child {i + 1} at layer {layer + 1}")
                child.text = tokenizer.decode(child.ids[0], skip_special_tokens = True)
                child_new_ids = child.ids[0][len(root.ids[0]): ]
                root.children[layer].append(child)

                # children_extendable.append(child)
                # current_node.add_child(child)
                if not ((22478 in child_new_ids) or (24994 in child_new_ids) or (2 in child_new_ids)):
                # if not (2 in child_new_ids):
                    children_extendable.append(child)
                    current_node.add_child(child)
                # if len(child_new_ids) < node_len[layer]:
                #    print("Meet short child")
                #    stop_layers = True
            next_layer_nodes.extend(children_extendable)           
        current_layer_nodes = next_layer_nodes
        if stop_layers:
            break
    return root, extended_ids 


import re


def answer_cleansing(args, pred):

    # print("pred_before : " + pred)
    
    if args.current_method in ("search","cot_decoding", "greedy", "zs_cot_prompt", "sc", "sc_zscot"):
        preds = pred.split(args.direct_answer_trigger)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq","MAWPS", "MAWPS", "aime"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, pred is 999999.
    if len(pred) == 0:
        pred = "999999"
    else:
        if args.current_method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.current_method in ("search","cot_decoding", "greedy", "zs_cot_prompt", "sc", "sc_zscot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    # print("pred_after : " + pred)
    
    return pred

def Judge_path(path:str, model, tokenizer, gt, config):
    chunked_response, length_for_search = extract_first_qa_round(path, tokenizer)
    trigger = config.direct_answer_trigger
    prompt_find_answer = chunked_response + f". \n{trigger}"
    input_ids = tokenizer.encode(prompt_find_answer, return_tensors = "pt", add_special_tokens = True).to(model.device)
    out = model.generate(input_ids, max_new_tokens = 10, do_sample = False, pad_token_id = tokenizer.eos_token_id)
    out = tokenizer.decode(out[0], skip_special_tokens = True)
    pred = answer_cleansing(config, out)

    return gt, pred, out


def Judge_path2(path:str, model, tokenizer, gt, config):
    chunked_response, length_for_search = extract_first_qa_round(path, tokenizer)
    trigger = config.direct_answer_trigger
    prompt_find_answer = chunked_response + f". \n{trigger}"
    if config.dataset in ("gsm8k", "aime"):
        input_ids = tokenizer.encode(prompt_find_answer, return_tensors = "pt", add_special_tokens = True).to(model.device)
        out = model.generate(input_ids, max_new_tokens = 10, do_sample = False, pad_token_id = tokenizer.eos_token_id)
        out = tokenizer.decode(out[0], skip_special_tokens = True)
        answer = extract_number(out, prefix = trigger)

    elif config.dataset in ("aqua", "commonsensqa"):
        input_ids = tokenizer.encode(prompt_find_answer, return_tensors = "pt", add_special_tokens = True).to(model.device)
        out = model.generate(input_ids, max_new_tokens = 5, do_sample = False, pad_token_id = tokenizer.eos_token_id)
        out = tokenizer.decode(out[0], skip_special_tokens = True)
        preds = out.split(trigger)
        pred = preds[-1]
        answer = re.findall(r'A|B|C|D|E', pred)[0]

    elif config.dataset in ("addsub", "multiarith", "svamp", "singleeq"):
        input_ids = tokenizer.encode(prompt_find_answer, return_tensors = "pt", add_special_tokens = True).to(model.device)
        out = model.generate(input_ids, max_new_tokens = 10, do_sample = False, pad_token_id = tokenizer.eos_token_id)
        pred = tokenizer.decode(out[0], skip_special_tokens = True)
        preds = pred.split(trigger)
        pred = preds[-1]
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)][0]
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
        answer = pred
    return gt, answer, out

def extract_number(text: str, prefix: str) -> str:
    result = re.search(f'{prefix}\s*[^\d]*([0-9,]+)', text)
    if result:
        # 移除逗号并转换为整数
        return result.group(1).replace(',', '')
    else:
        print(f"Failed to extract number, return 9999") 
        return 9999

from pydantic import BaseModel
from openai import OpenAI
# Set your OpenAI API key as an environment variable before running:
# export OPENAI_API_KEY="your-api-key-here"
# or set it in your code: os.environ["OPENAI_API_KEY"] = "your-api-key"
client = OpenAI()
def extract_answer_GPT(text):
    class Result(BaseModel):
        answer: int | float 
        
    Prompt = """I will provide you with a question and its solution text. Your task is to:
            Understand what the question is asking for
            Follow the reasoning steps provided in the solution
            Based on the question requirement and the solution's reasoning, determine the appropriate final numerical answer
            For example:
            Example 1:
            Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
            Answer: [reasoning steps]
            So the answer is 16-7=9 eggs.
            Output: 18 (because the question asks for dollars, and 9 eggs × $2 = $18)
            Example 2:
            Question: Jeff owns a catering company. During a recent event, he sent 8 dozen glasses and 4 dozen plates for the party. When they were returned, 10 glasses were broken as well as 6 plates. How many glasses and plates does Jeff have now?
            Answer: [reasoning steps]
            So the answer is 84 glasses and 44 plates.
            Output: 128 (because the question asks for total items, so we add 84 glasses + 44 plates)
            Your task is to provide the final numerical answer that correctly addresses what the question is asking for, based on the reasoning provided in the solution."""

    completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": Prompt},
        {"role": "user", "content": text},
    ],
    response_format=Result,
    )

    ans = completion.choices[0].message.parsed
    return ans.answer



def extract_first_qa_round(text, tokenizer, is_zs_cot = False):
    if is_zs_cot == False:
        qa_round_pattern = re.compile(r'Question: .+?Answer:\n?.+?(?=Question(?:\s*\d+)?: |\Z)', re.DOTALL)
    elif is_zs_cot == True:
        qa_round_pattern = re.compile(r'Question: .+?\nAnswer: Let\'s think step by step.\n?.+?(?=\nQuestion: |\Z)', re.DOTALL)
    
    qa_rounds = qa_round_pattern.findall(text)
    first_qa = qa_rounds[0] if qa_rounds else text
    length = len(tokenizer.encode(first_qa, add_special_tokens=False))
    
    return first_qa, length

def find_symbol_and_range(text, phrase = "So the answer(digit number) is"):
    # Find the index of the phrase
    start_index = text.find(phrase)
    if start_index == -1:
        return None, (-1, -1)

    # Calculate the start index of the symbol by adding the length of the phrase
    symbol_start_index = start_index + len(phrase)
    
    # Use regex to find the first sequence of digits after the phrase
    match = re.search(r'\b\d+\b', text[symbol_start_index:])
    if not match:
        return None, (-1, -1)

    # Extract the symbol
    symbol = match.group(0)
    
    # Find the range of the symbol in the original string before the phrase
    symbol_index_start = text[:start_index].rfind(symbol)
    symbol_index_end = symbol_index_start + len(symbol) - 1  # End index is start index + length of the symbol - 1

    # Return the symbol and its range (start and end indices)
    return symbol, (symbol_index_start, symbol_index_end)



class ExampleConfig:
    def __init__(
        self,
        top_k: int = 10,
        top_n: int = 1,
        node_length: list = [5] * 3 + [30] * 10,
        init_layers: int = 1,
        special_tokens: list = [],
        description: str = ""
    ):
        self.top_k = top_k
        self.node_length = node_length if node_length else [5] * 3 + [30] * 10
        self.init_layers = init_layers
        self.special_tokens = [
            "+", "-", "*", "/", "(", ")", "[", "]", "{", "}", 
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
        ] if special_tokens is None else special_tokens
        self.description = description
        self.top_n = [top_n] * min(self.init_layers, self.max_layers) + \
               [1] * max(0, self.max_layers - self.init_layers)
    
    @property
    def max_layers(self):
        return len(self.node_length) - 1
    
    def to_dict(self):
        """将配置转换为字典形式，便于保存"""
        return {
            'top_k': self.top_k,
            'node_length': self.node_length,
            'init_layers': self.init_layers,
            'special_tokens': self.special_tokens,
            'description': self.description
        }

def process_example(dataset, example_id, config: ExampleConfig, tokenizer, model, score_fn):
    """使用指定配置处理单个样本"""
    # Get question and answer
    question = dataset["test"][example_id]['question']
    answer = dataset["test"][example_id]['answer']
    digit_answer = extract_answer_from_output(answer)
    
    # Prepare prompt
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Get non-breakable tokens
    non_breakable_tokens = [
        tokenizer.encode(token, add_special_tokens=False)[0] 
        for token in config.special_tokens
    ]
    
    # Generate response
    with torch.no_grad():
        root = generate_with_tree(
            model=model,
            tokenizer=tokenizer,
            inp=inputs,
            top_k=config.top_k,
            top_n=config.top_n,
            node_len=config.node_length,
            scorer_fn=score_fn,
            max_layers=config.max_layers,
            unbreakable_tokens=non_breakable_tokens
        )
    
    # Get best result
    result = next(
        ((key, value) for key, value in reversed(root.children.items()) if value), 
        None
    )
    leaves = result[1]
    best_node = max(leaves, key=lambda x: x.score)
    
    # Decode and evaluate
    text = tokenizer.decode(best_node.ids[0], skip_special_tokens=True)
    ground_truth, prediction,_ = Judge_path(text, model, tokenizer, digit_answer)
    
    return {
        'config': config.to_dict(),
        'text': text,
        'ground_truth': ground_truth,
        'prediction': prediction,
        'answer': digit_answer,
        'question': question
    }, root


import numpy as np
from scipy import stats

def calculate_fluctuation_metrics(data):
    """
    计算给定序列的各种浮动指标。
    
    参数:
        data (list or array): 数字序列
        
    返回:
        dict: 包含多个浮动指标的字典
    """
    data = np.array(data)
    metrics = {}

    # 首节点分数
    metrics = {"First Score": data[0]}

    # 尾节点分数
    metrics["Last Score"] = data[-1]

    # 极差
    metrics["Range"] = np.max(data) - np.min(data)
    
    # 标准差
    metrics["Standard Deviation"] = np.std(data)
    
    # 方差
    metrics["Variance"] = np.var(data)
    
    # 均值
    mean_value = np.mean(data)
    metrics["Mean"] = mean_value

    #总和
    metrics["Sum"] = np.sum(data)
    
    # 变异系数
    metrics["Coefficient of Variation"] = (
        metrics["Standard Deviation"] / mean_value if mean_value != 0 else float('inf')
    )
    
    # 平均绝对偏差
    metrics["Mean Absolute Deviation"] = np.mean(np.abs(data - mean_value))
    
    # 最大单步变动
    step_changes = np.abs(np.diff(data))
    metrics["Max Step Change"] = np.max(step_changes) if len(step_changes) > 0 else 0
    
    # 均方根浮动
    metrics["RMS Fluctuation"] = np.sqrt(np.mean((data - mean_value) ** 2))
    
    # 自相关 (滞后为1)
    def autocorrelation(data, lag=1):
        if len(data) <= lag + 1:
            return 0
        y1 = np.array(data[:-lag])
        y2 = np.array(data[lag:])
        
        if np.all(y1 == y1[0]) or np.all(y2 == y2[0]):
            return 1 if y1[0] == y2[0] else 0
            
        try:
            corr = np.corrcoef(y1, y2)[0, 1]
            return corr if not np.isnan(corr) else 0
        except:
            return 0
    
    metrics["Autocorrelation (Lag 1)"] = autocorrelation(data)
    
    # 添加趋势指标
    # 计算相邻差值平均值
    diffs = np.diff(data)
    metrics["Average Difference"] = np.mean(diffs)
    
    # 计算上升和下降的比例
    increases = np.sum(diffs > 0)
    decreases = np.sum(diffs < 0)
    total_changes = len(diffs)
    metrics["Increase Ratio"] = increases / total_changes if total_changes > 0 else 0
    metrics["Decrease Ratio"] = decreases / total_changes if total_changes > 0 else 0
    
    # Kendall's Tau
    time_seq = np.arange(len(data))
    kendall_tau, _ = stats.kendalltau(time_seq, data)
    metrics["Kendall Tau"] = kendall_tau if not np.isnan(kendall_tau) else 0
    
    return metrics

