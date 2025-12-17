import re, json, os
from typing import List, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def initialize_model_and_tokenizer(config):
    """Initialize the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.MODEL_CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, cache_dir=config.MODEL_CACHE_DIR, 
        torch_dtype=torch.bfloat16, device_map=config.device
    )
    return model, tokenizer


def answer_cleansing(args, pred):

    # print("pred_before : " + pred)
    
    if args.method in ("few_shot", "few_shot_cot","zero_shot", "zero_shot_cot"):
        preds = pred.split(args.direct_answer_trigger)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
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

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
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


def Judge_path(text:str, model, tokenizer, gt, config):
    # chunked_response, length_for_search = extract_first_qa_round(text, tokenizer)
    trigger = config.direct_answer_trigger
    prompt_find_answer = text + f". \n{trigger}"
    input_ids = tokenizer.encode(prompt_find_answer, return_tensors = "pt", add_special_tokens = True).to(model.device)
    out = model.generate(input_ids, max_new_tokens = 10, do_sample = False, pad_token_id = tokenizer.eos_token_id)
    out = tokenizer.decode(out[0], skip_special_tokens = True)
    pred = answer_cleansing(config, out)

    return gt, pred, out


# Helper function to save data to JSON
def save_to_json(file_path, data):
    if os.path.exists(file_path):
        # Load existing data and append
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # Append new data
    existing_data.extend(data)

    # Save back to JSON
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


# Get our initial top k tokens
def get_topk_tokens(model, inputs, num_branches=10):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
    # Generate logits for the next token after the prompt 
    with torch.no_grad():
        outputs = model(inputs["input_ids"], return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
    
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get the top k tokens and their probabilities
    topk_values, topk_indicies = torch.topk(probabilities, num_branches)
    argmax_id = torch.argmax(probabilities, dim = -1).unsqueeze(0)

    return topk_values, topk_indicies,  argmax_id 

def get_topk_tokens2(model, inputs, num_branches=10):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
    # Generate logits for the next token after the prompt 
    with torch.no_grad():
        outputs = model(input_ids= inputs["input_ids"], return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
    
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get the top k tokens and their probabilities
    topk_values, topk_indicies = torch.topk(probabilities, num_branches)# shape: (1, num_branches), (1, num_branches)
    argmax_id = torch.argmax(probabilities, dim = -1).unsqueeze(0)

    return topk_values, topk_indicies


# Generate a full response from the model and log the difference in probabilities between the top two tokens
def generate_response(model, tokenizer, inputs, max_length = 500):

    # Create variables to store our response and each token's probabilities
    id_delta = []
    
    # Loop through the max length of the response
    for i in range(max_length-1):

        topk_values, topk_indices,  argmax_id  = get_topk_tokens(model, inputs, num_branches = 2)

        # Get the difference in probabilities between the top two tokens
        prob_diff = topk_values[:, 0] - topk_values[:, 1]
        triple_point = (argmax_id.item(), tokenizer.convert_ids_to_tokens(argmax_id.item()), prob_diff.item())
        id_delta.append(triple_point)

        # Stop if this token is the end of sequence token
        invalid_tokens = tokenizer.convert_tokens_to_ids(['▁Question','Question'])
        invalid_tokens.append(tokenizer.eos_token_id)
        if  argmax_id in invalid_tokens:
            print("End of sequence token found")
            break

        # Add the token to the input for the next iteration
        inputs['input_ids'] = torch.cat([inputs['input_ids'], argmax_id], dim = 1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((1, 1)).to(model.device)], dim=1)
    out_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return out_text, id_delta

# Generate all branching responses
def generate_branching_responses(model, tokenizer, prompt, num_branches = 10, max_length = 500):

    # First we tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get our initial top k tokens
    topk_values, topk_indices = get_topk_tokens2(model, inputs, num_branches)

    # Create a list to store our responses and each token's probabilities
    responses = []
    id_deltas = []
    for k in tqdm(range(num_branches)):

        # Add the kth most likely token to this new branch
        new_inputs = inputs.copy()
        new_inputs['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, k].unsqueeze(-1)], dim=1)
        new_inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((1, 1)).to(model.device)], dim=1)
        # Generate a response and log the difference in probabilities between the top two tokens
        text, id_delta = generate_response(model, tokenizer, new_inputs, max_length)

        responses.append(text)
        id_deltas.append(id_delta)

    id_delta: List[List[Tuple[int, float]]]
    responses: List[str]
    return responses, id_deltas
    


def find_symbol_and_range(text, phrase = "So the answer is"):
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

def extract_first_qa_round(text, tokenizer, is_zs_cot = False):
    # Define a regular expression pattern to match a QA round
    # This pattern assumes that the 'Q:' prefix is only used for questions
    # and that the answer can span multiple lines, possibly starting with a newline character.
    if is_zs_cot == False:
        qa_round_pattern = re.compile(r'Question: .+?\nAnswer:\n?.+?(?=\nQuestion: |\Z)', re.DOTALL)
    elif is_zs_cot == True:
            qa_round_pattern = re.compile(r'Question: .+?\nAnswer: Let\'s think step by step.\n?.+?(?=\nQuestion: |\Z)', re.DOTALL)
    
    # Find all matches of the QA pattern in the text
    qa_rounds = qa_round_pattern.findall(text)
    
    # If matches found, return first QA round, otherwise return original text
    first_qa = qa_rounds[0] if qa_rounds else text
    length = len(tokenizer.encode(first_qa, add_special_tokens=False))
    
    return first_qa, length


def list_to_dict(lst):
    return {i: element for i, element in enumerate(lst)}

def find_first_indexes(my_list, targets):
    if not targets or len(my_list) < len(targets):
        return []
    
    # 找到目标序列最后一次出现的起始位置
    last_sequence_start = -1
    for i in range(len(my_list) - len(targets) + 1):
        if tuple(my_list[i:i+len(targets)]) == tuple(targets):
            last_sequence_start = i
            
    # 如果找到了序列，返回每个位置的索引
    if last_sequence_start != -1:
        return [last_sequence_start + i for i in range(len(targets))]
    
    # 如果没找到完整序列，返回空列表
    return []


import pickle
import matplotlib.pyplot as plt
def load_delta_data(path = "/home/jameswang/CoT-decoding/res/Delta"):
    try:
        with open(os.path.join(path, 'deltas_cot.pkl'), 'rb') as f:
            deltas_cot = pickle.load(f)
        with open(os.path.join(path, 'deltas_non_cot.pkl'), 'rb') as f:
            deltas_non_cot = pickle.load(f)
        return deltas_cot, deltas_non_cot
    
    except FileNotFoundError as e:
        print(f"Error: 文件未找到 - {e}")
        return None, None
    except Exception as e:
        print(f"Error: 加载数据时发生错误 - {e}")
        return None, None

def plot_two_lists_with_means(list1, list2):
    # 找到最短长度
    min_length = min(len(list1), len(list2))
    
    # 截取两个列表到相同的长度
    list1 = list1[:min_length]
    list2 = list2[:min_length]
    
    # 计算均值
    mean_list1 = sum(list1) / len(list1)
    mean_list2 = sum(list2) / len(list2)
    
    # 绘制两个列表的折线图
    plt.plot(list1, marker='o', color='blue', label='cot')  # 第一个列表的折线图
    plt.plot(list2, marker='x', color='red', label='non-cot')   # 第二个列表的折线图
    
    # 绘制均值线
    plt.axhline(mean_list1, color='blue', linestyle='--', linewidth=1, label='Mean of cot')  # 第一个列表的均值线
    plt.axhline(mean_list2, color='red', linestyle='--', linewidth=1, label='Mean of non-cot')  # 第二个列表的均值线
    
    # 图表标题和坐标轴标签
    plt.title("Line Chart of Two Lists with Means")  
    plt.xlabel("Index")  
    plt.ylabel("Value")  
    
    # 显示图例和网格
    plt.legend()         
    plt.grid(True)       
    plt.show()