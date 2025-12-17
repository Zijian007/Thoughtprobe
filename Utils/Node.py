import torch
import torch.nn as nn
import re, json, os
import numpy as np
from scipy import stats
from typing import List
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Set, List, Tuple

class Node:
    def __init__(self, model, tokenizer, score_fn, ids, text= None, parent = None, depth = 0, pos = None):
        """
        ids: tensor(1, length) - 当前路径的token ids
        parent: Path - 父节点
        score: float - 当前路径的评分
        depth: int - 当前节点在树中的深度
        """
        self.text = text
        self.state = text
        self.ids = ids
        self.model = model
        self.score_fn = score_fn

        self.parent = parent
        self.score = None
        self.original_score = None
        self.depth = depth
        self.children = {}
        self.childs = []
        self.all_childs = {}
        self.last_sample_step = 0
        self.extendable:bool = False
        self.is_sampled:bool = False
        self.all_scores = None
        self.pos = pos

    @property
    def length(self):
        return self.ids.size(1)
    
    def scoring(self, layers_range: List, rep, use_last_token = True):
        if rep == "hiddens":
        # 对路径进行打分
            hs = self.model(self.ids.to(self.model.device), output_hidden_states = True).hidden_states[1: ]   #hidden_states_tuple: (tensor(bsz, length, hidden_size),....) len: layers_num
            hs = reshape_hidden_states(hs) # list[tensor(length, layers_num, hidden_size)] len: bsz
            hs = torch.stack(hs, dim = 0).to(torch.float32) # (bsz, length, layers_num, hidden_size)
        else:
            hs = compute_activations(self.ids, self.model, representation = rep)
            hs = hs.to(torch.float32)
        original_score = self.score_fn(hs)  # return (bsz, token_length, num_layers)
        
        self.original_score = original_score
        # 只保留指定layer的分数
        selected_scores = original_score[:, :, layers_range]  # (bsz, length, len(layer_range))
        # 计算每个样本的得分
        if use_last_token:
            # 使用最后一个token的分数, 在指定层上取平均
            sample_score = selected_scores[:, -1, :].mean(dim = -1)  # (bsz,)
        else:
            # 使用所有token的平均分数
            sample_score = selected_scores.mean(dim = [1, 2])  # (bsz,)
        self.score = sample_score.item()
        self.metrics_dict= calculate_fluctuation_metrics(self.get_sequence_score())

        hs = hs.cpu()
        del hs
        torch.cuda.empty_cache()

    def add_child(self, child):
        """添加子节点"""
        self.childs.append(child)
        
    def get_full_path(self):
        """获取从根节点到当前节点的完整路径"""
        if self.parent is None:
            return [self]
        return self.parent.get_full_path() + [self]
    
    def get_sequence_score(self, include_root=False):
        """
        获取从根节点到当前节点的分数序列
        
        Args:
            include_root: 是否包含根节点的分数
        """
        if self.parent is None:
            return [self.score] if self.score is not None else [0]
            
        path = self.get_full_path()
        if not include_root:
            path = path[1:]  # 排除根节点
            
        scores = [p.score for p in path if p.score is not None]
        return scores
    
    @property
    def is_leaf(self):
        """判断是否为叶子节点"""
        return len(self.childs) == 0
    
    def get_leaves(self):
        """获取以当前节点为根的所有叶子节点，按depth排序"""
        if self.is_leaf:
            return [self]
            
        leaves = []
        for child in self.childs:
            leaves.extend(child.get_leaves())
    
        # 按depth排序
        return sorted(leaves, key=lambda x: x.depth)
    
    def get_all_parents(self):
        """
        获取从当前节点到根节点的所有父节点（不包括当前节点）
        
        返回:
            List[Node] - 从当前节点的直接父节点到根节点的列表
        """
        parents = []
        current = self.parent
        while current is not None:
            parents.append(current)
            current = current.parent
        return parents
    
    def collect_all_related_nodes(self):
        """收集所有相关联的节点（包括parents和leaves）"""
        nodes_dict = {}
        
        def collect_node(node, nodes):
            # 用id作为唯一标识符
            node_id = id(node)
            if node_id not in nodes:
                nodes[node_id] = node
                # 收集所有子节点
                for child in node.childs:
                    collect_node(child, nodes)
                # 收集父节点
                if node.parent:
                    collect_node(node.parent, nodes)
        
        collect_node(self, nodes_dict)
        return nodes_dict

    def to_dict(self):
        """将Node实例转换为可序列化的字典"""
        return {
            'id': id(self),  # 用于标识节点
            'ids': self.ids.tolist() if self.ids is not None else None,
            'score': float(self.score) if self.score is not None else None,
            'original_score': float(self.original_score) if self.original_score is not None else None,
            'depth': self.depth,
            'text': self.text,
            'pos': self.pos,
            'last_sample_step': self.last_sample_step,
            'parent_id': id(self.parent) if self.parent is not None else None,
            'child_ids': [id(child) for child in self.childs]
        }

    def save_to_file(self, filepath):
        """将节点及其所有相关节点保存到文件"""
        # 收集所有相关节点
        all_nodes = self.collect_all_related_nodes()
        
        # 转换为可序列化的格式
        nodes_data = {
            'nodes': {str(node_id): node.to_dict() for node_id, node in all_nodes.items()},
            'root_id': str(id(self))  # 保存当前节点的ID作为入口点
        }
        
        # 保存到文件
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath):
        """从文件加载节点及其关系"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 第一遍：创建所有节点
        nodes_dict = {}
        for node_id, node_data in data['nodes'].items():
            node = cls()
            node.ids = torch.tensor(node_data['ids']) if node_data['ids'] is not None else None
            node.score = node_data['score']
            node.original_score = node_data['original_score']
            node.depth = node_data['depth']
            node.text = node_data['text']
            node.pos = node_data['pos']
            node.last_sample_step = node_data['last_sample_step']
            nodes_dict[node_id] = node
        
        # 第二遍：建立节点之间的关系
        for node_id, node_data in data['nodes'].items():
            node = nodes_dict[node_id]
            # 设置父节点
            if node_data['parent_id'] is not None:
                node.parent = nodes_dict[str(node_data['parent_id'])]
            # 设置子节点
            for child_id in node_data['child_ids']:
                child = nodes_dict[str(child_id)]
                node.add_child(child)
        
        # 返回入口节点
        return nodes_dict[data['root_id']]

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

def remove_padding(tokens: torch.Tensor, pad_token_id: int = None, tokenizer = None) -> torch.Tensor:
    """
    去除tensor中的padding tokens
    
    参数:
        tokens: torch.Tensor - shape为[batch_size, seq_len]或[seq_len]的tensor
        pad_token_id: int - padding token的ID
        tokenizer: transformers.PreTrainedTokenizer - tokenizer对象
    
    返回:
        torch.Tensor - 移除padding后的tensor
    """
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id
    
    # 确保是2D tensor
    is_1d = tokens.dim() == 1
    if is_1d:
        tokens = tokens.unsqueeze(0)
    
    # 找到每个序列中非padding的位置
    mask = tokens != pad_token_id
    
    # 获取每个序列的有效长度
    seq_lengths = mask.sum(dim=1)
    max_len = seq_lengths.max().item()
    
    # 创建新的tensor只保留有效部分
    batch_size = tokens.size(0)
    output = tokens.new_zeros((batch_size, max_len))
    
    for i in range(batch_size):
        valid_length = seq_lengths[i].item()
        output[i, :valid_length] = tokens[i, mask[i]][:valid_length]
    
    # 如果输入是1D，返回1D
    if is_1d:
        output = output.squeeze(0)
    return output

def pad_and_concat(tensor_list: List[torch.Tensor], pad_value: int = 0, dim: int = 0) -> torch.Tensor:
    """
    将形状为(bsz, seq_len)或(bsz, top_k, seq_len)的tensor列表填充到相同长度并在指定维度拼接
    
    参数:
        tensor_list: List[torch.Tensor] - tensor列表
        pad_value: int - 用于填充的值
        dim: int - 拼接的维度
        
    返回:
        torch.Tensor - 填充并拼接后的tensor
    """
    # 获取每个tensor的shape
    shapes = [t.shape for t in tensor_list]
    
    # 确定是2维还是3维tensor
    is_2d = len(shapes[0]) == 2
    
    # 确保所有tensor维度一致
    assert all(len(s) == len(shapes[0]) for s in shapes), "All tensors must have same number of dimensions"
    
    if is_2d:
        bsz = shapes[0][0]
        # 确保batch size一致
        assert all(s[0] == bsz for s in shapes), "Batch size must be consistent"
        # 找到最大seq_len
        max_len = max(s[1] for s in shapes)
        
        # 填充每个tensor到相同长度
        padded_tensors = []
        for tensor in tensor_list:
            cur_len = tensor.size(1)
            if cur_len < max_len:
                padding_size = (0, max_len - cur_len)  # 在最后一维右侧填充
                padded_tensor = F.pad(tensor, padding_size, mode='constant', value=pad_value)
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
    else:
        bsz, top_k = shapes[0][:2]
        # 确保batch size和top_k一致
        # assert all(s[0] == bsz and s[1] == top_k for s in shapes), "Batch size and top_k must be consistent"
        # 找到最大seq_len
        max_len = max(s[2] for s in shapes)
        
        # 填充每个tensor到相同长度
        padded_tensors = []
        for tensor in tensor_list:
            cur_len = tensor.size(2)
            if cur_len < max_len:
                padding_size = (0, max_len - cur_len)  # 在最后一维右侧填充
                padded_tensor = F.pad(tensor, padding_size, mode='constant', value=pad_value)
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
    
    # 在指定维度拼接
    return torch.cat(padded_tensors, dim=dim)

def top_k_then_greedy(model, tokenizer, batch_nodes, max_new_tokens = 30, top_k = 10, eos_token_ids = [28723, 13]):  #input_ids: tensor(bsz, length)
    with torch.no_grad():
        num_parents = len(batch_nodes)
        # inp = {}
        # inp["input_ids"] = input_ids.to(model.device)
        # inp["attention_mask"] = torch.ones_like(input_ids).to(model.device)
        batch_prompts = [node.text for node in batch_nodes] # 每个node的text长度可能不一样
        tokenizer.padding_side = "left"         # 为了保证每个batch的长度一样, 选择左填充
        inp = tokenizer(batch_prompts, return_tensors = "pt", padding = True).to(model.device) # (bsz, padded_seq_len)

        # 1. 第一步选择top_k
        outputs = model(**inp)
        next_token_logits = outputs.logits[:, -1, :]  # [ num_parents, vocab_size]
        next_token_probs = nn.functional.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_tokens = next_token_probs.topk(top_k)  # [ num_parents, top_k]
        top_k_tokens_flat = top_k_tokens.view(-1).unsqueeze(-1) # [ num_parents * top_k, 1]
        
        # 创建(num_parents  * k)个序列
        input_ids = inp["input_ids"]
        base_sequence = input_ids.repeat_interleave(top_k, dim=0)  # [ num_parents * top_k, seq_len]
        base_attention_mask = inp["attention_mask"].repeat_interleave(top_k, dim=0)

        new_inp_ids = torch.cat([base_sequence, top_k_tokens_flat], dim=1) #
        new_attention_mask = torch.cat([base_attention_mask, torch.ones_like(top_k_tokens_flat)], dim=1)
        if max_new_tokens > 1:
        # 2. 使用generate进行贪心生成
            new_inp = {}
            new_inp["input_ids"] = new_inp_ids
            new_inp["attention_mask"] = new_attention_mask
            outputs = model.generate(
                **new_inp,
                eos_token_id = eos_token_ids, 
                max_new_tokens = max_new_tokens-1,  # 因为已经生成了一个token
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id 
            ) 
            #得到的是(num_parents  * k)个序列, 每个序列长度为seq_len + max_new_tokens - 1
            # shape: [batch_size * top_k, seq_len + max_new_tokens - 1] 
        else:
            outputs = new_inp_ids
            del new_inp_ids
            del inp
            torch.cuda.empty_cache()
    # 重塑输出为 [num_parents , top_k, seq_len]
    final_seq_len = outputs.shape[1]
    outputs = outputs.view(num_parents, top_k, final_seq_len)
    return outputs

def greedy(model, tokenizer, batch_nodes, max_new_tokens = 30, eos_token_ids = [28723, 13]):  #input_ids: tensor(bsz, length)
    tokenizer.padding_side = "left"         # 为了保证每个batch的长度一样, 选择左填充
    with torch.no_grad():
        num_parents = len(batch_nodes)
        batch_prompts = [node.text for node in batch_nodes] # 每个node的text长度可能不一样
        
        inp = tokenizer(batch_prompts, return_tensors = "pt", padding = True).to(model.device) # (bsz, padded_seq_len)
        
        outputs = model.generate(
            **inp,
            eos_token_id = eos_token_ids, 
            max_new_tokens = max_new_tokens,  # 因为已经生成了一个token
            do_sample = False,
            pad_token_id = tokenizer.pad_token_id 
        ) 
        outputs = outputs.unsqueeze(1)  # [num_parents, 1, seq_len]
    return outputs

def expand(model, tokenizer, root, current_all_leaves: List, 
           bsz = 1, strategy = "top_k_then_greedy", max_length_node = 30, top_k = 10, eos_token_ids = [])-> List[Node]:
    # 1. 分批生成候选子节点
    num_all_leaves = len(current_all_leaves)
    all_childs_ids_list = []
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 8257
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(8257)
    for i in range(0, num_all_leaves, bsz):
        # print(f"Expanding batch {i} with bsz {bsz}, total {num_all_leaves}")

        # 2. 获取当前批次的父节点
        batch_leaves = current_all_leaves[i: i + bsz]
        if strategy == "top_k_then_greedy":
            batch_child_ids = top_k_then_greedy(model, tokenizer, batch_leaves, max_new_tokens = max_length_node, top_k = top_k, eos_token_ids = eos_token_ids)  # [bsz, top_k, seq_len]
            # all_childs_ids_list.append(batch_child_ids)
        elif strategy == "greedy":
            batch_child_ids = greedy(model, tokenizer, batch_leaves, max_new_tokens = max_length_node, eos_token_ids = eos_token_ids)  # [bsz, seq_len]
        all_childs_ids_list.append(batch_child_ids)

    all_childs_ids = pad_and_concat(all_childs_ids_list, pad_value=tokenizer.pad_token_id, dim = 0) # [num_next_leaves, top_k, seq_len] or [num_next_leaves, 1, seq_len]
    num_per_parent = all_childs_ids.size(1)
    
    all_children = []
    extendable_children = []
    candidate_children = []
    # 3. 为每个父节点处理其候选子节点
    for i, current_node in enumerate(current_all_leaves):
        childs_per_parent = []
        # 3.1 创建当前父节点的所有候选子节点
        for j in range(num_per_parent):
            clean_ids = remove_padding(all_childs_ids[i, j], pad_token_id = tokenizer.pad_token_id, tokenizer = tokenizer)
            text = tokenizer.decode(clean_ids, skip_special_tokens = True)
            child = Node(
                model,
                tokenizer,
                score_fn = root.score_fn,
                text = text,
                ids = all_childs_ids[i, j].unsqueeze(0),
                parent = current_node,
                depth = current_node.depth + 1,
                pos = (i,j)
            )
            all_children.append(child)
            childs_per_parent.append(child)
            current_node.add_child(child)
            # 检查是否包含无效token
            child_new_ids = child.ids[0][len(root.ids[0]):]
            invalid_tokens = []
            invalid_tokens = tokenizer.convert_tokens_to_ids(['▁Question','Question'])
            
            invalid_tokens.append(tokenizer.eos_token_id)
            trimmed_ids = trim_invalid_tokens(child_new_ids, invalid_tokens = invalid_tokens).to(model.device)
            if len(trimmed_ids) < len(child_new_ids):
                # 存在非法token,需要更新child的ids
                child.ids = torch.cat([root.ids[0].to(model.device), trimmed_ids]).unsqueeze(0)
                child.extendable = False
                # print("Invalid token in child")
            # if not any(token in child_new_ids for token in [22478, 24994, tokenizer.eos_token_id]):
            #     child.extandable = False
            else:
                child.extendable = True
                extendable_children.append(child)       
        candidate_children.append(childs_per_parent)  
        
    candidate_children: List[List[Node]]
    all_children: List[Node]
    extendable_children: List[Node] 
    return candidate_children, all_children, extendable_children

# # 2D tensor示例
# t1 = torch.randn(2, 3)  # batch_size=2, seq_len=3
# t2 = torch.randn(2, 4)  # batch_size=2, seq_len=4
# result_2d = pad_and_concat([t1, t2])
# print(result_2d.shape)
# # 3D tensor示例
# t3 = torch.randn(2, 5, 3)  # batch_size=2, top_k=5, seq_len=3
# t4 = torch.randn(2, 5, 4)  # batch_size=2, top_k=5, seq_len=4
# result_3d = pad_and_concat([t3, t4])
# print(result_3d.shape)


def trim_invalid_tokens(ids, invalid_tokens=[]):
    """裁剪掉非法token及其后续token"""
    # 查找第一个非法token的位置
    first_invalid_pos = -1
    for i, token in enumerate(ids):
        if token in invalid_tokens:
            # print(f"Found invalid token {token} at position {i}")
            first_invalid_pos = i
            break
            
    if first_invalid_pos != -1:
        # 如果找到非法token,只保留之前的token
        return ids[:first_invalid_pos]
    return ids

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

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Literal
RepresentationType = Literal[
    "hiddens", "pre-attn", "queries", "keys", "values", "heads", "mlp", "post-attn"
]
from numpy.typing import NDArray
from baukit import TraceDict

def compute_activations(
    inputs, 
    model: PreTrainedModel, 
    representation: RepresentationType = "hiddens",  
    output_dtype=torch.float32,
):
    device = next(model.parameters()).device

    trace_modules = []
    if representation == "pre-attn":
        target_key = "input_layernorm"
        error_msg = f"Pre-norm not found in {model.__class__.__name__}"
    elif representation == "queries":
        target_key = "self_attn.q_proj"
        error_msg = f"Query not found in {model.__class__.__name__}"
    elif representation in ("heads", "post-attn"):
        target_key = "self_attn"
        error_msg = f"Self-attention not found in {model.__class__.__name__}"
    elif representation == "mlp":
        target_key = "mlp"
        error_msg = f"MLP not found in {model.__class__.__name__}"
    else:
        target_key = None
        error_msg = None

    if target_key is not None:
        for i, layer in enumerate(model.model.layers):
            found = False
            for name, _ in layer.named_modules():
                if target_key in name:
                    trace_modules.append(f"model.layers.{i}.{name}")
                    found = True
                    break
            if not found:
                raise ValueError(error_msg)

    with TraceDict(model, trace_modules) as trace:
        inputs = inputs.to(device)
        outputs = model(inputs, output_hidden_states=True, return_dict=True, use_cache=True)
        if representation == "hiddens":
            reps = torch.cat(outputs.hidden_states[1:]).squeeze(1)
        elif representation == "pre-attn":
            hiddens = [trace[name].output for name in trace_modules]
            reps = torch.cat(hiddens, dim=0)
        elif representation == "queries":
            queries = [trace[name].output for name in trace_modules]
            reps = torch.cat(queries, dim=0)
        elif representation == "keys":
            reps = torch.cat([x[0] for x in outputs.past_key_values]).squeeze(1).transpose(1, 2)
            reps = reps.reshape(reps.shape[:2] + (-1,))
        elif representation == "values":
            reps = torch.cat([x[1] for x in outputs.past_key_values]).squeeze(1).transpose(1, 2)
            reps = reps.reshape(reps.shape[:2] + (-1,))
        elif representation == "heads":
            num_heads = model.config.num_attention_heads
            hiddens = [trace[name].output[0] for name in trace_modules]
            hiddens = torch.cat(hiddens, dim=0)
            heads = hiddens.reshape(*hiddens.shape[:2], num_heads, -1)
            reps = heads
        elif representation == "post-attn":
            post_attn = [trace[name].output[0] for name in trace_modules]
            reps = torch.cat(post_attn, dim=0)
        elif representation == "mlp":
            mlp = [trace[name].output for name in trace_modules]
            reps = torch.cat(mlp, dim=0)
        else:
            raise ValueError(f"Unknown representation: {representation}")
        reps = reps.permute(1, 0, 2).unsqueeze(0)
    return reps