import numpy as np
from typing import List, Optional, Callable, Dict, Union
import random
from Utils.Node import Node


class TreeSearch:
    def __init__(self, 
                 model,
                 tokenizer,
                 expansion_func: Callable,
                 score_fun: Callable,
                 num_samples: int = 3,
                 temperature: float = 1.0,
                 score_layers: List = []
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.expansion_func = expansion_func
        self.scoring_func = score_fun
        self.num_samples = num_samples
        self.temperature = temperature
        self.root = None
        self.current_leaves = []
        self.sampled_nodes_by_level = {}
        self.all_nodes_by_level = {}
        self.extendable_nodes_by_level = {}
        assert all(node.is_sampled for node in self.current_leaves), "All nodes should be sampled in current layers."

        self.score_layers = score_layers

    def initialize(self, initial_state:str, rep = "mlp"):
        """初始化搜索树"""
        root_ids = self.tokenizer.encode(initial_state, return_tensors="pt")
        self.root = Node(self.model, self.tokenizer, self.scoring_func, text = initial_state, ids = root_ids)
        self.root.is_sampled = True
        self.root.extendable = True
        # 为根节点打分
        # self.score_nodes([self.root])
        self.root.scoring(layers_range=self.score_layers, rep = rep)
        self.current_leaves = [self.root]
        self.sampled_nodes_by_level = {0: [self.root]}
        self.all_nodes_by_level = {0: [self.root]}
        # self.all_nodes = [self.root]


    def score_nodes(self, nodes:Union[ List[Node] ,List[List[Node]]], rep = "hiddens"):
        """为节点列表打分
        
        Args:
            nodes: 可以是节点列表 List[Node] 或嵌套节点列表 List[List[Node]]
        """
        # 检查是否为嵌套列表
        if nodes and isinstance(nodes[0], list):
            # 处理嵌套列表
            for sublist in nodes:
                for node in sublist:
                    node.scoring(layers_range = self.score_layers, rep = rep)
        else:
            # 处理普通列表
            for node in nodes:
                node.scoring(layers_range = self.score_layers, rep = rep)

    def sample_nodes(self, nodes_list: List[List[Node]], num_samples: int, flatten: bool = False, deterministic: bool = True) -> List[Node]:
        """基于分数采样节点
        
        Args:
            nodes_list: 嵌套的节点列表
            num_samples: 需要采样的节点数量
            flatten: 是否展开列表进行采样
                - True: 从所有节点中采样num_samples个节点
                - False: 从每个子列表中采样num_samples个节点
            deterministic: 是否使用确定性采样
                - True: 选择分数最高的节点
                - False: 使用概率采样
        
        Returns:
            采样得到的节点列表
        """
        if not nodes_list:
            return []
            
        if flatten:
            # 展开所有节点
            nodes = [node for sublist in nodes_list for node in sublist]
            if not nodes:
                return []
                
            if deterministic:
                # 确定性采样：选择分数最高的节点
                scores = np.array([node.score for node in nodes])
                selected_indices = np.argsort(scores)[-min(num_samples, len(nodes)):]
                return [nodes[i] for i in selected_indices]
            
            else:
                # 随机采样
                scores = np.array([node.score for node in nodes])
                scores = scores - np.max(scores)
                probs = np.exp(scores / self.temperature)
                probs = probs / probs.sum()
                
                selected_indices = np.random.choice(
                    len(nodes),
                    size=min(num_samples, len(nodes)),
                    p=probs,
                    replace=False
                )
                return [nodes[i] for i in selected_indices]
        
        else:
            # 从每个子列表中采样
            sampled_nodes = []
            for sublist in nodes_list:
                if not sublist:
                    continue

                if deterministic:
                    # 确定性采样：选择每个子列表中分数最高的节点
                    scores = np.array([node.score for node in sublist])
                    sample_size = min(num_samples, len(sublist))
                    selected_indices = np.argsort(scores)[-sample_size:]
                    sampled_nodes.extend([sublist[i] for i in selected_indices])

                else:
                    # 随机采样
                    scores = np.array([node.score for node in sublist])
                    scores = scores - np.max(scores)
                    probs = np.exp(scores / self.temperature)
                    probs = probs / probs.sum()
                    
                    sample_size = min(num_samples, len(sublist))
                    selected_indices = np.random.choice(
                        len(sublist),
                        size=sample_size,
                        p=probs,
                        replace=False
                    )
                    sampled_nodes.extend([sublist[i] for i in selected_indices])
                    
            return sampled_nodes
    
    def expand_node(self, nodes: List[Node], 
                    strategy = "top_k_then_greedy", bsz = 3,
                    max_length_node = 30, top_k = 10, eos_token_ids = []) -> List[Node]:
        """扩展一个节点，生成其所有可能的子节点"""

        candidate_children: List[List[Node]]
        all_children: List[Node]
        extendable_children: List[Node]
        candidate_children, all_children, extendable_children = self.expansion_func(self.model, self.tokenizer, self.root, nodes, 
                                             bsz = bsz, strategy = strategy, max_length_node = max_length_node, top_k = top_k,
                                             eos_token_ids = eos_token_ids)
        return candidate_children, all_children, extendable_children
    

    def search_step(self, step: int, sample_steps: int = 3, max_length_node = 30, top_k = 10,
                    flatten: bool = True, bsz: int = 3, eos_token_ids = [], rep = "mlp"):
        """执行一步搜索"""
        if not self.current_leaves:
            return False
        step = step + 1
        # 确定搜索策略
        strategy = "top_k_then_greedy" if step < sample_steps else "greedy"
        strategy_desc = "increase the number of branches" if step < sample_steps else "use greedy strategy per branch"
        print(f"Step {step} {'<' if step < sample_steps else '>='} {sample_steps}, {strategy_desc}.")
        
        # 扩展当前叶子节点
        all_candidate_nodes, all_children, extendable_children = self.expand_node(
            self.current_leaves,
            strategy = strategy,
            max_length_node = max_length_node,
            top_k = top_k,
            bsz = bsz,
            eos_token_ids = eos_token_ids
        )
        self.all_nodes_by_level[step] = all_children
        self.extendable_nodes_by_level[step] = extendable_children
        # 打印候选节点信息
        num_candidates = sum(len(sublist) for sublist in all_candidate_nodes)
        sublists_len = [len(sublist) for sublist in all_candidate_nodes]
        print(f"Step {step} gets {num_candidates} candidate nodes for extendable identification from "
            f"{len(self.current_leaves)} parents, with sublists length: {sublists_len}.")
        
        # 为新节点打分
        self.score_nodes(all_candidate_nodes, rep = rep)
        
        # 筛选可扩展节点
        extandable_new_nodes = [[node for node in sublist if node.extendable] 
                            for sublist in all_candidate_nodes]
        num_extandable = sum(len(sublist) for sublist in extandable_new_nodes)
        
        # 验证可扩展节点数量
        assert num_extandable == len(extendable_children), \
            "The number of extendable new nodes should be equal to the length of extendable_children."
        
        sublists_len = [len(sublist) for sublist in extandable_new_nodes]
        print(f"Step {step} found {num_extandable} extendable candidates, "
            f"with sublists length: {sublists_len}.")
        
        # 采样新的叶子节点
        self.current_leaves = self.sample_nodes(extandable_new_nodes, 
                                            self.num_samples, 
                                            flatten=flatten)
        
        # 更新节点关系
        for node in self.current_leaves:
            node.parent.is_sampled = True
            node.parent.add_child(node)
            
        print(f"Step {step} sampled {len(self.current_leaves)} new nodes from {num_extandable} extendable candidates.")
        
        # 保存本轮结果
        if self.current_leaves:
            self.score_nodes(self.current_leaves, rep = rep)
            self.sampled_nodes_by_level[step] = self.current_leaves

            
        return True
            
    def search(self, initial_state, num_steps: Optional[int] = None, 
               sample_steps = 3, max_length_node  = [],
               top_k = 10, flatten: bool = True, bsz: int = 3, 
               chunk:bool = False, rep = "mlp"):
        """执行完整的搜索过程"""
        self.initialize(initial_state, rep = rep)
        if chunk:
            eos_token_ids = [28723, 13]  #换行符和句号
        else:
            eos_token_ids = []
        for step in range(0, num_steps):
            print(f"Process step {step+1}")
            if not self.search_step(step, sample_steps = sample_steps, max_length_node = max_length_node[step], 
                                    top_k = top_k, flatten = flatten, bsz = bsz, eos_token_ids = eos_token_ids,
                                    rep = rep):
                break
            print(f"Step {step+1} completed. Current leaves: {len(self.current_leaves)}")

    def get_best_path(self):
        """获取得分最高的路径"""
        if not self.current_leaves:
            return []
        best_leaf = max(self.current_leaves, 
                       key=lambda x: x.score if x.score is not None else float('-inf'))
        return best_leaf

    def get_nodes_at_level(self, level: int) -> List[Node]:
        """获取特定层级的所有节点"""
        return self.nodes_by_level.get(level, [])

    def get_best_node_at_level(self, level: int) -> Optional[Node]:
        """获取特定层级得分最高的节点"""
        nodes = self.get_nodes_at_level(level)
        if not nodes:
            return None
        return max(nodes, key=lambda x: x.score if x.score is not None else float('-inf'))
    
    def get_node_parent(self, node: Node) -> Optional[Node]:
        """获取指定节点的父节点"""
        return node.parent

    def get_node_children(self, node: Node) -> List[Node]:
        """获取指定节点的所有子节点"""
        return node.children

    def trace_path_to_root(self, node: Node) -> List[Node]:
        """从指定节点追踪到根节点的路径"""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]  # 反转列表使其从根节点开始

    def get_node_info(self, node: Node) -> dict:
        """获取节点的详细信息"""
        return {
            'state': node.state,
            'score': node.score,
            'depth': node.depth,
            'parent_state': node.parent.state if node.parent else None,
            'num_children': len(node.children),
            'children_states': [child.state for child in node.children]
        }

    def print_node_info(self, node: Node):
        """打印节点的详细信息"""
        info = self.get_node_info(node)
        print(f"\n节点信息:")
        print(f"状态: {info['state']}")
        print(f"分数: {info['score']:.2f if info['score'] is not None else 'None'}")
        print(f"深度: {info['depth']}")
        print(f"父节点状态: {info['parent_state']}")
        print(f"子节点数量: {info['num_children']}")
        if info['num_children'] > 0:
            print(f"子节点状态: {info['children_states']}")

    def print_tree_structure(self):
        """打印整个树的结构，包含父子关系"""
        for depth in sorted(self.sampled_nodes_by_level.keys()):
            nodes = self.sampled_nodes_by_level[depth]
            
            print(f"\nLevel {depth} (Total nodes: {len(nodes)}):")
            for i, node in enumerate(nodes):
                print(f"\nLevel node.depth {node.depth} ")
                parent_state = node.parent.state if node.parent else "None"
                score_str = f"{node.score:.2f}" if node.score is not None else "None"
                print(f"Level {depth}, Node {i}: {node.state}| \n ***score={score_str}***, ***pos = {node.pos}***, \
                      extendable={node.extendable}, is_leaf={node.is_leaf}, is_sampled={node.is_sampled},\
                      childs = {len(node.childs)}")
                print(f"    ↳ parent_state={parent_state}")
                print("=========================================")

# 使用示例
def example_usage():
    def expand(state):
        num_children = 3
        return [state + [i] for i in range(num_children)]

    def score(states):
        return [sum(state) / len(state) for state in states]

    searcher = TreeSearch(
        expansion_func=expand,
        scoring_func=score,
        num_samples = 2,
        max_depth = 3,
        temperature = 0.5
    )

    initial_state = [0]
    best_path = searcher.search(initial_state, num_steps = 3)

    # 打印树结构（包含父子关系）
    searcher.print_tree_structure()

    # 打印展开过程
    searcher.print_expansion_process()

    # 示例：获取特定节点的相关信息
    print("\n示例节点分析:")
    if len(searcher.all_nodes) > 1:
        example_node = searcher.all_nodes[9]
        print(f"选择节点: {example_node.state}")
        
        # 获取父节点
        parent = searcher.get_node_parent(example_node)
        if parent:
            print(f"父节点: {parent.state}")
        
        # 获取子节点
        children = searcher.get_node_children(example_node)
        print(f"子节点: {[child.state for child in children]}")
        
        # 追踪到根节点的路径
        path = searcher.trace_path_to_root(example_node)
        print(f"到根节点的路径: {[node.state for node in path]}")

if __name__ == "__main__":
    example_usage()