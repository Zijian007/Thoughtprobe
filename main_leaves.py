from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, json, sys, logging
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from data_utils import setup_data_loader

import numpy as np
from methods import tree_search_decoding, greed_decoding, cot_decoding, zs_cot_prompting
    # Import custom modules
sys.path.append(os.getcwd())
# sys.path.append("/hdd/zijianwang/CoT-decoding")

from Prober.SVM import MultiLayerSVM
from Prober.LR import  MultiLayerLogisticRegression

# ------------------ Utility Functions ------------------
def setup_logging(config):
    """Configure logging settings"""
    log_filename = os.path.join(config.LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.exp_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), 
                  logging.StreamHandler()]
    )

def save_checkpoint(data_id, metric_counts, config):
    """Save checkpoint to file"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.exp_name}.json")
    with open(checkpoint_path, 'w') as f:
        json.dump({'data_id': data_id, 'metric_counts': metric_counts}, f)

def load_checkpoint(config):
    """Load the latest checkpoint"""
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.exp_name}.json")
    return json.load(open(checkpoint_path)) if os.path.exists(checkpoint_path) else None

def initialize_model_and_tokenizer(config):
    """Initialize the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.MODEL_CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, cache_dir=config.MODEL_CACHE_DIR, 
        torch_dtype=torch.bfloat16, device_map=config.device
    )
    return model, tokenizer

def initialize_prober(config):
    """Initialize the prober model"""
    if "SVM" in config.prober_type:
        prober = MultiLayerSVM(input_dim=config.prober_input_dim, 
                               num_layers=config.prober_num_layers,
                               base_save_path=os.path.join(config.prober_base_path, config.prober_type), 
                               device=config.device)
        prober.load_all_models()

    elif "LR" in config.prober_type:
        prober = MultiLayerLogisticRegression(input_dim=config.prober_input_dim, 
                            num_layers=config.prober_num_layers,
                            base_save_path=os.path.join(config.prober_base_path, config.prober_type), 
                            device=config.device)
        prober.load_all_models()
    return prober

# ------------------ Processing Functions ------------------


def process_single_example(data, model, tokenizer, score_fn, config):
    tree_leaves, greedy_pred, cot_decoding_pred, zs_cot_prompt_pred   = None, None, None, None
    for method in config.methods:
        if method == "search":
            config.current_method = "search"
            tree_leaves, searcher = tree_search_decoding(data, model, tokenizer, score_fn, config)
        elif method == "greedy":
            config.current_method = "greedy"
            greedy_pred = greed_decoding(model, tokenizer, data, config)
        elif method == "cot_decoding":
            config.current_method = "cot_decoding"
            cot_decoding_pred = cot_decoding(model, tokenizer, data, config)
        elif method == "zs_cot_prompt":
            config.current_method = "zs_cot_prompt"
            zs_cot_prompt_pred = zs_cot_prompting(model, tokenizer, data, config)
    return tree_leaves, greedy_pred, cot_decoding_pred, zs_cot_prompt_pred

def process_greedy_metrics(config, greedy_pred:str, gt:int, data_id:int, metric_counts:dict, total_samples:int):
    """Process metrics and logging for the results"""
    def estimate_acc(acc, num, total=1319):
        return (acc /num) * total
    if np.array(gt) == np.array(greedy_pred):
        metric_counts["greedy_pred"] = metric_counts.get("greedy_pred", 0) + 1
    acc_greedy_pred = metric_counts["greedy_pred"] / total_samples
    logging.info(f"Iteration {data_id+1} / {total_samples}- True answer: {gt}")
    logging.info(f"Iteration {data_id+1} / {total_samples} - Greedy prediction: {greedy_pred}")
    logging.info(f"Iteration {data_id+1} / {total_samples} - Greedy prediction accuracy: {acc_greedy_pred:.4f}, estimated: {estimate_acc(acc_greedy_pred, data_id+1, total_samples)}")
    return metric_counts

def process_zs_cot_prompt_metrics(config, zs_cot_prompt_pred:str, gt:int, data_id:int, metric_counts:dict, total_samples:int):
    """Process metrics and logging for the results"""
    def estimate_acc(acc, num, total=1319):
        return (acc /num) * total
    if  np.array(gt) == np.array(zs_cot_prompt_pred):
        metric_counts["zs_cot_prompt_pred"] = metric_counts.get("zs_cot_prompt_pred", 0) + 1
    acc_zs_cot_prompt_pred = metric_counts["zs_cot_prompt_pred"] / total_samples
    logging.info(f"Iteration {data_id+1} / {total_samples}- True answer: {gt}")
    logging.info(f"Iteration {data_id+1} / {total_samples} - ZS Cot Prompt prediction: {zs_cot_prompt_pred}")
    logging.info(f"Iteration {data_id+1} / {total_samples} - ZS Cot Prompt prediction accuracy: {acc_zs_cot_prompt_pred:.4f}, estimated: {estimate_acc(acc_zs_cot_prompt_pred, data_id+1, total_samples)}")
    return metric_counts

def process_cot_decoding_metrics(config, cot_decoding_pred:str, gt:int, data_id:int, metric_counts:dict, total_samples:int):
    """Process metrics and logging for the results"""
    def estimate_acc(acc, num, total=1319):
        return (acc /num) * total
    if np.array(gt) == np.array(cot_decoding_pred):
        metric_counts["cot_decoding_pred"] = metric_counts.get("cot_decoding_pred", 0) + 1
    acc_cot_decoding_pred = metric_counts["cot_decoding_pred"] / total_samples
    logging.info(f"Iteration {data_id+1} / {total_samples}- True answer: {gt}")
    logging.info(f"Iteration {data_id+1} / {total_samples} - Cot decoding prediction: {cot_decoding_pred}")
    logging.info(f"Iteration {data_id+1} / {total_samples} - Cot decoding prediction accuracy: {acc_cot_decoding_pred:.4f}, estimated: {estimate_acc(acc_cot_decoding_pred, data_id+1, total_samples)}")
    return metric_counts

def process_leaves_metrics(config, leaves:list, gt:int, data_id:int, metric_counts:dict, total_samples:int):
    """Process metrics and logging for the results"""
    def estimate_acc(acc, num, total=1319):
        return (acc /num) * total
    
    pred_metrics = defaultdict(lambda: defaultdict(float))
    pred_counts = defaultdict(int)

    for leaf in leaves:
        pred = leaf.pred
        for metric in config.ALL_METRICS[:-1]:
            pred_metrics[pred][metric] += leaf.metrics_dict[metric]
            pred_counts[pred] += 1
            pred_metrics[pred]['Voting'] = pred_counts[pred]
    # 评估结果
    metric_predictions = {}
    for metric in config.ALL_METRICS:
        if metric in config.INCREASING_METRICS:
            best_pred = max(pred_metrics, key = lambda x: pred_metrics[x][metric])
        else:
            best_pred = min(pred_metrics, key = lambda x: pred_metrics[x][metric])

        metric_predictions[metric] = {
                                    'best_pred': best_pred,
                                    'best_value': pred_metrics[best_pred][metric],
                                    'is_best': best_pred == gt
                                    }

        if np.array(float(gt)) == np.array(float(metric_predictions[metric]['best_pred'])):
            metric_counts["search_pred"][metric] = metric_counts["search_pred"].get(metric, 0) + 1

        acc_pred = metric_counts["search_pred"][metric] / total_samples
        print(f"Iteration {data_id+1}  / {total_samples}- True answer: {gt}")
        logging.info(f"Iteration {data_id+1}  / {total_samples}- True answer: {gt}")
        print(f"Iteration {data_id+1}  / {total_samples}- {metric} predictions: {metric_predictions[metric]}")
        logging.info(f"Iteration {data_id+1}  / {total_samples}- {metric} predictions: {metric_predictions[metric]}")
        print(f"Iteration {data_id+1}  / {total_samples} - {metric} accuracy of Pred: {acc_pred:.4f}, estimated: {estimate_acc(acc_pred, data_id+1, total_samples)}")
        logging.info(f"Iteration {data_id+1}  / {total_samples} - {metric} accuracy of Pred: {acc_pred:.4f}, estimated: {estimate_acc(acc_pred, data_id+1, total_samples)}")
    return metric_counts

def process_metrics(config, gt:int, data_id:int, metric_counts:dict, total_samples:int, 
                    leaves:list, 
                    greedy_pred:str,
                    cot_decoding_pred:str,
                    zs_cot_prompt_pred:str):
    """Process metrics and logging for the results"""
    if leaves:
        metric_counts = process_leaves_metrics(config, leaves, gt, data_id, metric_counts, total_samples)
    if greedy_pred:
        metric_counts = process_greedy_metrics(config, greedy_pred, gt, data_id, metric_counts, total_samples)
    if cot_decoding_pred:
        metric_counts = process_cot_decoding_metrics(config, cot_decoding_pred, gt, data_id, metric_counts, total_samples)
    if zs_cot_prompt_pred:
        metric_counts = process_zs_cot_prompt_metrics(config, zs_cot_prompt_pred, gt, data_id, metric_counts, total_samples)
    return metric_counts


def save_leaves(leaves, gt, data_id, config):
    """Save the leaves to file"""
    
    leaves_path = os.path.join(config.LEAVES_DIR, config.model, config.dataset, f"{config.prober_type}_{config.rep}_{str(config.score_layers)}", f"{data_id}.json")
    os.makedirs(os.path.dirname(leaves_path), exist_ok=True)
    leaves_info = []
    for leaf in leaves:
        leaves_info.append({
            "text": leaf.text,
            "pred": leaf.pred,
            "gt": gt,
            "metrics": leaf.metrics_dict,
            "full_text": leaf.full_text,
            "scores": leaf.get_sequence_score()
        })
    with open(leaves_path, 'w') as f:
        json.dump(leaves_info, f)

# ------------------ Main Function ------------------
def main(config):
    model, tokenizer = initialize_model_and_tokenizer(config)
    """Main execution function"""
    setup_logging(config)
    print(f"Starting search with exp_name: {config.exp_name}")
    logging.info(f"Starting search with config: {config}")
    logging.info(f"The methods are {config.methods}")

    config.prober_input_dim = model.config.hidden_size
    config.prober_num_layers = model.config.num_hidden_layers

    if "search" in config.methods:
        prober = initialize_prober(config)
        score_fn = prober.score
    else:
        score_fn = None

    metric_counts = {}
    for method in config.methods:
        if method == "search":
            metric_counts.update({"search_pred":{metric: 0 for metric in config.ALL_METRICS}})
        elif method == "greedy":
            metric_counts.update({"greedy_pred":0})
        elif method == "cot_decoding":
            metric_counts.update({"cot_decoding_pred":0})
        elif method == "zs_cot_prompt":
            metric_counts.update({"zs_cot_prompt_pred":0})

    # checkpoint = load_checkpoint(config)
    checkpoint = None

    if checkpoint:
        print(f"Resuming from data_id: {checkpoint['data_id']}")
        logging.info(f"Resuming from data_id: {checkpoint['data_id']}")
        start_data_id = checkpoint['data_id']
        metric_counts = checkpoint['metric_counts']
        logging.info(f"Resuming from data_id: {start_data_id}")
    else:
        print("Starting from scratch")
        logging.info("Starting from scratch")
        start_data_id = 0

    try:
        data_loader = setup_data_loader(config)
        for data_id, data in tqdm(enumerate(data_loader)):
            if data_id <start_data_id:
                continue
            x, y = data   
            gt = y[0].strip()
            gt = gt.replace(',', '')
            gt = float(gt)
            logging.info(f"**************************Processing data {data_id} / {len(data_loader)}********************************")
            tree_leaves, greedy_pred, cot_decoding_pred, zs_cot_prompt_pred = process_single_example(
                data, model, tokenizer, score_fn, config
            )

            save_leaves(tree_leaves,gt, data_id, config)
            logging.info(f"**************************Save leaves of {data_id} / {len(data_loader)}********************************")
            # metric_counts = process_metrics(config, leaves, gt, data_id, metric_counts, total_samples=len(data_loader), greedy_pred=greedy_pred)
            metric_counts = process_metrics(config, gt, data_id, metric_counts, total_samples=len(data_loader), 
                                            leaves = tree_leaves, 
                                            greedy_pred = greedy_pred,
                                            cot_decoding_pred = cot_decoding_pred,
                                            zs_cot_prompt_pred = zs_cot_prompt_pred)
            
            # if data_id % config.checkpoint_interval == 0:
            #     save_checkpoint(data_id, metric_counts, config)
    except Exception as e:
        logging.error(f"Error at data_id {data_id}: {e}")
        # save_checkpoint(data_id, metric_counts, config)
        raise e

# ------------------ Configuration ------------------
if __name__ == "__main__":
    from config import Config

    config = Config(
    model = "mist-7b", #"mist-7b", "gemma-2-2b", "llama2-7b", "llama-3.1-8B", "llama-3.2-3B","phi-1.5"

    dataset = "gsm8k",   # "gsm8k", "multiarith", "svamp", MAWPS  "addsub", "singleeq", "aqua", "commonsensqa",  "strategyqa", 
                                                        # "bigbench_date", "object_tracking", "coin_flip", "last_letters"
    device = "cuda:0",

    NODE_LENGTH = [1]*1 + [30]*9,
    # mist-7b: [1]*1 + [30]*9
    # gemma2-2b: [1]*1 + [50]*5
    # phi-1.5: [1]*1 + [30]*9

    min_step = 6,
    # mist-7b: 5
    # gemma2-2b: 4
    # phi-1.5: 6

    batch_size = 3,
    # mist-7b: 3
    # gemma2-2b: 3
    # phi-1.5: 20

    methods = ["search"], #["search","cot_decoding", "greedy", "zs_cot_prompt"]

    root_path="/hdd/zijianwang/Thoughtprobe",

    MODEL_CACHE_DIR="/hdd/zijianwang/HF_CACHE",
    
    prober_type = "LR_ps",

    rep = "hiddens",  #["mlp", "hiddens", "post-attn"]

    score_layers =[30, 29, 28],

    )

    config.LEAVES_DIR = os.path.join(config.root_path, f"Leaves_new")
    config.LOG_DIR = os.path.join(config.root_path, "Log_leaves")
    os.makedirs(config.LOG_DIR, exist_ok = True)
    os.makedirs(config.LEAVES_DIR, exist_ok = True)
    main(config)