# data_utils.py
import json
import os
from datetime import datetime
import logging
from collections import defaultdict
from statistics import mean
import random
from torch.utils.data import Dataset
import torch
def setup_logging(config):
    """Configure logging settings
    
    Args:
        config: Configuration object containing logging settings
    """
    log_filename = os.path.join(
        config.LOG_DIR,
        f"decoding2_{config.EXP_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(checkpoint_data, checkpoint_dir):
    """Save checkpoint to file
    
    Args:
        checkpoint_data: Data to save
        checkpoint_dir: Directory to save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "svm4.json")
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)

def load_checkpoint(checkpoint_dir):
    """Load latest checkpoint
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        dict or None: Loaded checkpoint data if exists, else None
    """
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoint_files = sorted([
        f for f in os.listdir(checkpoint_dir) 
        if f.startswith('svm4')
    ])
    
    if not checkpoint_files:
        return None
        
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    with open(latest_checkpoint, 'r') as f:
        return json.load(f)
    
def shuffleDict(d):
  keys = list(d.keys())
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  keys = [(key, d[key]) for key in keys]
  #keys = d(keys)
  return dict(keys) 

def data_reader(args):

    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif args.dataset == "gsm8k":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif args.dataset == "commonsensqa":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "strategyqa":
      with open(args.dataset_path) as f:
        json_data = json.load(f)["examples"]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "svamp":
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            
    elif args.dataset in ("bigbench_date", "object_tracking"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if args.dataset == "bigbench_date":
            choice_index = ['A','B','C','D','E','F']
        elif args.dataset in ("object_tracking"):
            choice_index = ['A','B','C']
        else:
            raise ValueError("dataset is not properly defined ...")
        for line in json_data:
          q = line["input"].strip()
          if args.dataset == "bigbench_date":
              choice = "Answer Choices:"
              # Randomly shuffle the answer choice dictionary because the original answer is always A ...
              choice_dic = shuffleDict(line["target_scores"])
          elif args.dataset == "object_tracking":
              choice = "\nWhich choice is true ? Answer Choices:"
              choice_dic = line["target_scores"]
          else:
              raise ValueError("dataset is not properly defined ...")
          for i, key_value in enumerate(choice_dic.items()):
              key, value = key_value
              choice += " ("
              choice += choice_index[i]
              choice += ") "
              choice += key
              if value == 1:
                  a = choice_index[i]
                  #a = key
          q = q + " " + choice
          questions.append(q)
          answers.append(a)            
          
    elif args.dataset in ("coin_flip", "last_letters"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers


class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def setup_data_loader(args):
    dataset = MyDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset,
                shuffle=False,
                batch_size=1)
    return dataloader

if __name__ == "__main__":
    import dataclasses
    @dataclasses.dataclass
    class Args:
        dataset = "aqua"
        dataset_path = "/ssd/jameswang/CoT-decoding/dataset/AQuA/test.json"
    args = Args()
    dataset = MyDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset,
                shuffle=False,
                batch_size=1)
    for i, data in enumerate(dataloader):
        x, y = data
        print('*************************')
        print("{}st data".format(i+1))
        print(x)
        print(y)    
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
  

def extract_after(a, b):
    """
    提取字符串 b 中从 a 之后的部分。
    
    参数:
    - a: 字符串，作为起始部分
    - b: 字符串，从中提取 a 之后的部分
    
    返回:
    - b 中从 a 之后的部分，如果 b 不以 a 开头，返回 None
    """
    if b.startswith(a):
        result = b[len(a):]  # 提取从 a 的长度到 b 末尾的部分
        return result[1:] if result.startswith(",") else result  # 去掉多余的逗号
    return []

def save_to_json(file_path, data):
    if os.path.exists(file_path):
        # Load existing data and append
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # Append new data
    existing_data.append(data)

    # Save back to JSON
    with open(file_path, 'w') as file:
        json.dump(existing_data, file)
