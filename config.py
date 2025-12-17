from dataclasses import dataclass, field
import os

@dataclass
class Config:

    # Model and Dataset
    model: str = "mist-7b"  #"mist-7b", "gemma-2-2b", "llama2-7b"
    device: str = "cuda:1"
    dataset: str = "multiarith"  # "gsm8k", "addsub", "multiarith", "svamp", "singleeq", "aqua", "commonsensqa", "addsub", "multiarith", "strategyqa", 
                                                     # "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"
    methods: list = field(default_factory=lambda: ["search"])   #cot_decoding, search, greedy, zs_cot_prompt

    MODEL_CACHE_DIR: str = ""
    
    # Other parameters
    root_path: str = "/hdd/zijianwang/Thoughtprobe"
    model_name: str = None
    dataset_path: str = None
    direct_answer_trigger: str = None

    ALL_METRICS: list = field(default_factory=lambda: [
        # 'Range', 'Standard Deviation',  'Coefficient of Variation',
        #     'Max Step Change', 'RMS Fluctuation','Kendall Tau', 
        'Variance',
        'First Score', 'Last Score', 'Mean', 'Increase Ratio', 'Voting', 
    ])
    INCREASING_METRICS: list = field(default_factory=lambda: [
         'Increase Ratio', 'First Score', 'Last Score', 'Mean','Voting'
    ])
    
    # Search Parameters
    NODE_LENGTH: list = field(default_factory=lambda: [1]*1 + [30] * 9)
    MAX_LAYERS: int = 9
    num_samples: int = 3
    top_k: int = 10
    temperature: float = 0.015
    sample_steps: int = 4
    batch_size: int = 3
    chunk: bool = False
    min_step: int = 5

    # Prober
    prober_type: str = "SVM"
    prober_input_dim: int = 4096
    prober_num_layers: int = 32
    score_layers:list  = field(default_factory=lambda: [31, 30, 29, 28, 27, 26, 25, 24, 23, 22])
    rep: str = "mlp"


    exp_name: str = None
    checkpoint_interval: int = 1
    use_gpt: bool = False

    def __post_init__(self):
        # Update any parameters passed through initialization
        # Set model_name based on model
        if self.model == "llama2-7b":
            self.model_name = "meta-llama/Llama-2-7b-hf"
        elif self.model == "mist-7b":
            self.model_name = "mistralai/Mistral-7B-v0.1"
        elif self.model == "gemma-2-2b":
            self.model_name = "google/gemma-2-2b"
        elif self.model == "llama-3.1-8B":
            self.model_name = "meta-llama/Llama-3.1-8B"
        elif self.model == "llama-3.2-3B":
            self.model_name = "meta-llama/Llama-3.2-3B"
        elif self.model == "phi-1.5":
            self.model_name = "microsoft/phi-1_5"
        elif self.model == "dpsk_distill":
            self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        elif self.model == "qwen3-8b":
            self.model_name = "Qwen/Qwen3-8B"

        # Set dataset_path and direct_answer_trigger based on dataset
        if self.dataset == "gsm8k":
            self.dataset_path = os.path.join(self.root_path, "dataset/grade-school-math/test.json")
            self.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        elif self.dataset == "multiarith":
            self.dataset_path = os.path.join(self.root_path, "dataset/MultiArith/test.json")
            self.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        elif self.dataset == "svamp":
            self.dataset_path = os.path.join(self.root_path, "dataset/SVAMP/test.json")
            self.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
            
        elif self.dataset == "addsub":
            self.dataset_path = os.path.join(self.root_path, "dataset/AddSub/AddSub.json")
            self.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        elif self.dataset == "singleeq":
            self.dataset_path = os.path.join(self.root_path, "dataset/SingleEq/questions.json")
            self.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        elif self.dataset == "MAWPS":
            self.dataset_path = os.path.join(self.root_path, "dataset/MAWPS/train.json")
            self.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        elif self.dataset == "aime":
            self.dataset_path = os.path.join(self.root_path, "dataset/AIME/AIME_Dataset_1983_2024.csv")
            self.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        # Set pari_data_save_path
        self.pari_data_save_path = os.path.join(self.root_path, "Pari_data_test", self.model, f"{self.dataset}_CoTdecoding.json")
        
        # Set exp_name
        self.exp_name = f"{self.model}_{self.prober_type}_{self.dataset}_[{','.join(self.methods)}]"
        
        # Set MAX_LAYERS
        self.MAX_LAYERS = len(self.NODE_LENGTH) - 1

        self.LOG_DIR: str = os.path.join(self.root_path, "_Log")
        # self.CHECKPOINT_DIR: str = os.path.join(self.LOG_DIR, "ckpt")
        self.CHECKPOINT_DIR: str = os.path.join(self.root_path, "ckpt")
        
        # Set prober_base_path
        self.prober_base_path = os.path.join(self.root_path, "Probe_weights", self.model, self.dataset)

if __name__ == "__main__":
    config = Config(
    model="llama2-7b", #"mist-7b", "gemma-2-2b", "llama2-7b"

    dataset="gsm8k",   # "gsm8k", "addsub", "multiarith", "svamp", "singleeq", "aqua", "commonsensqa", "addsub", "multiarith", "strategyqa", 
                                                        # "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"
    device="cuda:0",

    methods=["search_svm","cot_decoding", "greedy"], #["search_svm","cot_decoding", "greedy", "zs_cot_prompt"]
    
    root_path="/hdd/zijianwang/CoT-activation",

    MODEL_CACHE_DIR="/hdd/.cache/huggingface/hub",

    prober_type = "SVM",
    rep = "mlp",
    NODE_LENGTH = [5]*1 + [30] * 9,

    score_layers = [24, 23, 22, 21,20, 19, 18,17, 16]  # [31, 30, 29, 28, 27, 26, 25, 24, 23, 22]  
    )

    direct_answer_trigger = "The answer is"

    # pred = "Tom is 10 years old. His sister is 3 years younger than him. How old is his sister? The answer is 7 years old."
    # preds = pred.split(direct_answer_trigger)
    # answer_flag = True if len(preds) > 1 else False 
    # pred = preds[-1]
    # print(preds)
    # print(pred)

    config.current_method = "search_svm"
    config.__post_init__()
    log_filename = os.path.join(config.LOG_DIR, "test.log")
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logging(log_filename)
    logging.info(config)