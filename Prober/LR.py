import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import List, Dict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 使用
set_seed(42)
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, device="cuda:0"):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1).to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        return torch.sigmoid(self.linear(x))  # 使用 sigmoid 函数返回概率值

    def score(self, x):
        x = x.to(self.device)
        return self.linear(x)

    def save_model(self, path):
        """Save model weights to path"""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model weights from path"""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError(f"No model found at {path}")


class MultiLayerLogisticRegression:
    def __init__(
        self,
        input_dim: int = 4096,
        num_layers: int = 32,
        base_save_path: str = "/hdd/zijianwang/CoT-decoding/Probe_weights/mist-7b/LogisticRegression/",
        device="cuda:0",
    ):
        """
        初始化多层 Logistic Regression 分类器

        Args:
            input_dim: 输入特征维度
            num_layers: 总层数
            base_save_path: 模型保存的基础路径
        """
        self.num_layers = num_layers
        self.base_save_path = base_save_path
        self.models = {}
        self.input_dim = input_dim
        self.device = device

        # 创建保存目录
        os.makedirs(base_save_path, exist_ok=True)

    def _get_model_path(self, layer_id: int) -> str:
        """获取特定层的模型保存路径"""
        return os.path.join(self.base_save_path, f"logreg_model_layer_{layer_id}.pkl")

    def train_logistic_regression(
        self, X, y, learning_rate=0.01, num_epochs=200, model_save_path=None, device="cuda:0"
    ):
        X = X.to(device)
        y = y.to(device)

        # 初始化模型
        model = LogisticRegression(X.shape[1], device=device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()  # 使用二分类交叉熵损失

        for epoch in range(num_epochs):
            # 前向传播
            output = model(X).squeeze()

            # 计算损失
            loss = criterion(output, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # 如果提供了保存路径，则保存模型
        if model_save_path:
            model.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        return model

    def train_all_layers(
        self,
        pos_act: List[torch.Tensor],
        neg_act: List[torch.Tensor],
        token_id_pos: np.ndarray,
        token_id_neg,
        learning_rate: float = 0.01,
        num_epochs: int = 200,
        test_size: float = 0.1,
    ) -> Dict[int, float]:
        """
        训练所有层的 Logistic Regression 模型，包含训练集和测试集的划分

        Args:
            pos_act: 正样本激活值列表
            neg_act: 负样本激活值列表
            token_id_pos: 正样本的token位置
            token_id_neg: 负样本的token位置
            learning_rate: 学习率
            num_epochs: 训练轮数
            test_size: 测试集比例

        Returns:
            Dict[int, Dict[str, float]]: 每一层的训练集和测试集准确率字典
        """
        accuracies = {}
        for layer_id in range(self.num_layers):
            print(f"\nTraining layer {layer_id}")
            # 提取该层的训练数据
            pos_layer_features = torch.stack(
                [
                    i[tid][layer_id]
                    for i in pos_act
                    for tid in token_id_pos
                    if tid < i.shape[0]
                ]
            )  # (num_pos_samples, input_dim)
            neg_layer_features = torch.stack(
                [
                    i[tid][layer_id]
                    for i in neg_act
                    for tid in token_id_neg
                    if tid < i.shape[0]
                ]
            )  # (num_neg_samples, input_dim)

            # 合并该层的正负样本
            X = torch.cat([pos_layer_features, neg_layer_features], dim=0)
            y = torch.cat(
                [torch.ones(len(pos_layer_features)), torch.zeros(len(neg_layer_features))]
            )
            X = X.to(self.device)
            y = y.to(self.device)
            # 随机打乱数据
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]

            # 划分训练集和测试集
            test_size_samples = int(len(X) * test_size)
            X_train = X[:-test_size_samples]
            y_train = y[:-test_size_samples]
            X_test = X[-test_size_samples:]
            y_test = y[-test_size_samples:]

            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

            # 训练该层模型
            model_path = self._get_model_path(layer_id)
            model = self.train_logistic_regression(
                X_train, y_train, learning_rate, num_epochs, model_path, self.device
            )

            # 计算训练集准确率
            with torch.no_grad():
                train_pred = model(X_train).squeeze()
                train_pred = (train_pred > 0.5).float()  # 概率大于 0.5 则为正类
                train_accuracy = (train_pred == y_train).float().mean().item()

                # 计算测试集准确率
                test_pred = model(X_test).squeeze()
                test_pred = (test_pred > 0.5).float()
                test_accuracy = (test_pred == y_test).float().mean().item()

            self.models[layer_id] = model
            accuracies[layer_id] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }

            print(f"Layer {layer_id} results:")
            print(f"Train accuracy: {train_accuracy:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")

        return accuracies

    def load_model(self, layer_id: int):
        """加载特定层的模型"""
        model_path = self._get_model_path(layer_id)
        model = LogisticRegression(self.input_dim)
        model.load_model(model_path)
        self.models[layer_id] = model
        return model

    def load_all_models(self):
        """加载所有层的模型"""
        for layer_id in range(self.num_layers):
            self.load_model(layer_id)

    def predict(self, X: torch.Tensor, layer_id: int) -> torch.Tensor:
        """使用特定层的模型进行预测"""
        if layer_id not in self.models:
            self.load_model(layer_id)
        with torch.no_grad():
            pred = self.models[layer_id](X).squeeze()
            pred = (pred > 0.5).float()  # 概率大于 0.5 则为正类
        return pred

    def score(self, X):
        """对多层输入特征进行预测"""
        probs = []
        bsz, token_length, num_layers, dim = X.shape
        for layer_id in range(num_layers):
            new_model = self.models[layer_id]
            inp = X[:, :, layer_id, :]
            bsz, token_length, dim = inp.shape
            inp = inp.view(-1, dim).to("cpu").to(torch.float32)
            prob = new_model.score(inp)  # (bsz*length_token_sequence, 1)
            prob = prob.to("cpu")
            prob = prob.reshape(bsz, token_length, 1)
            probs.append(prob.tolist())
        probs = torch.tensor(probs)  # (num_layers, bsz, token_length, 1)
        probs = probs.permute(1, 2, 0, 3)  # (bsz, token_length, num_layers, 1)
        probs = probs.squeeze(-1)  # (bsz, token_length, num_layers)
        return probs
    

    def test_performance(self, pos_act, neg_act):
        """
        Test the performance of SVM models across all layers
        
        Args:
            pos_act: List of positive activations
            neg_act: List of negative activations
            top_k: Number of top performing layers to display
            
        Returns:
            dict: Dictionary containing metrics for all layers
            list: List of top-k layer information sorted by ROC-AUC
        """
        pos_token_id_pos = np.arange(min([i.shape[0] for i in pos_act]), 
                                max([i.shape[0] for i in pos_act]) - 1, step=1)
        neg_token_id_pos = np.arange(min([i.shape[0] for i in neg_act]), 
                                max([i.shape[0] for i in neg_act]) - 1, step=1)

        layer_metrics = {}

        for layer_id in range(self.num_layers):
            pos_features = torch.stack([
                i[tid][layer_id]
                for i in pos_act
                for tid in pos_token_id_pos
                if tid < i.shape[0]
            ])
            neg_features = torch.stack([
                i[tid][layer_id]
                for i in neg_act
                for tid in neg_token_id_pos
                if tid < i.shape[0]
            ])

            X1_val = pos_features[:]
            X2_val = neg_features[:]
            X_val = torch.cat([X1_val, X2_val], dim=0).to(self.device)
            y_val = torch.cat([torch.ones(len(X1_val)), 
                            torch.zeros(len(X2_val))]).to(self.device)
            
            predictions = self.predict(X_val, layer_id = layer_id)
            
            predictions = predictions.cpu().numpy()
            y_val = y_val.cpu().numpy()

            # 计算整体指标
            acc = (predictions == y_val).mean()
            roc_auc = roc_auc_score(y_val, predictions)
            
            try:
                precision = precision_score(y_val, predictions, zero_division=0)
                recall = recall_score(y_val, predictions, zero_division=0)
                f1 = f1_score(y_val, predictions, zero_division=0)
                roc_auc = roc_auc_score(y_val, predictions)
            except:
                precision = recall = f1 = roc_auc = 0.0

            # 分别计算正样本和负样本的预测结果
            pos_predictions = predictions[:len(X1_val)]
            neg_predictions = predictions[len(X1_val):]

            # 计算正样本指标
            pos_acc = (pos_predictions == 1).mean()
            try:
                pos_y = np.ones(len(X1_val))
                pos_precision = precision_score(pos_y, pos_predictions, zero_division=0)
                pos_recall = recall_score(pos_y, pos_predictions, zero_division=0)
                pos_f1 = f1_score(pos_y, pos_predictions, zero_division=0)
            except:
                pos_precision = pos_recall = pos_f1 = 0.0

            # 计算负样本指标
            neg_acc = (neg_predictions == 0).mean()
            try:
                neg_y = np.zeros(len(X2_val))
                neg_precision = precision_score(neg_y, neg_predictions, zero_division=0, pos_label=0)
                neg_recall = recall_score(neg_y, neg_predictions, zero_division=0, pos_label=0)
                neg_f1 = f1_score(neg_y, neg_predictions, zero_division=0, pos_label=0)
            except:
                neg_precision = neg_recall = neg_f1 = 0.0

            layer_metrics[layer_id] = {
                # 整体指标
                "accuracy": acc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                
                # 正样本指标
                "pos_acc": pos_acc,
                "pos_precision": pos_precision,
                "pos_recall": pos_recall,
                "pos_f1": pos_f1,
                
                # 负样本指标
                "neg_acc": neg_acc,
                "neg_precision": neg_precision,
                "neg_recall": neg_recall,
                "neg_f1": neg_f1
            }

        # top_k_layers = sorted(layer_metrics.items(), 
        #                     key=lambda x: x[1]["roc_auc"], 
        #                     reverse=True)[:top_k]

        return layer_metrics
    
    def calculate_layer_scores(self, pos_act, neg_act, layer_indices=None):
        """
        计算每一层的正负样本得分
        
        Args:
            pos_act: 正样本激活值列表
            neg_act: 负样本激活值列表
            layer_indices: 指定要计算的层索引列表，默认为None表示计算所有层
            
        Returns:
            dict: 包含每层得分的字典
        """
        # 如果未指定层索引，则使用所有层
        if layer_indices is None:
            layer_indices = range(self.num_layers)
            
        # 初始化每层的得分列表
        layer_scores = {i: {'pos_scores': [], 'neg_scores': [], 'pos_scores_mean': [], 'neg_scores_mean': []} for i in layer_indices}

        # 计算正样本每层得分
        for pos_sample in pos_act:
            scores = self.score(pos_sample.unsqueeze(0))  # (1, seq_len, num_layers)
            for layer_id in layer_indices:
                layer_score_mean= scores[:, :, layer_id].mean().item()
                layer_scores[layer_id]['pos_scores'].append(scores)
                layer_scores[layer_id]['pos_scores_mean'].append(layer_score_mean)


        # 计算负样本每层得分
        for neg_sample in neg_act:
            scores = self.score(neg_sample.unsqueeze(0))  # (1, seq_len, num_layers)
            for layer_id in layer_indices:
                layer_score_mean = scores[:, :, layer_id].mean().item()
                layer_scores[layer_id]['neg_scores'].append(scores)
                layer_scores[layer_id]['neg_scores_mean'].append(layer_score_mean)
        # return layer_scores

        # 计算每层的平均分数和差异
        results = {}
        for layer_id in layer_indices:
            pos_avg = sum(layer_scores[layer_id]['pos_scores_mean']) / len(layer_scores[layer_id]['pos_scores_mean'])
            neg_avg = sum(layer_scores[layer_id]['neg_scores_mean']) / len(layer_scores[layer_id]['neg_scores_mean'])
            diff = pos_avg - neg_avg
            
            results[layer_id] = {
                'pos_avg': pos_avg,
                'neg_avg': neg_avg,
                'diff': diff
            }
        
        return results,  layer_scores
