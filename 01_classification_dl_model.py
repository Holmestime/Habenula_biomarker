# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from scipy.io import savemat
from torchsummary import summary
from sklearn.preprocessing import StandardScaler


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 定义计算 Specificity 和 Sensitivity 的函数
def calculate_specificity_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity


def load_data():

    """

    :return:
    """
    file_path = f"./data/dl_data.mat"
    raw_data = loadmat(file_path)
    data = raw_data['fea']
    label = raw_data['label']
    label = np.squeeze(label)
    print(f"data: {data.shape}, label: {label.shape}")
    return data, label


# 定义 Z-score 归一化函数
def zscore_normalize(train_data, val_data):
    """
    对训练数据和验证数据进行 Z-score 归一化。
    参数:
        train_data: (num_train_samples, seq_len, num_features)
        val_data: (num_val_samples, seq_len, num_features)
    返回:
        归一化后的训练数据和验证数据。
    """
    scaler = StandardScaler()
    num_train_samples, seq_len, num_features = train_data.shape
    num_val_samples, _, _ = val_data.shape

    # 将训练数据 reshape 为 (num_train_samples * seq_len, num_features)
    train_reshaped = train_data.reshape(-1, num_features)
    # 对训练数据进行归一化
    train_normalized = scaler.fit_transform(train_reshaped)
    # 将归一化的训练数据 reshape 回原始形状
    train_normalized = train_normalized.reshape(num_train_samples, seq_len, num_features)

    # 对验证数据使用训练数据的均值和标准差进行归一化
    val_reshaped = val_data.reshape(-1, num_features)
    val_normalized = scaler.transform(val_reshaped)
    val_normalized = val_normalized.reshape(num_val_samples, seq_len, num_features)

    return train_normalized, val_normalized


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embedding_size):
        super(SimpleLSTM, self).__init__()
        self.embedding = None
        self.hidden_size = hidden_size

        if embedding_size is not None:
            self.embedding = nn.Linear(input_size, embedding_size)
            self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # 取最后一个时间步的输出
        return out


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embedding_size):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = None
        if embedding_size is not None:
            self.embedding = nn.Linear(input_size, embedding_size)
            self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        else:
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        h_gru, _ = self.gru(x)
        out = self.fc(h_gru[:, -1, :])  # 取最后一个时间步的输出
        return out


def feature_aug(raw_data):
    """

    :param raw_data: (51 * 237 * 24)
    :return:
    """
    # 防止除零的小常数（根据数据特性调整）
    EPSILON = 1e-8

    new_features = []
    n_features = raw_data.shape[2]

    # 生成所有唯一的特征组合
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # 计算特征i除以特征j（带防零处理）
            ratio = raw_data[:, :, i] / (raw_data[:, :, j] + EPSILON)
            new_features.append(ratio)

    # 将列表转换为numpy数组并合并维度
    new_features_array = np.stack(new_features, axis=2)

    # 合并原始数据和新特征
    augmented_data = np.concatenate([raw_data, new_features_array], axis=2)

    print("原始数据形状：", raw_data.shape)
    print("增强后数据形状：", augmented_data.shape)
    return augmented_data


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, num_classes, embedding_size):
        super(TransformerModel, self).__init__()
        self.embedding = None

        if embedding_size is not None:
            self.embedding = nn.Linear(input_size, embedding_size)  # 将输入嵌入到 hidden_size 维度

            self.linear = nn.Linear(embedding_size, hidden_size)
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
                num_layers=num_layers
            )
            self.fc = nn.Linear(hidden_size, num_classes)  # 全连接层输出分类结果
        else:
            self.linear = nn.Linear(input_size, hidden_size)

            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
                num_layers=num_layers
            )
            self.fc = nn.Linear(hidden_size, num_classes)  # 全连接层输出分类结果

    def forward(self, x):
        if self.embedding is not None:
            # 输入形状: (batch_size, seq_len, input_size)
            x = self.embedding(x)  # (batch_size, seq_len, hidden_size)

        x = self.linear(x) # (batch_size, seq_len, hidden_size)
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, hidden_size) 作为 Transformer 输入
        x = self.transformer_encoder(x)  # 输出形状: (seq_len, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # 转换回 (batch_size, seq_len, hidden_size)
        out = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return out


def choose_model(model_name, embedding_size=None, aug_flag=False):
    """

    :param model_name:
    :param embedding_size:
    :param aug_flag:
    :return:
    """
    # 超参数
    if not aug_flag:
        input_size = 24
    else:
        input_size = 300
    hidden_size = 16
    num_classes = 2
    # num_epochs = 100
    # batch_size = 8

    # transformer specific
    num_heads = 4  # 自注意力头的数量
    num_layers = 2  # Transformer 的编码器层数

    model = None
    if model_name == "lstm":
        model = SimpleLSTM(input_size, hidden_size, num_classes,embedding_size)
    elif model_name == "gru":
        model = SimpleGRU(input_size, hidden_size, num_classes,embedding_size)
    elif model_name == "trans":
        model = TransformerModel(input_size, num_heads, hidden_size, num_layers, num_classes,embedding_size)

    return model


def get_performance(model_name, embedding_size=None, aug_flag=False, device_id=0):
    seed = 42
    set_random_seeds(seed)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    accuracy_list = []
    spec_list = []
    sen_list = []
    f1_list = []
    auc_list = []

    n_iter = 200
    num_epochs = 100
    batch_size = 8
    # embedding_size = None

    X, y = load_data()

    if aug_flag:
        X = feature_aug(X)


    # 将数据转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')

    for iii in range(n_iter):
        # 交叉验证循环
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f'Fold {fold + 1}')

            X_train, X_val = X[train_idx].numpy(), X[val_idx].numpy()
            y_train, y_val = y[train_idx], y[val_idx]
            X_train_normalized, X_val_normalized = zscore_normalize(X_train, X_val)
            # 将归一化后的数据转换为 PyTorch 张量
            X_train_normalized = torch.tensor(X_train_normalized, dtype=torch.float32)
            X_val_normalized = torch.tensor(X_val_normalized, dtype=torch.float32)

            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_normalized, y_train),
                                                       batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_normalized, y_val),
                                                     batch_size=batch_size, shuffle=False)



            # 初始化模型、损失函数和优化器
            model = choose_model(model_name, embedding_size=embedding_size, aug_flag=aug_flag)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 训练模型
            for epoch in range(num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

            # 验证模型
            model.eval()
            y_true, y_pred, y_score = [], [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.extend(batch_y.detach().cpu().numpy())
                    y_pred.extend(predicted.detach().cpu().numpy())
                    y_score.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())  # 获取正类的概率

            # 计算指标
            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            specificity, sensitivity = calculate_specificity_sensitivity(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_score)

            # 输出结果
            print(f'Validation Accuracy: {accuracy:.2f}')
            print(f'Validation Specificity: {specificity:.2f}')
            print(f'Validation Sensitivity: {sensitivity:.2f}')
            print(f'Validation F1 Score: {f1:.2f}')
            print(f'Validation AUC: {auc:.2f}')

            accuracy_list.append(accuracy)
            spec_list.append(specificity)
            sen_list.append(sensitivity)
            f1_list.append(f1)
            auc_list.append(auc)

    accuracy_list = np.array(accuracy_list)
    spec_list = np.array(spec_list)
    sen_list = np.array(sen_list)
    f1_list = np.array(f1_list)
    auc_list = np.array(auc_list)

    performance = {
        'acc': accuracy_list,
        'spec': spec_list,
        'sen': sen_list,
        'f1': f1_list,
        'auc': auc_list
    }

    savemat(f"./data/{model_name}_performance_embed_{embedding_size}_aug_{aug_flag}.mat", performance)

    print(f"5-fold\n"
          f"Accuracy: {np.mean(accuracy_list):.3f}, +- std: {np.std(accuracy_list):.3f}\n"
          f"spec_list: {np.mean(spec_list):.3f}, +- std: {np.std(spec_list):.3f}\n"
          f"sen_list: {np.mean(sen_list):.3f}, +- std: {np.std(sen_list):.3f}\n"
          f"f1_list: {np.mean(f1_list):.3f}, +- std: {np.std(f1_list):.3f}\n"
          f"auc_list: {np.mean(auc_list):.3f}, +- std: {np.std(auc_list):.3f}\n"
          )

def test_feature_aug():
    """

    :return:
    """
    x = np.random.rand(51, 237, 24)
    feature_aug(x)


if __name__ == '__main__':
    # model_name_list = ['lstm','gru','trans']
    device_id = 2
    model_name_list = ['trans']

    for c_name in model_name_list:
        for c_embedding in [None, 300]:
            for c_aug in [False, True]:
                if c_embedding is not None and c_aug:
                    continue
                print(f"{c_name}, embedding, {c_embedding}, aug {c_aug}")
                get_performance(c_name, c_embedding, c_aug, device_id)
