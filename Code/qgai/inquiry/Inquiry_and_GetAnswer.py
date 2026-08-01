import os
from pathlib import Path
from threading import Lock
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 0=所有日志 1=过滤INFO 2=过滤
import torch
import torch.nn as nn
import re
from transformers import BertModel, BertConfig, BertTokenizerFast
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
__all__=['get_answer','inquire']

MAX_LEN   = 256
D_MODEL   = 256
N_HEAD    = 8
N_LAYER   = 4
D_FF      = 1024
VOCAB     = 21128          # 中文 BERT 词表大小，可换成自己统计
PAD_ID    = 0

TF_ENABLE_ONEDNN_OPTS=0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class QADataset(Dataset):
    def __init__(self, tokenizer, qa_data):
        self.tokenizer = tokenizer
        self.data = [item for item in qa_data if item[2] in item[1]]  # 筛选答案在上下文中的数据

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, context, answer = self.data[idx]
        encoded = encode_qa_pair(question, context, answer,self.tokenizer)
        return {
            'input_ids': encoded['input_ids'],
            'start_pos': encoded['start_pos'],
            'end_pos': encoded['end_pos']
        }

class EnhancedQAModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 num_additional_layers=2,
                 use_relative_attention=True,
                 use_focal_loss=False):
        super().__init__()

        # 加载基础BERT模型
        config = BertConfig.from_pretrained(model_name)
        config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob
        })

        self.bert = BertModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        # 增强的注意力层
        self.additional_attentions = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=hidden_dropout_prob
            ) for _ in range(num_additional_layers)
        ]) if num_additional_layers > 0 else None

        # 增强的预测头
        self.start_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

        self.end_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

        # 是否使用Focal Loss
        self.use_focal_loss = use_focal_loss
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

            # 确保有默认token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs.last_hidden_state

        # 额外的注意力层
        if self.additional_attentions is not None:
            for layer in self.additional_attentions:
                sequence_output = layer(sequence_output)

        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_head(sequence_output).squeeze(-1)
        end_logits = self.end_head(sequence_output).squeeze(-1)

        return start_logits, end_logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

_runtime = None
_runtime_lock = Lock()


def _load_runtime():
    """Load the QA model on first inference instead of during package import."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        model_path = Path(
            os.getenv(
                "QGAI_QA_MODEL_PATH",
                str(Path(__file__).with_name("QG3_enhanced_qa_model.pth")),
            )
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                f"QA model not found at {model_path}. Set QGAI_QA_MODEL_PATH."
            )
        runtime_tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        runtime_model = EnhancedQAModel(
            model_name="bert-base-chinese",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            num_additional_layers=2,
            use_focal_loss=True,
        ).to(device)
        runtime_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        runtime_model.eval()
        _runtime = runtime_model, runtime_tokenizer
        return _runtime


def encode_qa_pair(question, context, answer_text, tokenizer=None):
    if tokenizer is None:
        _, tokenizer = _load_runtime()
    enc = tokenizer.encode_plus(
        question, context,
        max_length=MAX_LEN,
        truncation='only_second',
        return_offsets_mapping=True,
        padding='max_length',
    )
    offset = enc['offset_mapping']

    # 找到答案的字符级起止
    ans_char_start = context.find(answer_text)
    if ans_char_start == -1:
        raise ValueError(f"答案不在上下文中: {answer_text} vs {context}")

    ans_char_end = ans_char_start + len(answer_text)

    # 精确映射到 token
    token_start = token_end = 0
    for idx, (s, e) in enumerate(offset):
        if s <= ans_char_start < e:
            token_start = idx
        if s < ans_char_end <= e:
            token_end = idx
            break

    # 检查 token 是否正确
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
    predicted = ''.join(tokens[token_start:token_end + 1]).replace('##', '')
    if predicted != answer_text:
        print(f" token 映射错误: {predicted} != {answer_text}")

    return {
        'input_ids': torch.tensor(enc['input_ids']),
        'start_pos': torch.tensor(token_start),
        'end_pos': torch.tensor(token_end),
    }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    start_pos = torch.stack([item['start_pos'] for item in batch])
    end_pos = torch.stack([item['end_pos'] for item in batch])
    return input_ids, start_pos, end_pos

def train_enhanced_model(training_data):
    """Train with caller-supplied data; private samples are never bundled."""
    if not training_data:
        raise ValueError("training_data must not be empty")
    # 超参数配置
    config = {
        "model_name": "bert-base-chinese",
        "hidden_dropout_prob": 0.3,
        "attention_probs_dropout_prob": 0.2,
        "num_additional_layers": 2,
        "use_focal_loss": True,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 1.0,
        "num_epochs": 10,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 2
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = EnhancedQAModel(
        model_name=config["model_name"],
        hidden_dropout_prob=config["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        num_additional_layers=config["num_additional_layers"],
        use_focal_loss=config["use_focal_loss"]
    ).to(device)

    # 准备数据
    tokenizer = BertTokenizerFast.from_pretrained(config["model_name"])
    dataset = QADataset(tokenizer, training_data)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 优化器和学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    total_steps = len(dataloader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 损失函数
    if config["use_focal_loss"]:
        loss_fn = FocalLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()


    # 训练循环
    model.train()
    global_step = 0
    for epoch in range(config["num_epochs"]):
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            input_ids, start_pos, end_pos = [x.to(device) for x in batch]
            start_logits, end_logits = model(input_ids)
            print(f"Logits范围: start={start_logits.min().item():.2f}~{start_logits.max().item():.2f}")

            # 获取模型输出
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=(input_ids != tokenizer.pad_token_id).long()
            )

            # 计算损失
            start_loss = loss_fn(start_logits, start_pos)
            end_loss = loss_fn(end_logits, end_pos)
            loss = (start_loss + end_loss) / 2

            # 梯度累积
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["max_grad_norm"]
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item()

            # 日志记录
            if global_step % 50 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), "QG3_enhanced_qa_model.pth")
    return model
def get_answer(answer, key_q, model=None, tokenizer=None):
    id_match = ""
    id_match = re.search(r'\d{17}[\dXx]', answer)
    if id_match:
        return id_match.group(0)
    if "民族" in key_q:
        match = re.search(
            r'汉族|回族|藏族|维吾尔族|壮族|苗族|彝族|布依族|朝鲜族|满族|侗族|瑶族|白族|土家族|哈尼族|哈萨克族|傣族|黎族|傈僳族|佤族|畲族|高山族|拉祜族|水族|东乡族|纳西族|景颇族|柯尔克孜族|土族|达斡尔族|仫佬族|羌族|布朗族|撒拉族|毛南族|仡佬族|锡伯族|阿昌族|普米族|塔吉克族|怒族|乌孜别克族|俄罗斯族|鄂温克族|德昂族|保安族|裕固族|京族|塔塔尔族|独龙族|鄂伦春族|赫哲族|门巴族|珞巴族|基诺族',
            answer
        )
        return match.group(0) if match else ""
    if "申领原因" in key_q:
        return answer

    if model is None or tokenizer is None:
        model, tokenizer = _load_runtime()
    device = next(model.parameters()).device
    enc = tokenizer.encode_plus(
    key_q,
    answer,
    max_length=MAX_LEN,
    truncation='only_second',
    return_offsets_mapping=True,
    padding='max_length',
    return_tensors='pt'
    )

    # 准备所有必需输入
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    token_type_ids = torch.zeros_like(input_ids)  # EnhancedQAModel需要

    # 生成token_type_ids（区分问题和上下文）
    sep_pos = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=False)[-2, 1].item()
    context_start = sep_pos + 1
    token_type_ids[:, context_start:] = 1

    model.eval()
    with torch.no_grad():
        start_logits, end_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 调试信息

        # 屏蔽问题和padding部分
    context_start = sep_pos + 1
    mask = (attention_mask == 0) | (torch.arange(MAX_LEN).to(device) < context_start)
    start_logits[mask] = -1e9
    end_logits[mask] = -1e9

        # 获取预测位置
    start = start_logits.argmax(dim=1).item()
    end = end_logits.argmax(dim=1).item()
    print("start_logits.argmax():", start_logits.argmax().item())
    print("end_logits.argmax():", end_logits.argmax().item())
    print("context_start:", context_start)

    answer_tokens = input_ids[0, start:end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer.replace(" ", "").replace("##","")

def inquire(data_dict):
    """
    检查字典中是否有缺失字段，并返回提示信息
    参数:
        data_dict: 要检查的字典数据
        required_fields: 必填字段列表
    返回:
        list: 缺失字段的提示信息列表，若无缺失则返回空列表
    """
    for field in data_dict.keys():
        # 检查字段是否存在或值为空（None/空字符串/空列表等）
        if data_dict[field] is None or data_dict[field] == "":
            return {field: f"请问您的「{field}」是什么？"}

    return None
