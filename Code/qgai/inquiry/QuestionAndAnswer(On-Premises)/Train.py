import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from Model import QAModel, train_step, predict, encode_qa_pair, MAX_LEN
from qa_data import qa_data
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, tokenizer, qa_data):
        self.tokenizer = tokenizer
        self.data = qa_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, context, answer = self.data[idx]
        encoded = encode_qa_pair(self.tokenizer, question, context, answer)
        return {
            'input_ids': encoded['input_ids'],
            'start_pos': encoded['start_pos'],
            'end_pos': encoded['end_pos']
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    start_pos = torch.stack([item['start_pos'] for item in batch])
    end_pos = torch.stack([item['end_pos'] for item in batch])
    return input_ids, start_pos, end_pos

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
dataset = QADataset(tokenizer, qa_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

model = QAModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
dataset = QADataset(tokenizer, qa_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(20):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids, start_pos, end_pos = [x.to(device) for x in batch]
        loss = torch.nn.CrossEntropyLoss()(model(input_ids)[0], start_pos) + \
               torch.nn.CrossEntropyLoss()(model(input_ids)[1], end_pos)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  # 实时显示当前 batch loss
    print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

model.eval()
test_questions = [
        ("谁发明了电灯？", "电灯是约瑟夫·斯旺发明的"),
        ("今天天气怎么样？", "今天天气晴转多云"),
        ("你在哪里上学？", "我在清华大学学习")
    ]

for q, c in test_questions:
    pred = predict(model, tokenizer, q, c)
    print(f"Q: {q}\nC: {c}\nA: {pred}\n")