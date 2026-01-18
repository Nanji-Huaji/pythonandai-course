import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
from datasets import load_dataset
from data_utils import basic_english_tokenizer, build_vocab_from_iterator_custom, save_vocab, yield_tokens
from model import SentimentLSTM

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 数据处理 pipeline
tokenizer = basic_english_tokenizer

print("Loading IMDB dataset via HuggingFace datasets...")
dataset = load_dataset("imdb")
train_data_raw = dataset['train']
test_data_raw = dataset['test']

print("Building vocabulary...")
# Helper generator for vocab building
def get_text_iterator(data):
    for item in data:
        yield item['text']

vocab = build_vocab_from_iterator_custom(yield_tokens(get_text_iterator(train_data_raw), tokenizer), specials=["<unk>", "<pad>"])
print(f"Vocabulary size: {len(vocab)}")
save_vocab(vocab, 'vocab.pt')
print("Vocabulary saved to vocab.pt")

# Pipeline helpers
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: float(x)

# 3. Dataset & DataLoader
class IMDBDataset(Dataset):
    def __init__(self, hf_data):
        self.data = hf_data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['label'], item['text']

def collate_batch_lstm(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
        
    label_list = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)
    
    if len(text_list) > 0:
        text_padded = pad_sequence(text_list, batch_first=True, padding_value=vocab.pad_index)
    else:
        text_padded = torch.tensor([])
        
    lengths = torch.tensor(lengths, dtype=torch.int64)
    
    return label_list.to(device), text_padded.to(device), lengths.cpu()

# Prepare Datasets
train_full = IMDBDataset(train_data_raw)
test_set = IMDBDataset(test_data_raw)

train_size = int(len(train_full) * 0.95)
valid_size = len(train_full) - train_size
train_set, valid_set = torch.utils.data.random_split(train_full, [train_size, valid_size])

BATCH_SIZE = 64
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch_lstm)
valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch_lstm)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch_lstm)

# 4. 初始化模型
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.5
PAD_IDX = vocab.pad_index

model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX)
model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# 5. Helper Functions
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for labels, text, lengths in iterator:
        optimizer.zero_grad()
        predictions = model(text, lengths)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for labels, text, lengths in iterator:
            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 6. 开始训练
if __name__ == "__main__":
    N_EPOCHS = 5
    print(f"Starting training for {N_EPOCHS} epochs...")

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save strictly the state_dict
            torch.save(model.state_dict(), 'lstm-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # Test
    model.load_state_dict(torch.load('lstm-model.pt'))
    test_loss, test_acc = evaluate(model, test_dataloader, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
