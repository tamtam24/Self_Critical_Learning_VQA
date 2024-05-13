import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, ViTModel, get_linear_schedule_with_warmup, BeitModel, DeiTModel


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class QuesEmbedding(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(QuesEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.lstm = nn.LSTM(input_size, output_size, batch_first=True)

    def forward(self, ques):
        tokenized_input = self.tokenizer(ques, return_tensors='pt', padding='max_length', max_length=27, truncation=True)
        ques = self.phobert(**tokenized_input.to(device)).last_hidden_state
        _, (h, _) = self.lstm(ques)
        return h.squeeze(0)