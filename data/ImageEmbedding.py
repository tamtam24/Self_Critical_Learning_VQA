import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, ViTModel, get_linear_schedule_with_warmup, BeitModel, DeiTModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ImageEmbedding(nn.Module):
    def __init__(self, output_size=768, mode='train'):
        super(ImageEmbedding, self).__init__()
        self.process = AutoImageProcessor.from_pretrained("path to image model")
        self.model = ViTModel.from_pretrained("path to image model")
        
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, image_ids):
        inputs = self.process(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))
            
        image_embedding = outputs.last_hidden_state
        return image_embedding, image_ids
