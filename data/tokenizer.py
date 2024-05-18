from transformers import AutoTokenizer

class PhoBERTTokenizer:
    def __init__(self, model_name='vinai/phobert-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
    
    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)