#!/usr/bin/python3
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F



class FastText(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.embedding = nn.Embedding(25000, 100)
        
        self.fc = nn.Linear(100, 1)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.reshape(embedded.shape[0], 1, embedded.shape[1], embedded.shape[2])
        
        pooled = F.avg_pool2d(embedded, (5, 1))
        
        pooled = pooled.squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)

model = FastText()
model.eval()
dummy_input = torch.randint(0, 10000, (5, 1))
#torch.onnx.export(model, dummy_input, "bert_tiny.onnx")
torch.onnx.export(model, dummy_input, "embedding.onnx", opset_version=12,)
