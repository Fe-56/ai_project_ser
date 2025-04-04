import torch
import torch.nn as nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load class weights
        self.class_weights = torch.load(class_weights_path).to(self.device)
        
        # Custom loss function: Cross entropy loss weighted
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").to(self.device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        
        return (loss, outputs) if return_outputs else loss