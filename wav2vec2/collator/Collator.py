import torch

def data_collator(batch, processor):
    # Extract input_values from each batch sample
    input_values = [item['input_values'] for item in batch]
    # Pad the input_values to the maximum length in the batch using the provided processor
    padded_inputs = processor.pad({"input_values": input_values}, return_tensors="pt")
    
    # Stack labels into a tensor
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    padded_inputs['labels'] = labels
    return padded_inputs
