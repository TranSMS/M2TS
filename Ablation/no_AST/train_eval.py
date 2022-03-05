import math
import torch


def train(optimizer, data_loader, model, criterion, device):
    model.train()
    epoch_loss = 0
    for enc_inputs, dec_inputs, dec_outputs in data_loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_outputs in data_loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            epoch_loss += loss.item()
    losses = epoch_loss / len(data_loader)
    # perplexity = math.exp(losses)
    # perplexity = torch.tensor(perplexity)
    return losses