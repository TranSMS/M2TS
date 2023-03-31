import torch
import math


def train(gcn_optimizer, trans_optimizer, data_loader, gcn_model, trans_model, criterion, device):
    epoch_loss = 0
    gcn_model.train()
    trans_model.train()

    for adj, x, a2, a3, a4, a5, enc_inputs, dec_inputs, dec_outputs in data_loader:
        # for adj, x, a2, a3 in data_loader:
        adj, x, a2, a3, a4, a5 = adj.to(device), x.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
        ast_outputs, ast_embed = gcn_model(x, adj, a2, a3, a4, a5)  
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns, dec_ast_attns = trans_model(enc_inputs, dec_inputs, ast_outputs)  
        loss = criterion(outputs, dec_outputs.view(-1))
        trans_optimizer.zero_grad()
        loss.backward()
        trans_optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def evaluate(data_loader, gcn_model, trans_model, criterion, device):

    epoch_loss = 0
    gcn_model.eval()
    trans_model.eval()

    with torch.no_grad():
        for adj, x, a2, a3, a4, a5, enc_inputs, dec_inputs, dec_outputs in data_loader:
            adj, x, a2, a3, a4, a5 = adj.to(device), x.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
            ast_outputs, ast_embed = gcn_model(x, adj, a2, a3, a4, a5)
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs1, enc_self_attns, dec_self_attns, dec_enc_attns, dec_ast_attns = trans_model(enc_inputs, dec_inputs, ast_outputs)  # 变动
            loss = criterion(outputs1, dec_outputs.view(-1))
            epoch_loss += loss.item()
    losses = epoch_loss / len(data_loader)
    perplexity = math.exp(losses)
    perplexity = torch.tensor(perplexity).item()
    return losses, perplexity
