import torch
import math
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def train(gcn_optimizer, trans_optimizer,  data_loader, gcn_model, trans_model, criterion, device):

    epoch_loss = 0
    gcn_model.train()
    trans_model.train()
    # trans2_model.train()

    for adj, x, a2, a3, a4, a5, enc_inputs, dec_inputs, dec_outputs in data_loader:
        # for adj, x, a2, a3 in data_loader:
        adj, x, a2, a3, a4, a5 = adj.to(device), x.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
        ast_outputs, ast_embed = gcn_model(x, adj, a2, a3, a4, a5)  # 变动

        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)

        outputs, enc_self_attns, dec_self_attns, dec_enc_attns, dec_ast_attns = trans_model(enc_inputs, dec_inputs, ast_outputs, ast_embed)  # 变动
        loss = criterion(outputs, dec_outputs.view(-1))

        trans_optimizer.zero_grad()
        gcn_optimizer.zero_grad()
        loss.backward()
        trans_optimizer.step()
        gcn_optimizer.step()

        epoch_loss += loss.item()
    # epoch_loss = epoch_loss / niters
    return epoch_loss / len(data_loader)


def evaluate(data_loader, gcn_model, trans_model, criterion, device):

    epoch_loss = 0
    gcn_model.eval()
    trans_model.eval()
    # trans2_model.eval()

    with torch.no_grad():
        for adj, x, a2, a3, a4, a5, enc_inputs, dec_inputs, dec_outputs in data_loader:
            # for adj, x, a2, a3 in data_loader:
            adj, x, a2, a3, a4, a5 = adj.to(device), x.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
            ast_outputs, ast_embed = gcn_model(x, adj, a2, a3, a4, a5)  # 变动
            # ast_outputs = ast_outputs.unsqueeze(0).to(device)
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            # output1, self_attns, de_self_attns, de_enc_attns = trans2_model(enc_inputs, dec_inputs)
            outputs1, enc_self_attns, dec_self_attns, dec_enc_attns, dec_ast_attns = trans_model(enc_inputs, dec_inputs, ast_outputs, ast_embed)  # 变动
            loss1 = criterion(outputs1, dec_outputs.view(-1))
            # loss2 = criterion(outputs2, dec_outputs.view(-1))
            # loss = 0.8*loss1 + 0.2*loss2
            loss = loss1

            epoch_loss += loss.item()

    losses = epoch_loss / len(data_loader)
    perplexity = math.exp(losses)
    perplexity = torch.tensor(perplexity).item()
    return losses, perplexity