import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as functional

class Encoder(nn.Module):
    def __init__(self, hidden_dim, embedding_matrix, dropout_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        _, embedding_dim = embedding_matrix.shape
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout_ratio)
        
        self.ques_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Sentinel vectors
        self.sentinel_c = nn.Parameter(torch.randn(self.hidden_dim))
        self.sentinel_q = nn.Parameter(torch.randn(self.hidden_dim))
    
    def encode_sequence(self, idxs, mask, sentinel):
        lengths = mask.sum(dim=1)  # [batch]
        sorted_lens, sorted_idx = lengths.sort(descending=True)
        _, orig_idx = sorted_idx.sort()
        
        # Sort sequences for packing
        idxs_sorted = idxs[sorted_idx]
        emb = self.embedding(idxs_sorted)
        packed = pack_padded_sequence(emb, sorted_lens.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM encoding
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [batch, max_len, hidden]
        out = self.dropout(out)
        out = out[orig_idx]  # restore original order
        
        # Append sentinel
        batch_size = out.size(0)
        sentinel_expanded = sentinel.unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
        
        out_with_sentinel = torch.cat([out, torch.zeros_like(sentinel_expanded)], dim=1)  # [batch, max_len+1, hidden]
        lens = lengths.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)  # [batch, 1, hidden]
        out_with_sentinel = out_with_sentinel.scatter(1, lens, sentinel_expanded)
        
        return out_with_sentinel # [batch, seq_len + 1, hidden]
    
    def forward(self, doc_idxs, doc_mask, q_idxs, q_mask):
        D = self.encode_sequence(doc_idxs, doc_mask, self.sentinel_c)  # [batch, m+1, hidden]
        Q_prime = self.encode_sequence(q_idxs, q_mask, self.sentinel_q)  # [batch, n+1, hidden]
        
        # Nonlinear projection: Q = tanh(W * Q′ + b)
        Q = torch.tanh(self.ques_projection(Q_prime))  # [batch, n+1, hidden]
        
        return D, Q
    
class FusionBiLSTM(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(FusionBiLSTM, self).__init__()
        self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True,
                                     bidirectional=True, dropout=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        packed = pack_padded_sequence(seq_, lens_sorted, batch_first=True)
        output, _ = self.fusion_bilstm(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout(e)
        return e

class DynamicDecoder(nn.Module):

    def __init__(self, input_size, hidden_dim, maxout_pool_size, max_steps, dropout_ratio=0.0):
        super().__init__()
        self.max_steps = max_steps
        self.lstm = nn.LSTM(input_size, hidden_dim, 1, batch_first=True)

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.maxout_start = MaxOutHighWay(hidden_dim, maxout_pool_size, dropout_ratio)
        self.maxout_end = MaxOutHighWay(hidden_dim, maxout_pool_size, dropout_ratio)

    def forward(self, U, doc_pad_mask, target):
        b,m,_ = list(U.size())

        curr_change_mask_s, curr_change_mask_e = None, None

        masks_s, masks_e, results_s, results_e, losses = [], [], [], [], []

        # invert the document pad mask -> multiply padded values with smalles possible value -> no influence on loss computation
        pad_mask = (1.0-doc_pad_mask.float()) * torch.finfo(torch.float32).min

        idxs = torch.arange(0,b,out=torch.LongTensor(b))

        #init start and end index to 0 and last word in document
        s_idx_prev = torch.zeros(b,).long()
        # sum evaluates to all words in document, since pad tokens == 0 and rest == 1 
        e_idx_prev = torch.sum(doc_pad_mask,1) - 1

        decoder_state = None
        s_target = None
        e_target = None
        
        #extract idx from given answer span
        if target is not None:
            s_target = target[:,0]
            e_target = target[:,1]

        #get previously computed start index coattention representation
        u_s_idx_prev = U[idxs, s_idx_prev,:]

        #decoder iterations (recommmended: 16)

        for i in range(self.max_steps):
            #get previously computed end index coattention represenation
            u_e_idx_prev = U[idxs, e_idx_prev.long(), :]
            u_s_e = torch.cat((u_s_idx_prev, u_e_idx_prev), 1)

            lstm_out, decoder_state = self.lstm(u_s_e.unsqueeze(1), decoder_state)
            #extract final hidden state h_i
            c_i, h_i = decoder_state

            #compute new start index
            s_idx_prev, curr_change_mask_s, loss_s = self.maxout_start(h_i, U, u_s_e, pad_mask, s_idx_prev, curr_change_mask_s, s_target) 

            #update start index with index computed above
            u_s_idx_prev = U[idxs, s_idx_prev, :]
            u_s_e = torch.cat((u_s_idx_prev, u_e_idx_prev), 1)

            #compute new end index
            e_idx_prev, curr_change_mask_e, loss_e = self.maxout_end(h_i, U, u_s_e, pad_mask, e_idx_prev, curr_change_mask_e, e_target) 

            if target is not None:
                loss = loss_s + loss_e
                losses.append(loss)

            masks_s.append(curr_change_mask_s)
            masks_e.append(curr_change_mask_e)
            results_s.append(s_idx_prev)
            results_e.append(e_idx_prev)

        #retrieve last index predictions where updates halted
        #idx should have shape (b,)
        result_idx_s = torch.sum(torch.stack(masks_s,1),1).long() - 1
        idx_s = torch.gather(torch.stack(results_s,1),1,result_idx_s.unsqueeze(1)).squeeze()
        result_idx_e = torch.sum(torch.stack(masks_e,1),1).long() - 1
        idx_e = torch.gather(torch.stack(results_e,1),1,result_idx_e.unsqueeze(1)).squeeze()

        loss = None

        #compute loss while training and evaluating
        if target is not None:
            sum_losses = torch.sum(torch.stack(losses,1),1)
            avg_loss = sum_losses/self.max_steps
            loss = torch.mean(avg_loss)
        # print(f"DEBUG: Before return - type(loss): {type(loss)}, value: {loss}")
        # print(f"DEBUG: Before return - type(idx_s): {type(idx_s)}, value: {idx_s}")
        # print(f"DEBUG: Before return - type(idx_e): {type(idx_e)}, value: {idx_e}")
        return loss, idx_s, idx_e
    
class MaxOutHighWay(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, dropout_ratio=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.maxout_pool_size = maxout_pool_size
        self.w_d = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)
        self.w_1 = nn.Linear(3 * hidden_dim, hidden_dim*maxout_pool_size)
        self.w_2 = nn.Linear(hidden_dim, hidden_dim*maxout_pool_size)
        self.w_3 = nn.Linear(2 * hidden_dim, hidden_dim*maxout_pool_size)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, h_i, U, u_s_e, pad_mask, idx_prev, change_mask, target=None):
        b,m,_ = list(U.size())
        #use view if dimensions dont match for cat
        r_in = self.w_d(torch.cat((h_i.view(-1,self.hidden_dim), u_s_e),1))
        r = functional.tanh(r_in)
        # print("r.shape after tanh: ",r.shape)
        r = r.unsqueeze(1).expand(b,m,self.hidden_dim).contiguous()

        m_t_1_in = torch.cat((U,r),2).view(-1, self.hidden_dim*3)
        m_t_1, _ = self.w_1(m_t_1_in).view(-1, self.hidden_dim, self.maxout_pool_size).max(2)
        # print("m_t_1 shape: ", m_t_1.shape)

        m_t_2, _ = self.w_2(m_t_1).view(-1, self.hidden_dim, self.maxout_pool_size).max(2)

        score, _ = self.w_3(torch.cat((m_t_1,m_t_2),1)).max(1)
        score = functional.softmax((score.view(-1,m) + pad_mask), dim=1)
        _, idx = torch.max(score, dim=1)

        if change_mask is None:
            change_mask = (idx == idx)
        else:
            idx = idx * change_mask.long()
            idx_prev = idx_prev * change_mask.long()
            change_mask = (idx!=idx_prev)

        loss = None
        if target is not None:
            loss = self.loss(score, target)
            loss = loss * change_mask.float()
        
        return idx, change_mask, loss

class CoattentionModel(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, emb_matrix, max_dec_steps, dropout_ratio = 0.0):
        super(CoattentionModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(hidden_dim, emb_matrix, dropout_ratio)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_bilstm = FusionBiLSTM(hidden_dim, dropout_ratio)
        self.dynamic_decoder = DynamicDecoder(4 * hidden_dim, hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _lengths_to_mask(self, lengths, max_len):
        """Convert lengths to boolean mask."""
        return torch.arange(max_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)

    def forward(self, d_seq, d_mask, q_seq, q_mask, d_lens, span=None):
        # Call encoder with masks
        D, Q = self.encoder(d_seq, d_mask, q_seq, q_mask)
        
        # project q
        Q = torch.tanh(self.q_proj(Q.view(-1, self.hidden_dim))).view(Q.size()) #B x n + 1 x l

        # Co attention
        D_t = D.transpose(1, 2) #B x l x m + 1
        L = torch.bmm(Q, D_t) # L = B x n + 1 x m + 1

        # Attention weights for question
        A_Q_ = torch.softmax(L, dim=1) # B x n + 1 x m + 1
        A_Q = A_Q_.transpose(1, 2) # B x m + 1 x n + 1
        C_Q = torch.bmm(D_t, A_Q) # (B x l x m + 1) x (B x m x n + 1) => B x l x n + 1

        # Attention weights for document
        Q_t = Q.transpose(1, 2)  # B x l x n + 1
        A_D = torch.softmax(L, dim=2)  # B x n + 1 x m + 1
        
        C_D = torch.bmm(torch.cat((Q_t, C_Q), 1), A_D) # B x 2l x m+1
        C_D_t = C_D.transpose(1, 2)  # B x m + 1 x 2l
        #fusion BiLSTM
        bilstm_in = torch.cat((C_D_t, D), 2) # B x m + 1 x 3l
        bilstm_in = self.dropout(bilstm_in)
        U = self.fusion_bilstm(bilstm_in, d_mask) #B x m x 2l

        _, seq_len, _ = U.shape
        context_pad_mask = self._lengths_to_mask(d_lens, seq_len).float()
        loss, start_pred, end_pred = self.dynamic_decoder(U, context_pad_mask, span)

        return loss, start_pred, end_pred