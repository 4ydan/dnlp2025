import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as functional

class Encoder(nn.Module):
    def __init__(self, hidden_dim, embedding_matrix, dropout_ratio):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        vocab_size, embedding_dim = embedding_matrix.shape
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=True, padding_idx=0)
        self.dropout = nn.Dropout(dropout_ratio)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.ques_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Sentinel vectors
        self.sentinel_c = nn.Parameter(torch.randn(self.hidden_dim))
        self.sentinel_q = nn.Parameter(torch.randn(self.hidden_dim))

    def forward(self, context_ids, question_ids, context_lengths, question_lengths):
        batch_size = context_ids.size(0)
        
        # Sort context
        context_lengths, perm_index_C = context_lengths.sort(descending=True)
        context_ids = context_ids[perm_index_C]

        # Sort question
        question_lengths, perm_index_Q = question_lengths.sort(descending=True)
        question_ids = question_ids[perm_index_Q]

        # Embeddings
        context_emb = self.dropout(self.embedding(context_ids))
        question_emb = self.dropout(self.embedding(question_ids))

        # Pack sequences
        packed_context = pack_padded_sequence(context_emb, lengths=context_lengths.view(-1).cpu(), batch_first=True, enforce_sorted=True)
        packed_question = pack_padded_sequence(question_emb, lengths=question_lengths.view(-1).cpu(), batch_first=True, enforce_sorted=True)

        # Encode
        packed_context_output, _ = self.encoder(packed_context)
        D, _ = pad_packed_sequence(packed_context_output, batch_first=True)
        D = D.contiguous()

        packed_question_output, _ = self.encoder(packed_question)
        Q_intermediate, _ = pad_packed_sequence(packed_question_output, batch_first=True)
        Q_intermediate = Q_intermediate.contiguous()

        # Project question
        Q = torch.tanh(self.ques_projection(Q_intermediate))

        # Append sentinel vectors
        sentinel_c = self.sentinel_c.unsqueeze(0).expand(batch_size, self.hidden_dim).unsqueeze(1)
        sentinel_q = self.sentinel_q.unsqueeze(0).expand(batch_size, self.hidden_dim).unsqueeze(1)

        D = torch.cat((D, sentinel_c), dim=1)  # B x (m+1) x l
        Q = torch.cat((Q, sentinel_q), dim=1)  # B x (n+1) x l

        return D, Q
    
class DynamicDecoder(nn.Module):

    def __init__(self, input_size, hidden_dim, maxout_pool_size, max_steps, dropout_ratio):
        super().__init__()
        self.max_steps = max_steps
        self.lstm = nn.LSTM(input_size, hidden_dim, 1, batch_first=True)

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

        if target is not None:
            max_valid_index = score.size(1) - 1
            if (target > max_valid_index).any():
                # print(f"Warning: Clamping {torch.sum(target > max_valid_index)} target indices out of bounds.")
                target = torch.clamp(target, max=max_valid_index)
            loss = self.loss(score, target)
            loss = loss * change_mask.float()
        
        return idx, change_mask, loss

class CoattentionModel(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, emb_matrix, max_dec_steps, dropout_ratio):
        super(CoattentionModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(hidden_dim, emb_matrix, dropout_ratio)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_fusion = nn.LSTM(3 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dynamic_decoder = DynamicDecoder(4 * hidden_dim, hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, d_seq, d_lens, q_seq, q_lens, span=None):
        D, Q = self.encoder(d_seq, q_seq, d_lens, q_lens)

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

        # fusion of temporal information to the coattention context via a bidirectional LSTM
        bilstm_in = torch.cat((C_D_t, D), 2) # B x m + 1 x 3l
        # Exclude the sentinel vector from further computation
        bilstm_in = bilstm_in[:, :-1, :]
        packed_bilstm_in = pack_padded_sequence(bilstm_in, lengths=d_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_U, (_) = self.temporal_fusion(packed_bilstm_in)
        U, (_) = pad_packed_sequence(packed_U, batch_first=True)

        # === SANITY CHECKS ===

        # 1. Check U shape: [batch_size, seq_len, hidden_dim]
        assert U.dim() == 3, f"U must be 3D but got shape {U.shape}"

        b, seq_len, hidden_dim = U.shape
        assert b == d_lens.size(0), f"Batch size mismatch: U batch {b}, d_lens batch {d_lens.size(0)}"

        # 2. Check d_lens is 1D, and dtype is long or int
        assert d_lens.dim() == 1, f"d_lens must be 1D but got {d_lens.shape}"
        assert d_lens.dtype in [torch.int32, torch.int64], f"d_lens must be integer type but got {d_lens.dtype}"

        # 3. Check device consistency
        assert U.device == d_lens.device, f"U device {U.device} != d_lens device {d_lens.device}"

        # 4. Check padding mask shape will match U sequence length
        #    d_lens max must not exceed seq_len
        max_len = d_lens.max().item()
        assert max_len <= seq_len, f"d_lens max {max_len} cannot be larger than U seq_len {seq_len}"

        context_pad_mask = (torch.arange(seq_len).unsqueeze(0).to(U.device) < d_lens.unsqueeze(1)).float()

        # Check context_pad_mask shape matches U batch and seq length
        assert context_pad_mask.shape == (b, seq_len), f"context_pad_mask shape {context_pad_mask.shape} invalid"

        loss, start_pred, end_pred = self.dynamic_decoder(U, context_pad_mask, span)
        if self.training:
            return loss
        else:
            return start_pred, end_pred
