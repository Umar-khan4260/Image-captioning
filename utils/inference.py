import torch

def greedy_search(model, image_features, vocab, max_length=50, device='cpu'):
    model.eval()
    
    with torch.no_grad():
        image_features = image_features.to(device)
        hidden_state = model.encoder(image_features)
        
        current_word = torch.tensor([vocab.word2idx[vocab.start_token]]).unsqueeze(0).to(device)
        caption_indices = []
        
        h = hidden_state.unsqueeze(0)
        c = torch.zeros_like(h)
        
        for _ in range(max_length):
            embeddings = model.decoder.embedding(current_word)
            lstm_out, (h, c) = model.decoder.lstm(embeddings, (h, c))
            output = model.decoder.fc(lstm_out.squeeze(1))
            predicted = output.argmax(1)
            
            if predicted.item() == vocab.word2idx[vocab.end_token]:
                break
            
            caption_indices.append(predicted.item())
            current_word = predicted.unsqueeze(0)
        
        caption = vocab.decode(caption_indices)
        return caption

def beam_search(model, image_features, vocab, beam_width=3, max_length=50, device='cpu'):
    model.eval()
    
    with torch.no_grad():
        image_features = image_features.to(device)
        hidden_state = model.encoder(image_features)
        
        start_token = vocab.word2idx[vocab.start_token]
        end_token = vocab.word2idx[vocab.end_token]
        
        h = hidden_state.unsqueeze(0)
        c = torch.zeros_like(h)
        
        beams = [([start_token], 0.0, h, c)]
        completed_beams = []
        
        for step in range(max_length):
            all_candidates = []
            
            for sequence, score, h_state, c_state in beams:
                if sequence[-1] == end_token:
                    completed_beams.append((sequence, score))
                    continue
                
                last_word = torch.tensor([[sequence[-1]]]).to(device)
                embeddings = model.decoder.embedding(last_word)
                lstm_out, (h_new, c_new) = model.decoder.lstm(embeddings, (h_state, c_state))
                output = model.decoder.fc(lstm_out.squeeze(1))
                log_probs = torch.log_softmax(output, dim=1)
                top_probs, top_indices = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    word_idx = top_indices[0][i].item()
                    word_prob = top_probs[0][i].item()
                    new_sequence = sequence + [word_idx]
                    new_score = score + word_prob
                    all_candidates.append((new_sequence, new_score, h_new, c_new))
            
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if all(seq[-1] == end_token for seq, _, _, _ in beams):
                completed_beams.extend([(seq, score) for seq, score, _, _ in beams])
                break
        
        completed_beams.extend([(seq, score) for seq, score, _, _ in beams])
        
        if completed_beams:
            best_sequence = max(completed_beams, key=lambda x: x[1])[0]
        else:
            best_sequence = beams[0][0]
        
        caption_indices = [idx for idx in best_sequence if idx not in [start_token, end_token]]
        caption = vocab.decode(caption_indices)
        return caption