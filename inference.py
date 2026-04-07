import torch
import torch.nn as nn
from data_loader import get_transforms

class CaptionGenerator:
    def __init__(self, model, vocab, device, image_size=224):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.transform = get_transforms(image_size=image_size, is_train=False)

    def generate_caption_greedy(self, image, max_len=20):
        self.model.eval()
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoder_out = self.model.encoder(image)
        
        result_caption = [self.vocab.stoi["<SOS>"]]
        
        for _ in range(max_len):
            captions_input = torch.tensor(result_caption).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.model.decoder(captions_input, encoder_out)
            
            # (1, seq_len, vocab_size) -> (1, vocab_size)
            predicted_tensor = predictions[0, -1, :]
            predicted_id = predicted_tensor.argmax().item()
            
            result_caption.append(predicted_id)
            
            if predicted_id == self.vocab.stoi["<EOS>"]:
                break
        
        return [self.vocab.itos[idx] for idx in result_caption[1:-1]]

    def generate_caption_beam_search(self, image, beam_size=3, max_len=20):
        """
        Implementation of Beam Search decoding for improved caption quality.
        """
        self.model.eval()
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoder_out = self.model.encoder(image)
        
        # Start node: (caption_ids, log_prob)
        start_id = self.vocab.stoi["<SOS>"]
        beams = [([start_id], 0.0)]
        completed_beams = []

        for _ in range(max_len):
            all_candidates = []
            
            for caption_ids, log_prob in beams:
                if caption_ids[-1] == self.vocab.stoi["<EOS>"]:
                    all_candidates.append((caption_ids, log_prob))
                    continue
                
                captions_input = torch.tensor(caption_ids).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    predictions = self.model.decoder(captions_input, encoder_out)
                
                # Get log probabilities for the last projected word
                log_probabilities = torch.log_softmax(predictions[0, -1, :], dim=0)
                
                # Get top K candidates
                topk_probs, topk_indices = torch.topk(log_probabilities, beam_size)
                
                for k in range(beam_size):
                    token_id = topk_indices[k].item()
                    
                    # Repetition Penalty: Heavily penalize words that were already used in the same caption
                    # (except for basic grammar glue like 'a', 'the', 'in', etc.)
                    penalty = 0
                    if token_id in caption_ids:
                        token_word = self.vocab.itos.get(token_id, "")
                        if token_word not in ["a", "the", "in", "and", "is", "of", "on", "sitting", "standing"]:
                            penalty = 5.0 # Large penalty
                        else:
                            penalty = 1.0 # Smaller penalty for glue words
                            
                    candidate = (caption_ids + [token_id], log_prob + topk_probs[k].item() - penalty)
                    all_candidates.append(candidate)
            
            # Sort all candidates by score and pick top beam_size
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_size]
            
            # If all top beams ended with <EOS>, we can stop
            if all(b[0][-1] == self.vocab.stoi["<EOS>"] for b in beams):
                break
        
        # Pick the best beam and cleanup
        best_beam = beams[0][0]
        final_caption = []
        for idx in best_beam:
            if idx in [self.vocab.stoi["<SOS>"], self.vocab.stoi["<PAD>"]]:
                continue
            if idx == self.vocab.stoi["<EOS>"]:
                break
            
            # Safe word lookup (filter out UNKs for better readability)
            word = self.vocab.itos.get(idx, "")
            if word and word != "<UNK>":
                final_caption.append(word)
                
        return final_caption
