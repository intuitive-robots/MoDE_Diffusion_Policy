import threading
from collections import OrderedDict
import pickle
import torch

class AdvancedLangEmbeddingBuffer:
    def __init__(self, language_encoder, goal_instruction_buffer_size=10000):
        self.language_encoder = language_encoder
        self.goal_instruction_buffer_size = goal_instruction_buffer_size
        self.goal_instruction_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()

    def get_or_encode_batch(self, texts):
        # print(texts)
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            with self.buffer_lock:
                uncached_texts = [text for text in texts if text not in self.goal_instruction_buffer]
            
            if uncached_texts:
                # Directly use the language encoder on the uncached texts
                encoded_batch = self.language_encoder(uncached_texts)
                
                for text, embedding in zip(uncached_texts, encoded_batch):
                    self.add_to_buffer(text, embedding)
            
            with self.buffer_lock:
                encoded_texts = [self.goal_instruction_buffer[text] for text in texts]
            
            return torch.stack(encoded_texts)

        except Exception as e:
            print(f"Error encoding texts: {e}")
            # If all else fails, return a dummy tensor
            # Assuming the output dimension of the language encoder is known
            return torch.zeros((len(texts), self.language_encoder.output_dim))

    def add_to_buffer(self, key, value):
        with self.buffer_lock:
            if len(self.goal_instruction_buffer) >= self.goal_instruction_buffer_size:
                self.goal_instruction_buffer.popitem(last=False)
            self.goal_instruction_buffer[key] = value

    def get_goal_instruction_embedding(self, goal_instruction):
        return self.get_or_encode_batch([goal_instruction])

    def get_goal_instruction_embeddings(self, goal_instructions):
        return self.get_or_encode_batch(goal_instructions)

    def clear_buffer(self):
        with self.buffer_lock:
            self.goal_instruction_buffer.clear()

    def get_buffer_size(self):
        with self.buffer_lock:
            return len(self.goal_instruction_buffer)

    def preload_common_strings(self, goal_instruction_list):
        self.get_or_encode_batch(goal_instruction_list)

    def save_buffer(self, filepath):
        with self.buffer_lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.goal_instruction_buffer, f)

    def load_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)
        with self.buffer_lock:
            self.goal_instruction_buffer = OrderedDict(list(loaded_buffer.items())[-self.goal_instruction_buffer_size:])
