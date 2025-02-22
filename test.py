from core.bark import generate_semantic_tokens_from_text
import torch

torch.manual_seed(42)
if __name__ == "__main__":
    text = " this is a test text"
    semantic_tokens = generate_semantic_tokens_from_text(text, use_kv_caching=True)

    print(semantic_tokens)
