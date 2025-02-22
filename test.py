from core.bark import generate_audio
import torch

torch.manual_seed(42)
if __name__ == "__main__":
    text = " this is a test text"
    prompt_path = "/Users/hao/Desktop/ML/bark/bark/assets/prompts/v2/en_speaker_6.npz"
    semantic_tokens = generate_audio(text, prompt_path)

    print(semantic_tokens)
