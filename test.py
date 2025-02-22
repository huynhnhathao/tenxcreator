from core.bark import generate_semantic_tokens_from_text

if __name__ == "__main__":
    text = " this is a test text"
    semantic_tokens = generate_semantic_tokens_from_text(text)

    print(semantic_tokens)
