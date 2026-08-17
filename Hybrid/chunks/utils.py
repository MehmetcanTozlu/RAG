import tiktoken


def get_token_length(text: str) -> int:
    """
    Calculates the token length of a given text.
    """
    try:
        if not text:
            return 0
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
    
    except Exception as e:
        print(f"\033[91m[WARN] Could not calculate token length: {e}\033[0m")
        return len(text.split()) # Divide by average word length as fallback
