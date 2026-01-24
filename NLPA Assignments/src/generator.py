
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GEN_NAME = "google/flan-t5-base"

class Generator:
    def __init__(self, max_new_tokens=200):
        self.tokenizer = AutoTokenizer.from_pretrained(GEN_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GEN_NAME)
        self.max_new_tokens = max_new_tokens

    def format_prompt(self, q, contexts):
        ctx_blocks = []
        for i, c in enumerate(contexts, 1):
            ctx_blocks.append(f"[{i}] {c['text']}
(Source: {c['url']})")
        ctx = "

".join(ctx_blocks)
        return (
            "You are a helpful, concise assistant.
"
            "Answer the question using only the information in the sources.
"
            "Cite sources by their index in square brackets, e.g., [1][3].

"
            f"Question: {q}

"
            f"Sources:
{ctx}

"
            "Answer:"
        )

    def generate(self, q, contexts):
        prompt = self.format_prompt(q, contexts)
        inps = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        out = self.model.generate(**inps, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
