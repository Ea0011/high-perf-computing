# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Iacopo/Shakespear-GPT2")
model = AutoModelForCausalLM.from_pretrained("Iacopo/Shakespear-GPT2")


# Generate
from transformers import pipeline

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print(generator("<|endoftext|>", max_length=100, do_sample=False, temperature=0.7))