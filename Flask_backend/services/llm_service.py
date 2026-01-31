from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32
)

def generate_response(user_input: str) -> str:
    prompt = f"""<|system|>
                You are a helpful AI assistant.
                <|user|>
                {user_input}
                <|assistant|>
            """

    inputs = tokenizer(prompt, return_tensors="pt")

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=5000,
        do_sample=True,
        temperature=1.3,
        top_p=0.95
    )
    end = time.time()

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reply = decoded.split("<|assistant|>")[-1].strip()

    print(f"⏱️ Time: {end - start:.2f}s")
    return reply


 
