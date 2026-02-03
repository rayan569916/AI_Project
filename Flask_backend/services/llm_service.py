from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from langchain_community.tools import DuckDuckGoSearchRun

# rm -rf ~/.cache/huggingface 
from peft import PeftModel

base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_model_id = "xshubhamx/tiny-llama-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map=device,
    dtype=torch.float16 if device == "mps" else torch.float32
)
model = PeftModel.from_pretrained(model, adapter_model_id)

# Return the adapter model ID to the frontend
model_id = adapter_model_id
search_tool = DuckDuckGoSearchRun()

def clean_search_results(text: str, max_chars=800):
    text = text.replace("\n", " ")
    return text[:max_chars]

def condense_query(user_input: str, history: list):
    if not history:
        return user_input

    recent_history = history[-3:]
    history_query = "\n".join(recent_history)

    search_prompt = f"""<|system|>
                    You rewrite user questions into short web search queries (max 5 words).
                    Do not explain. Do not add punctuation.
                    Conversation:
                    {history_query}
                    User: {user_input}
                    <|assistant|>
                    """

    inputs = tokenizer(search_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

def is_complete(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
        
    # Check if it ends with partial list number like "1." or "2)"
    import re
    if re.search(r'\d+[.)]$', text):
        return False
        
    return text.endswith((".", "!", "?", '"'))

def generate_response(user_input: str,history,mode:str,file:str) -> str:

    smart_query=condense_query(user_input,history)
    try:
        search_results=clean_search_results(search_tool.run(smart_query))
        print(f"search result from browser :{search_results}")
    except Exception as e:
        search_results = "No current search data available."

    prompt = f"""<|system|>
            You are a helpful and friendly AI assistant.

            You have access to the following context:
            1. WEB SEARCH RESULTS
            2. UPLOADED DOCUMENTS (if any)

            Rules:
            - Answer the user's question directly and naturally.
            - Do NOT start with "Based on..." or "According to...".
            - Do NOT mention "search results", "documents", or "provided information" in your response.
            - Act as if you know the answer yourself.
            - If the contextual information is not relevant, ignore it.
            - Keep answers concise (2-3 sentences).

            <|context|>
            WEB SEARCH RESULTS:
            {search_results}

            UPLOADED DOCUMENTS:
            {file if file else "None"}

            <|user|>
            {user_input}
            <|assistant|>
            """
    max_token=200
    temperature = 0.7
    top_k = 50
    
    if mode=="fast":
        max_token=150
    elif mode=="thinking":
        max_token=300
        temperature = 0.8
    elif mode=="pro":
        max_token=600
        temperature = 0.5 

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_token,
        do_sample=True,
        temperature=temperature,
        top_k=top_k, 
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = decoded.split("<|assistant|>")[-1].strip()

    # Continuation Loop
    retry_count = 0
    max_retries = 3
    
    while not is_complete(reply) and retry_count < max_retries:
        retry_count += 1
        print(f"🔄 Extending response... (Attempt {retry_count})")
        
        continuation_prompt = prompt + reply
        
        inputs_cont = tokenizer(continuation_prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs_cont,
            max_new_tokens=100,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
        
        full_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the full reply again (since we fed prompt + partial_reply)
        reply = full_decoded.split("<|assistant|>")[-1].strip()

    end = time.time()
    print(f"⏱️ Time: {end - start:.2f}s")
    
    return {"reply":reply,
            "search_results":search_results,
            "model":model_id}