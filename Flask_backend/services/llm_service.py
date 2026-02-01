from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from langchain_community.tools import DuckDuckGoSearchRun

# rm -rf ~/.cache/huggingface 
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32
)
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

    inputs = tokenizer(search_prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()



def generate_response(user_input: str,history:[]) -> str:

    smart_query=condense_query(user_input,history)
    try:
        search_results=clean_search_results(search_tool.run(smart_query))
        print(f"search result from browser :{search_results}")
    except Exception as e:
        search_results = "No current search data available."

    prompt = f"""<|system|>
                You are a factual QA assistant.

                Rules:
                - Use ONLY the information from SEARCH RESULTS.
                - If the answer is not present, say "I don't know based on the search results."
                - Answer in 2–3 short sentences.
                - Do not add extra knowledge.

                SEARCH RESULTS:
                {search_results}
                <|user|>
                {user_input}
                <|assistant|>
            """
    device = model.device    

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,          # 🔥 key change
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    end = time.time()

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reply = decoded.split("<|assistant|>")[-1].strip()
    reply = reply.split("<|user|>")[0].strip()

    print(f"⏱️ Time: {end - start:.2f}s")
    return reply