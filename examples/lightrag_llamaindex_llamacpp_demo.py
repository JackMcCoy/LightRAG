import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.llama_index_impl import (
    llama_index_complete_if_cache,
    llama_index_embed,
)
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoModel, AutoTokenizer
import asyncio
import nest_asyncio

nest_asyncio.apply()

from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

setup_logger("lightrag", level="DEBUG")

# Configure working directory
WORKING_DIR = "./caro_pages-16x2560"
print(f"WORKING_DIR: {WORKING_DIR}")
BATCH_SIZE = 16

# Model configuration
LLM_MODEL = '/Users/davidgill/llama.cpp/models/llama3.1-8B-Instruct-Q4_K_M.gguf'

model_url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

def messages_to_prompt(messages):
    messages = [{"role": m.role.value, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def completion_to_prompt(completion):
    messages = [{"role": "user", "content": completion}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt

# Initialize LLM function
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:
        # Initialize LiteLLM if not in kwargs
        if "llm_instance" not in kwargs:
            llm_instance = LlamaCPP(
                model_url = model_url,
                #model_path=LLM_MODEL,
                temperature=0.7,
                max_new_tokens=2560,
                context_window=16384,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": -1},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=False,
            )
            kwargs["llm_instance"] = llm_instance

        response = await llama_index_complete_if_cache(
            kwargs["llm_instance"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        return response
    except Exception as e:
        print(f"LLM request failed: {str(e)}")
        raise


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                embed_model=AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
            ),
        ),
        addon_params={
        "insert_batch_size": 8  # Process 4 documents per batch
        }
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


rag = asyncio.run(initialize_rag())

files = os.listdir('pb_pages')
all_text = []
for f in files:
    if f.endswith('.md'):
        a = open(f'pb_pages/{f}')
        all_text.append(a.read())

rag.insert(all_text)

query = "In what ways did Governor Al Smith support Robert Moses' in the creation of the parks department?"
# Test different query modes
print("\nNaive Search:")
print(
    rag.query(
        query, param=QueryParam(mode="naive")
    )
)

print("\nLocal Search:")
print(
    rag.query(
        query, param=QueryParam(mode="local")
    )
)

print("\nGlobal Search:")
print(
    rag.query(
        query, param=QueryParam(mode="global")
    )
)

print("\nHybrid Search:")
print(
    rag.query(
        query, param=QueryParam(mode="hybrid")
    )
)
