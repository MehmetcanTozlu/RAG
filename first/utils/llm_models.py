import torch
import time
from langchain_core.prompts import PromptTemplate


class LLMEngine(object):
    """
    Build your own LLM model inference engine with both transformers and llama.cpp backends.
    """
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ):
        print(f"\033[94m---Initializing Local LLM Engine---\033[0m")
        start_time = time.time()
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        if self.model_path.lower().endswith(".gguf"):
            print("GGUF format detected. Routing to Llama.cpp engine..")
            self._load_gguf()
        else:
            print("Standart HuggingFace model detected. Routing to Transformers engine..")
            self._load_hf()
        
        print(f"\033[92m[SUCCESS] LLM Engine initialized in {time.time() - start_time:.2f} seconds!\033[0m")

    def _load_hf(self):
        """
        Standart HuggingFace Transformers engine.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_huggingface import HuggingFacePipeline

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        self.text_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True if self.temperature > 0 else False,
            return_full_text=False,
            stream=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        self.llm = HuggingFacePipeline(pipeline=self.text_pipe)
    
    def _load_gguf(self):
        """
        Llama.cpp engine for GGUF quantized models.
        """
        from langchain_community.llms import LlamaCpp

        stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|im_end|>", "<|im_start|>"]

        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=stop_tokens, 
            n_ctx=4096, # Context window
            n_gpu_layers=-1, # Offload all layers to GPU
            verbose=False, # Trace for debugging
            streaming=True, # Stream the response
        )
    
    def get_llm(self):
        return self.llm
    
    def create_prompt(self, system_message: str, user_message: str) -> PromptTemplate:
        """
        Creates a prompt template for the given model.
        """
        model_name = self.model_path.lower()

        if "qwen" in model_name:
            template = (
                "<|im_start|>system\n"
                "{sys_msg}<|im_end|>\n"
                "<|im_start|>user\n"
                "{usr_msg}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            template = (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                "{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                "{usr_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        
        template_string = template.format(sys_msg=system_message, usr_msg=user_message)
        return PromptTemplate.from_template(template_string)


# TEST
def main(
    model_path: str,
    temperature: float = 0.1,
    max_new_tokens: int = 2048,
    system_message: str = "You are a helpful assistant.",
    user_message: str = None,
):
    llm_engine = LLMEngine(
        model_path=model_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    llm = llm_engine.get_llm()

    prompt = llm_engine.create_prompt(
        system_message=system_message,
        user_message=user_message
    )

    chain = prompt | llm

    start_time = time.time()
    response = chain.invoke({"question": user_message})

    print(f"\033[94mGeneration time: {time.time() - start_time:.2f} seconds\033[0m")

    return response
    

if __name__ == "__main__":
    
    question = str(input(f"\033[94mYour Question:\033[0m "))
    result = main(
        model_path="",
        temperature=0.1,
        max_new_tokens=512,
        system_message="You are a helpful assistant.",
        user_message=question,
    )
    print(f"\n[\033[92mAI Answer\033[0m]: {result}")
    