from prompt import *
from transformers import pipeline, GenerationConfig
from transformers import LlamaForCausalLM
import torch


if __name__ == "__main__":
    from transformers import LlamaTokenizer
    model_checkpoint = "NousResearch/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint)

    system_message = Message()
    system_message["role"] = "system"
    system_message["content"] = "Answer only with emojis"

    user_message = Message()
    user_message["role"] = "user"
    # user_message["content"] = "Who won the 2016 baseball World Series?"
    user_message["content"] = """
    QUESTION: 예시 질문(1)
    ANSWER: 예시 답변(1)

    QUESTION: 예시 질문(2)
    ANSWER: 예시 답변(2)

    QUESTION: 예시 질문(3)
    ANSWER: 예시 답변(3)
    """

    list_of_messages = list()
    list_of_messages.append(system_message)
    list_of_messages.append(user_message)

    list_of_message_lists = list()
    list_of_message_lists.append(list_of_messages)
    
    # 프롬프트 생성
    prompt = convert_list_of_message_lists_to_input_prompt(list_of_message_lists, tokenizer)
    print(prompt)

    # 토큰화
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    print(tokenized_prompt["input_ids"][0])

    # 모델 로드
    model = LlamaForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model = model.eval()

    # 생성
    generation_config = GenerationConfig(max_new_tokens=2000)
    pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    pipeline(prompt, return_full_text=False)