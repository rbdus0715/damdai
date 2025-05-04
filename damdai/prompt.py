from typing import List, Literal, TypedDict
from transformers import PreTrainedTokenizer

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

MessageList = List[Message]

BEGIN_INST, END_INST = "[INST] ", " [/INST] "
BEGIN_SYS, END_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def convert_list_of_message_lists_to_input_prompt(list_of_message_lists: List[MessageList], tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Convert a list of message lists to input prompts for the model.

    Args:
        list_of_message_lists (List[MessageList]): A list of message lists.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding.

    Returns:
        List[str]: A list of input prompts.
    """
    input_prompts: List[str] = []
    print(type(list_of_message_lists))
    print(type(list_of_message_lists[0]))    
    for message_list in list_of_message_lists:
        if message_list[0]["role"] == "system":
            content = "".join([BEGIN_SYS, message_list[0]["content"], END_SYS, message_list[1]["content"]])
            message_list = [{"role": message_list[1]["role"], "content": content}] + message_list[2:]

        if not (
            all([msg["role"] == "user" for msg in message_list[::2]])
            and all([msg["role"] == "assistant" for msg in message_list[1::2]])
        ):
            raise ValueError(
                "Format must be in this order: 'system', 'user', 'assistant' roles.\nAfter that, you can alternate between user and assistant multiple times"
            )

        eos = tokenizer.eos_token
        bos = tokenizer.bos_token
        input_prompt = "".join(
            [
                "".join([bos, BEGIN_INST, (prompt["content"]).strip(), END_INST, (answer["content"]).strip(), eos])
                for prompt, answer in zip(message_list[::2], message_list[1::2])
            ]
        )

        if message_list[-1]["role"] != "user":
            raise ValueError(f"Last message must be from user role. Instead, you sent from {message_list[-1]['role']} role")

        input_prompt += "".join([bos, BEGIN_INST, (message_list[-1]["content"]).strip(), END_INST])

        input_prompts.append(input_prompt)

    return input_prompts


