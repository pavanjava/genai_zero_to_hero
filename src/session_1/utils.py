from dotenv import load_dotenv, find_dotenv
from typing import List
import time
from ollama import chat, ChatResponse

# Initialize global variables
_ = load_dotenv(find_dotenv())


def llama(
        prompt: str,
        add_inst: bool = True,
        model: str = "llama3.2:latest",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        verbose: bool = False,
        base: int = 2,
        max_tries: int = 3
) -> str:
    if add_inst:
        prompt = f"[INST]{prompt}[/INST]"

    if verbose:

        print(f"Prompt:\n{prompt}\n")
        print(f"model: {model}")

    messages = [{"role": "user", "content": prompt}]
    wait_seconds = [base ** i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response: ChatResponse = chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            return response.message.content
        except Exception as e:
            print(f"error message: {e}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])

    print(f"Tried {max_tries} times to make API call to get a valid response object")
    return ""


def llama_chat(
        prompts: List[str],
        responses: List[str],
        model: str = "llama3.2",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        verbose: bool = False,
        base: int = 2,
        max_tries: int = 3
) -> str:
    messages = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])

    if len(prompts) > len(responses):
        messages.append({"role": "user", "content": prompts[-1]})

    wait_seconds = [base ** i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response: ChatResponse = chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            return response.message.content
        except Exception as e:
            print(f"error message: {e}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])

    print(f"Tried {max_tries} times to make API call to get a valid response object")
    return ""
