import torch
from transformers import AutoTokenizer, AutoModelForCasualLM

from app.config import LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE
from app.logger import get_logger

logger = get_logger()

logger.info(f"Loading LLM model: {LLM_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCasualLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model.eval()

logger.info("Phi-3 Mini model loaded successfully")


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from Phi-3 using a grounded prompt
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    logger.info(
        f"Starting LLM inference | prompt_length={len(prompt)}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature = TEMPERATURE,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id
        )
    
    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens = True
    )

    # Strip the prompt from the generated output
    answer = decoded[len(prompt):].strip()

    logger.info(f"LLM inference completed | answer length={len(answer)}")

    return answer