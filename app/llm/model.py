import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from app.config import LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE
from app.logger import get_logger

logger = get_logger()

logger.info(f"Loading LLM model: {LLM_MODEL}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL,
    quantization_config=bnb_config,
    # torch_dtype=torch.float32,
)

model.eval()

logger.info("Flan-T5 model loaded successfully")


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from flan-t5 using a grounded prompt
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
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()


    logger.info(f"LLM inference completed | answer length={len(answer)}")

    return answer