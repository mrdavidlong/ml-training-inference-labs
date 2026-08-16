"""Load a saved LoRA adapter and generate a short response.

This is the other half of `lora_finetune.py`. Training produced a small adapter
on disk; this script stacks it back onto the original model and generates text,
which is the only real way to judge whether the fine-tuning did anything useful.

    base model (unchanged, ~135M numbers)
            +
    adapter (a few MB, from artifacts/lora-adapter)
            =
    adapted model that answers in the trained style

Note that the adapter alone is useless -- it is a *modification* to a specific
model, not a model. Both pieces must be present, and the base model must be the
same one used during training.

A useful exercise: run this, then run it again with the same prompt but pointing
`--base-model` at the model without the adapter, and compare. Training loss
going down does not prove the answers got better.

Generation options: https://huggingface.co/docs/transformers/en/main_classes/text_generation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training_labs.lora_finetune import choose_device, format_prompt


def main() -> None:
    """Generate one response from the fine-tuned model and print it.

    Loads the base model, applies the saved adapter, wraps the prompt in the
    same markers the training data used, and prints only the text the model
    generated -- the prompt itself is sliced back off. Nothing is returned.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Must match whatever `--model` was used for training. A mismatch either
    # errors out or produces nonsense, because the adapter's matrices are
    # shaped for one specific set of layers.
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/lora-adapter"))
    parser.add_argument("--prompt", default="Explain why batching helps an inference server.")
    # A cap, not a target. Generation also stops early if the model emits its
    # end-of-sequence token.
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    # Optional dependencies, imported here for the same reason as in
    # lora_finetune.py: the base install of this repository does not have them.
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install the LoRA dependencies with: uv sync --extra lora") from error

    device = choose_device(torch)
    # Loaded from the adapter directory, not the base model name. Training
    # saved a copy there, which guarantees the exact same text-to-number
    # mapping is used now as during training.
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    # Wraps the base model so that every adapted layer adds its LoRA detour.
    # `.eval()` switches off training-only behavior such as dropout, which
    # would otherwise make output randomly vary between runs.
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()

    # The prompt is wrapped in the same "### User:/### Assistant:" markers the
    # training data used. Sending the bare text instead makes the model emit
    # its end-of-sequence token straight away, because nothing in the input
    # resembles the shape it was taught to continue.
    #
    # Text -> token ids, as tensors on the same device as the model. Mixing
    # devices (a CPU input with a GPU model) is a common beginner error and
    # raises rather than silently working.
    inputs = tokenizer(format_prompt(args.prompt), return_tensors="pt").to(device)
    # How many tokens went in, so the prompt can be sliced back off below.
    prompt_length = inputs["input_ids"].shape[1]

    # `no_grad` tells PyTorch not to record the information it would need to
    # compute gradients. We are not training here, so that bookkeeping would be
    # pure wasted memory and time.
    with torch.no_grad():
        # do_sample=False selects the single highest-probability token at every
        # position ("greedy decoding"). The result is deterministic, which is
        # what you want when comparing before-and-after fine-tuning. Turning
        # sampling on would add randomness and muddy the comparison.
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    # `output[0]` picks the first (and here only) generated sequence. It holds
    # the prompt followed by the continuation, so slicing off `prompt_length`
    # leaves just what the model actually wrote. decode maps those token ids
    # back to text; skip_special_tokens hides internal markers such as the
    # end-of-sequence token from the printed result.
    print(tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
