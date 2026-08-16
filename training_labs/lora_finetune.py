"""Fine-tune a small causal language model with a real LoRA adapter.

WHAT PROBLEM DOES THIS SOLVE?

A pretrained language model such as SmolLM2-135M has already learned general
English from a huge amount of text. Suppose you want it to answer in a
particular style, or to know facts about your own domain. The obvious approach
is "full fine-tuning": keep training every number inside the model on your own
examples. That works, but every one of those numbers needs a stored gradient
and optimizer state during training, so even a small model becomes expensive,
and you end up with a complete second copy of the model on disk.

LoRA (Low-Rank Adaptation) is the cheap alternative. It *freezes* the original
model and trains a small set of extra numbers bolted onto some of its layers.

HOW LoRA WORKS

Inside the model are many "linear layers". A linear layer is just a big grid of
numbers, called a weight matrix W, that the input gets multiplied by:

    output = W @ x          ("@" means matrix multiply)

If W is 4096 x 4096, that is about 16.8 million numbers to train. LoRA leaves W
untouched and adds a detour made of two much smaller matrices, A and B:

    x ──┬──────────────► W ──────────────┐   W is FROZEN (never updated)
        │                                │
        │                                ▼
        └──► A ──► (r numbers) ──► B ──► + ──► output
             ↑                     ↑
        4096 x r                r x 4096      A and B are TRAINABLE

The middle is deliberately a bottleneck of width `r` (the "rank", 8 by
default). With r=8 the detour holds 4096*8 + 8*4096 = 65,536 numbers instead of
16.8 million -- roughly 0.4% as many. The bet LoRA makes is that the *change*
you need for a new task is simple enough to squeeze through that bottleneck,
even though the original model is not.

Because W never changes, what you save at the end is only A and B: a few
megabytes, called an "adapter". You still need the original model to use it.

WHAT THIS SCRIPT DOES

Loads a small instruct-tuned model, wraps it in LoRA layers, trains on a
handful of instruction/response examples, and saves the adapter. It is a
learning run: 24 examples and 20 steps is far too little to produce a genuinely
useful model, but it exercises every real step of the pipeline.

Further reading:
- LoRA paper: https://arxiv.org/abs/2106.09685
- PEFT quicktour: https://huggingface.co/docs/peft/quicktour
- Causal language modeling: https://huggingface.co/docs/transformers/en/tasks/language_modeling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# The bundled example data. `parents[1]` climbs from training_labs/ up to the
# repository root, so this works no matter which directory you run from.
DEFAULT_DATA = Path(__file__).parents[1] / "data" / "lora_tiny_instructions.jsonl"


def choose_device(torch) -> str:
    """Pick the fastest piece of hardware available for training.

    A "device" is where the arithmetic actually happens. The same code runs on
    all three, just at very different speeds.

    Args:
        torch: The imported `torch` module. It is passed in rather than
            imported at the top of this file because PyTorch is an optional
            dependency here (see the note in `main`).

    Returns:
        One of:
        - "cuda": an NVIDIA graphics card. Fastest by a wide margin.
        - "mps":  Apple's Metal Performance Shaders, i.e. the GPU built into an
                  Apple Silicon Mac. Good enough for this small lab.
        - "cpu":  the ordinary processor. Always works, but slowest.
    """
    if torch.cuda.is_available():
        return "cuda"
    # `getattr` guards against very old PyTorch builds that have no `mps`
    # attribute at all, where a plain `torch.backends.mps` would raise.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_local_records(path: Path, limit: int) -> list[dict[str, str]]:
    """Read training examples from a local JSON Lines file.

    JSON Lines ("jsonl") is a text format holding one complete JSON object per
    line. It is convenient for training data because you can read it one line
    at a time without parsing the whole file.

    Args:
        path: File to read. Each line must be a JSON object with the keys
            "instruction", "input", and "output".
        limit: Keep at most this many examples, counting from the top of the
            file. Small values make a training run finish quickly.

    Returns:
        A list of example dictionaries, each with "instruction" (the request),
        "input" (extra context, often an empty string), and "output" (the
        response the model should learn to produce).
    """
    with path.open(encoding="utf-8") as handle:
        # `if line.strip()` skips blank lines, which are legal whitespace in a
        # text file but would crash json.loads.
        records = [json.loads(line) for line in handle if line.strip()]
    return records[:limit]


def load_hub_records(name: str, limit: int) -> list[dict[str, str]]:
    """Download a short prefix of a public dataset from the Hugging Face Hub.

    Public instruction datasets are often gigabytes in size. `streaming=True`
    pulls examples over the network one at a time instead of downloading the
    whole archive first, so asking for 200 examples costs seconds rather than a
    long download.

    Args:
        name: Dataset identifier on the Hub, for example "yahma/alpaca-cleaned".
        limit: How many examples to pull from the beginning of the stream.

    Returns:
        Example dictionaries in the same shape as `load_local_records`, so the
        rest of the script does not care where the data came from.

    Note:
        Datasets vary in field naming. The `.get(a, .get(b))` calls accept
        either the "instruction"/"output" convention or the "prompt"/"response"
        one, and fall back to an empty string when neither is present.

    Dataset viewer and licenses: https://huggingface.co/docs/datasets/
    """
    from datasets import load_dataset

    stream = load_dataset(name, split="train", streaming=True)
    records = []
    for row in stream.take(limit):
        records.append(
            {
                "instruction": str(row.get("instruction", row.get("prompt", ""))),
                "input": str(row.get("input", "")),
                "output": str(row.get("output", row.get("response", ""))),
            }
        )
    return records


def format_prompt(user_text: str) -> str:
    """Return everything the model should see before it starts its own reply.

    This is the half of a training example that stays the same at generation
    time. Training appends the real answer after it; `lora_inference.py` stops
    here and lets the model fill in the rest. Keeping both callers on this one
    function is what stops the two halves of the lab from drifting apart -- a
    model prompted in a shape it never saw during training has no cue that a
    reply is wanted, and tends to emit its end-of-sequence token immediately.

    Args:
        user_text: The request, already including any extra context.

    Returns:
        The formatted prefix, ending with a newline after the assistant marker
        so the model's first generated token begins the answer itself.
    """
    return f"### User:\n{user_text}\n\n### Assistant:\n"


def format_record(record: dict[str, str], eos_token: str) -> str:
    """Turn one example into the single block of text the model trains on.

    A language model does not see "question" and "answer" as separate things.
    It only ever sees a stream of text and learns to continue it. So training
    data has to be flattened into one string, with markers the model can learn
    to recognize as "the request stops here, my reply starts here".

    Args:
        record: A dictionary with "instruction", optional "input", and "output".
        eos_token: The tokenizer's end-of-sequence string, appended after the
            answer. Pass `tokenizer.eos_token`. It is a required argument rather
            than an optional one because leaving it off is silent: training
            still runs, and the damage only shows up later as a model that
            never stops generating.

    Returns:
        One formatted string. For example, this record::

            {"instruction": "Expand the acronym GPU.",
             "input": "",
             "output": "GPU means graphics processing unit."}

        becomes (with the end-of-sequence token written here as ``</s>``)::

            ### User:
            Expand the acronym GPU.

            ### Assistant:
            GPU means graphics processing unit.</s>

        When "input" is non-empty it is appended under the instruction, so a
        record asking to summarize a paragraph keeps the paragraph attached to
        the request rather than to the answer.

    Important:
        The exact markers do not matter, but consistency does. Training with
        "### Assistant:" and then prompting with something else at generation
        time gives the model no cue that it is supposed to reply. That is why
        the prefix lives in `format_prompt`, which generation shares.

        The end-of-sequence token is what teaches the model where an answer
        *ends*. Without it every training example runs straight off the end of
        the answer into nothing, the model never learns to stop, and generation
        rambles until it hits the token limit -- often by repeating the "###
        Assistant:" marker over and over.
    """
    user_text = record["instruction"]
    if record.get("input"):
        user_text += f"\n\nInput:\n{record['input']}"
    return format_prompt(user_text) + record["output"] + eos_token


def main() -> None:
    """Run one end-to-end LoRA fine-tuning exercise from the command line.

    The stages are: parse options, load data, attach LoRA layers to a frozen
    base model, loop over examples updating only the LoRA weights, then save
    the adapter. Nothing is returned; progress and the output path are printed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Which pretrained model to adapt. 135M parameters is tiny by modern
    # standards, which is exactly why it fits comfortably on a laptop.
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--hf-dataset", help="optional Hugging Face dataset name")
    parser.add_argument("--max-examples", type=int, default=24)
    # Longer text costs memory roughly linearly, so this is the first knob to
    # turn down if the machine runs out of memory.
    parser.add_argument("--max-length", type=int, default=256)
    # One "step" here means one example processed. Not one pass over the data.
    parser.add_argument("--steps", type=int, default=20)
    # How big a correction to make after each measurement. Too large and
    # training becomes unstable; too small and nothing visibly changes.
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lora-adapter"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # These libraries are optional extras, not base dependencies of this
    # repository, so they are imported here rather than at the top of the file.
    # Importing at module level would break `import training_labs` for anyone
    # who installed only the small base environment.
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install the LoRA dependencies with: uv sync --extra lora") from error

    records = (
        load_hub_records(args.hf_dataset, args.max_examples)
        if args.hf_dataset
        else load_local_records(args.data, args.max_examples)
    )
    if not records:
        raise SystemExit("No training records were loaded")
    device = choose_device(torch)
    print(f"device: {device}; model: {args.model}; examples: {len(records)}")
    if args.dry_run:
        # Stop before anything is downloaded. This lets you confirm the install
        # and your arguments are correct without waiting on network transfers.
        print("dry run complete; no model was downloaded and no training was performed")
        return

    # ---- Tokenizer -------------------------------------------------------
    # Models cannot read text. A tokenizer chops a string into "tokens" (common
    # word fragments) and maps each to an integer id:
    #
    #     "graphics processing unit"  ->  ["graphics", " processing", " unit"]
    #                                 ->  [17692, 8863, 4326]
    #
    # The tokenizer must match the model, which is why it is loaded from the
    # same name. https://huggingface.co/docs/transformers/en/tokenizer_summary
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        # "Padding" fills short sequences so a batch forms a neat rectangle.
        # Some models ship without a dedicated pad token; reusing the
        # end-of-sequence token is the standard workaround. It is harmless here
        # because this loop feeds exactly one example at a time, so no padding
        # is actually inserted.
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Base model ------------------------------------------------------
    # "CausalLM" means a model that predicts the next token given everything
    # before it -- as opposed to models that can see text on both sides.
    model = AutoModelForCausalLM.from_pretrained(args.model)
    # The KV cache speeds up *generation* by remembering earlier tokens' work.
    # During training we process a whole sequence at once and need gradients,
    # so the cache only wastes memory. It is turned back on in lora_inference.
    model.config.use_cache = False

    # ---- Attach the LoRA adapter ----------------------------------------
    # get_peft_model freezes every original weight and inserts the trainable
    # A/B detour matrices described in this module's docstring.
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # `r` is the bottleneck width from the diagram above. Higher means
            # more capacity to learn and more numbers to train. 8 is a common
            # starting point; try 4 and 16 to see the tradeoff.
            r=8,
            # The adapter's output is scaled by lora_alpha / r before being
            # added back. With alpha=16 and r=8 that is a factor of 2. Keeping
            # alpha near twice r is a widely used convention.
            lora_alpha=16,
            # Randomly ignores 5% of the adapter's inputs during training. This
            # discourages the model from leaning on any single pathway, which
            # helps it generalize instead of memorizing the training examples.
            lora_dropout=0.05,
            # Which layers get an adapter. "all-linear" attaches one to every
            # linear layer found, which is the simplest choice that works well.
            target_modules="all-linear",
        ),
    )
    model.to(device)
    # Switches on training-time behavior such as dropout. `.eval()` is its
    # counterpart and is used at generation time.
    model.train()
    # Prints something like "trainable params: 4,884,480 || all params:
    # 139,400,064 || trainable%: 3.5" -- the concrete payoff of using LoRA.
    model.print_trainable_parameters()

    # ---- Convert every example to token ids ------------------------------
    # Done once up front rather than inside the loop, since the same examples
    # get revisited and tokenizing is pure repeated work.
    encoded = [
        tokenizer(
            format_record(record, tokenizer.eos_token),
            # Cut anything longer than max_length rather than erroring. Long
            # examples lose their tail, which is a real (if quiet) data loss --
            # and note that the tail is where the end-of-sequence token lives,
            # so a truncated example teaches the model nothing about stopping.
            # The bundled data tops out around 44 tokens, well under the limit,
            # but a long --hf-dataset example can hit this.
            truncation=True,
            max_length=args.max_length,
            # "pt" = PyTorch tensors, the array type the model expects.
            return_tensors="pt",
        )
        for record in records
    ]

    # ---- Optimizer -------------------------------------------------------
    # The optimizer is what actually adjusts numbers once we know which
    # direction each should move. AdamW keeps a per-number running average of
    # recent gradients, which makes progress smoother than naive descent.
    # Note the filter: only parameters with requires_grad=True are handed over,
    # which after get_peft_model means only the LoRA matrices. The frozen base
    # model is not given to the optimizer at all.
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)

    # ---- Training loop ---------------------------------------------------
    # Each step does four things:
    #   1. forward pass  -- run an example through and measure how wrong it is
    #   2. backward pass -- work out which direction each trainable number
    #                       should move to reduce that wrongness ("gradients")
    #   3. accumulate    -- add those directions onto a running total
    #   4. occasionally  -- apply the accumulated total and reset
    #
    # Step 4 does not happen every time. That is "gradient accumulation":
    #
    #     step 1   step 2   step 3   step 4        step 5 ...
    #     grad     grad     grad     grad
    #      \        |        |        /
    #       +-------+--------+-------+
    #                  |
    #             one weight update
    #
    # Updating from four examples at once gives a steadier estimate of the
    # right direction than updating from one, but unlike a real batch it needs
    # memory for only one example at a time. That is the whole trick: it buys
    # batch-like stability on hardware that cannot hold a batch.
    for step in range(args.steps):
        # Cycle through the examples, wrapping around when steps exceed the
        # number of examples. With 20 steps and 24 examples, four are unseen.
        item = encoded[step % len(encoded)]
        input_ids = item["input_ids"].to(device)
        # Marks which positions are real tokens versus padding, so the model
        # ignores filler. Always all-ones here, since we feed one example.
        attention_mask = item["attention_mask"].to(device)

        # Passing `labels=input_ids` is how next-token training is expressed.
        # The model internally shifts the labels by one position, so at every
        # spot it is graded on predicting the token that actually came next:
        #
        #     input:   ###  User  :  Expand  the  acronym  GPU
        #     target:  User   :  Expand  the  acronym  GPU   .
        #
        # The returned `loss` is a single number: how surprised the model was
        # by the true continuation. Lower is better. Dividing by the
        # accumulation count keeps the summed gradients at the same overall
        # scale as a single-example step, so the learning rate still means
        # what it meant before.
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        ).loss / args.gradient_accumulation

        # Computes gradients and *adds* them to whatever is already stored on
        # each parameter. That additive behavior is what makes accumulation
        # work without any extra bookkeeping.
        loss.backward()

        # Apply the update every Nth step, and also on the final step so the
        # last partial group is not silently thrown away.
        if (step + 1) % args.gradient_accumulation == 0 or step + 1 == args.steps:
            optimizer.step()
            # Clear the accumulator for the next group. `set_to_none=True`
            # frees the memory rather than filling it with zeros.
            optimizer.zero_grad(set_to_none=True)

        # Multiply back so the printed number is the true per-example loss,
        # not the scaled-down version used for accumulation. Expect it to jump
        # around: with one example per step it is a noisy signal, and the trend
        # over many steps matters far more than any single value.
        print(f"step {step + 1:03d}/{args.steps}: loss={loss.item() * args.gradient_accumulation:.4f}")

    # ---- Save ------------------------------------------------------------
    # Writes only the small A/B matrices, not the frozen base model. The
    # tokenizer is saved alongside so inference does not have to guess which
    # one was used. `artifacts/` is gitignored.
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
