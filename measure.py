import gc
import logging
import os
import time
import torch
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoTokenizer
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.data import load_data
from utils.models import has_tied_weights
from utils.clip import magnitude_clip
from utils.device import clear_device_cache, get_preferred_device, resolve_device_map, synchronize_device

log = logging.getLogger("measure")


class TqdmLoggingHandler(logging.Handler):
    """Write log records through tqdm.write so they don't corrupt progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            print(self.format(record))


def setup_logging(log_file: str | None = None, verbose: bool = False) -> None:
    """Log progress to stderr, and to `log_file` when provided."""
    handlers: list[logging.Handler] = [TqdmLoggingHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def fmt_duration(seconds: float) -> str:
    """Render elapsed seconds as mm:ss or h:mm:ss."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def log_vram(stage: str = "") -> None:
    """Log current/peak VRAM usage when running on CUDA."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
        suffix = f" [{stage}]" if stage else ""
        log.info(f"VRAM{suffix}: allocated {allocated:.2f} GiB, peak {peak:.2f} GiB")


def welford_gpu_batched_multilayer_float32(
    formatted_prompts: list[str],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_indices: list[int],
    position: int = 1,
    batch_size: int = 1,
    clip: float = 1.0,
    processor = None,  # Add processor parameter
    is_vision_model: bool = False,  # Add flag for vision models
) -> dict[int, torch.Tensor]:
    text_config = model.config
    if hasattr(text_config, "text_config"):
        text_config = text_config.text_config
    vocab_size = text_config.vocab_size

    max_tokens = position

    means = {layer_idx: None for layer_idx in layer_indices}
    counts = {layer_idx: 0 for layer_idx in layer_indices}
    dtype = model.dtype

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if is_vision_model and processor is not None:
        processor.tokenizer.padding_side = 'left'

    num_batches = max(1, (len(formatted_prompts) + batch_size - 1) // batch_size)
    log_interval = max(1, num_batches // 10)  # roughly 10 progress updates per pass
    log.info(f"{desc}: {len(formatted_prompts)} prompts | batch size {batch_size} | {num_batches} batches")
    pass_start = time.time()

    for batch_num, i in enumerate(tqdm(range(0, len(formatted_prompts), batch_size), desc=desc), start=1):
        batch_start = time.time()
        batch_prompts = formatted_prompts[i:i+batch_size]

        if is_vision_model and processor is not None:
            # For vision models, use the processor with text-only input
            batch_encoding = processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
            )
        else:
            # For text-only models, use the tokenizer
            batch_encoding = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
            )
        
        batch_input = batch_encoding['input_ids'].to(model.device)
        batch_mask = batch_encoding['attention_mask'].to(model.device)

        # Use generate to get hidden states at the first generated token position
        raw_output = model.generate(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
#            do_sample=False,                       # disable sampling
#            top_k=None,
#            top_p=None,
#            cache_implementation=None,
        )
        
        #last_non_pad = batch_mask.sum(dim=1) - 1  # shape: (batch,)
        del batch_input, batch_mask
        #hidden_states = raw_output.hidden_states[max_tokens-1] # Generation step
        hidden_states = [
            layer_tensor.detach().clone() 
            for layer_tensor in raw_output.hidden_states[-1]
        ]
        del raw_output
        #last_non_pad = last_non_pad.to(hidden_states.device)

        # Process layers with Welford in float32
        for layer_idx in layer_indices:
            # Cast to float32 for accumulation
            # Index each sample at its own last non-pad position
#            current_hidden = hidden_states[layer_idx][
#                torch.arange(hidden_states[layer_idx].size(0), device=hidden_states[layer_idx].device),
#                last_non_pad,
#                :
#            ].float() # (batch, hidden)
            # only examine state after last generated position
            current_hidden = hidden_states[layer_idx][:, -1, :].double()
            #current_hidden = hidden_states[layer_idx][:, pos, :].float()
            if (clip < 1.0):
                current_hidden = magnitude_clip(current_hidden, clip)

            batch_size_actual = current_hidden.size(dim=0)
            total_count = counts[layer_idx] + batch_size_actual

            if means[layer_idx] is None:
                # Initialize mean in float64
                means[layer_idx] = current_hidden.double().mean(dim=0)
            else:
                # All operations in float64 (means[layer_idx] is already float64)
                batch_mean = current_hidden.double().mean(dim=0)
                delta = batch_mean - means[layer_idx]
                means[layer_idx] += delta * batch_size_actual / total_count

                #delta = current_hidden.double() - means[layer_idx]
                #means[layer_idx] += delta.sum(dim=0) / total_count

            counts[layer_idx] = total_count
            del current_hidden

        del hidden_states
        #del last_non_pad
        clear_device_cache()

        batch_elapsed = time.time() - batch_start
        elapsed = time.time() - pass_start
        log.debug(f"{desc}: batch {batch_num}/{num_batches} took {batch_elapsed:.2f}s")
        if batch_num % log_interval == 0 or batch_num == num_batches:
            rate = batch_num / elapsed if elapsed > 0 else 0.0
            eta = (num_batches - batch_num) / rate if rate > 0 else 0.0
            log.info(
                f"{desc}: {batch_num}/{num_batches} ({batch_num / num_batches:.0%}) | "
                f"{batch_elapsed:.1f}s/batch | elapsed {fmt_duration(elapsed)} | ETA {fmt_duration(eta)}"
            )
            log_vram(desc)

    log.info(f"{desc}: finished in {fmt_duration(time.time() - pass_start)}")

    # Move to CPU
    return_dict = {
        layer_idx: mean.to(device="cpu")
        for layer_idx, mean in means.items()
    }
    del means
    clear_device_cache()
    return return_dict

def format_chats(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt_list: list[str],
    processor = None,
):
    # Use processor's tokenizer if available, otherwise use tokenizer directly
    actual_tokenizer = processor.tokenizer if processor is not None else tokenizer
    
    result_formatted = [
        actual_tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for inst in prompt_list
    ]
    return result_formatted

def compute_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    harmful_list: list[str],
    harmless_list: list[str],
    projected: bool = False,
    inference_batch_size: int = 32,
    clip: float = 1.0,
    processor = None,  # processor parameter
    is_vision_model: bool = False,  # flag for vision models
    token2: bool = False, # measure at second token instead of first
) -> torch.Tensor:
    # dtype = model.dtype
    if hasattr(model, "language_model"):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, "language_model"):
            layer_base = layer_base.language_model
    num_layers = len(layer_base.layers)

    pos = 1
    if token2:
        pos = 2

    focus_layers = range(num_layers) # sweep all layers
    log.info(f"Measuring {num_layers} layers at token position {pos} (clip={clip}, projected={projected})")

    phase_start = time.time()
    harmful_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmful_list, processor=processor)
    harmful_means = welford_gpu_batched_multilayer_float32(
        harmful_formatted, "Generating harmful outputs", model, tokenizer,
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model
    )
    log.info(f"Harmful pass finished in {fmt_duration(time.time() - phase_start)}")
    clear_device_cache()
    del harmful_formatted
    phase_start = time.time()
    harmless_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmless_list, processor=processor)
    harmless_means = welford_gpu_batched_multilayer_float32(
        harmless_formatted, "Generating harmless outputs", model, tokenizer, 
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model
    )
    log.info(f"Harmless pass finished in {fmt_duration(time.time() - phase_start)}")
    del harmless_formatted

    results = {}
    results["layers"] = num_layers

    # Keep all results in 32-bit float for analysis/ablation
    for layer in tqdm(focus_layers,desc="Compiling layer measurements"):
        harmful_mean = harmful_means[layer]
        results[f'harmful_{layer}'] = harmful_mean.to(dtype=model.dtype)
        harmless_mean = harmless_means[layer]
        results[f'harmless_{layer}'] = harmless_mean.to(dtype=model.dtype)

        harmful_d = harmful_mean.double()
        harmless_d = harmless_mean.double()

        # Compute raw difference of means in float64 to avoid cancellation at high cosine similarity.
        # Saved unnormalized — normalization is deferred to the ablation phase.
        # Note: once unit-normalized, harmful_hat - harmless_hat is exactly the normal of the
        # Householder reflector that maps harmless_hat onto harmful_hat, giving this direction
        # a clean geometric justification beyond naive contrastive difference of means.

        refusal_dir = harmful_d - harmless_d
        results[f'refuse_{layer}'] = refusal_dir.to(dtype=model.dtype)

        # Householder-inspired alternative of computing difference after normalization
        # Unfortunately, it is inferior numerically even if analytically correct
        #harmful_hat = torch.nn.functional.normalize(harmful_d, dim=0)
        #harmless_hat = torch.nn.functional.normalize(harmless_d, dim=0)
        #refusal_dir = harmful_hat - harmless_hat

        if projected: # semantic meaning: preserve activations along the harmless direction
            # Compute Gram-Schmidt second orthogonal vector/direction to remove harmless direction interference from refusal direction
            # Two-pass Gram-Schmidt — second pass catches residual from float cancellation
            harmless_hat = torch.nn.functional.normalize(harmless_d, dim=0)
            refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat
            refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat

        refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=0)

        results[f'refusenorm_{layer}'] = refusal_dir.to(dtype=model.dtype)

    clear_device_cache()
    gc.collect()
    return results


def clean_up() -> None:
    """
    Release VRAM/RAM after measurement is complete.

    Call this after deleting model/tokenizer/results in your code:
        del model, tokenizer, processor, results
        clean_up()

    Note: Callers must delete their own references to objects before calling
    this function. Python's scoping rules mean we cannot delete caller's
    variables from within a function.
    """
    gc.collect()
    synchronize_device()
    clear_device_cache()
    gc.collect()  # Second pass for any refs broken by cache clear
    log.info("Memory cleared successfully.")


def debug_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        inp = input[0] if isinstance(input, tuple) else input
        inp_max = inp.abs().max().item()
        out_max = t.abs().max().item() if not torch.isnan(t).any() else float('nan')
        print(f"Layer {name}: input_max={inp_max:.4f} | output_max={out_max:.4f}")
    return hook

def get_hf_token():
    """Read HF token from env HF_TOKEN, else from local hf_token.txt (one line, not committed)."""
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    for fn in ("hf_token.txt", os.path.join(os.path.dirname(__file__), "hf_token.txt")):
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None

if __name__ == "__main__":
    parser = ArgumentParser(description="Measure models for analysis and abliteration")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        required=True,
        help="Local model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--quant-measure", "-q",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Perform measurement using 4bit or 8bit bitsandbytes quant"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size during inference/calibration; default 32, stick to powers of 2 (higher will use more VRAM)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        required=True,
        help="Output file for measurements"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Fraction of prompt activation to clip by magnitude",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--data-harmful",
        type=str,
        default=None,
        help="Harmful prompts file"
    )
    parser.add_argument(
        "--data-harmless",
        type=str,
        default=None,
        help="Harmless prompts file"
    )
    parser.add_argument(
        "--deccp",
        action="store_true",
        default=False,
        help="For Chinese models, add topics to harmful prompts",
    )
    parser.add_argument(
        "--projected",
        action="store_true",
        default=False,
        help="Remove projection along harmless direction from contrast direction",
    )
    parser.add_argument(
        "--token2",
        action="store_true",
        default=False,
        help="Measure after second token instead of after first token",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Also write progress logs to this file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Log every batch instead of periodic progress updates",
    )

    args = parser.parse_args()

    setup_logging(args.log_file, args.verbose)
    log.info("measure.py starting")
    log.info(
        f"model={args.model} | output={args.output} | quant={args.quant_measure or 'none'} | "
        f"batch_size={args.batch_size} | clip={args.clip} | flash_attn={args.flash_attn} | "
        f"projected={args.projected} | token2={args.token2} | deccp={args.deccp}"
    )

    assert (
        isinstance(args.model, str)
        and
        isinstance(args.output, str)
    )

    torch.inference_mode()
    torch.set_grad_enabled(False)

    device = get_preferred_device()
    device_map = resolve_device_map()
    log.info(
        f"device={device} | device_map={device_map} | torch={torch.__version__} | "
        f"cuda={torch.version.cuda or 'not available'}"
    )
    if device != "cuda":
        log.warning("Running without CUDA — measurement will be very slow.")

    hf_token = get_hf_token()
    log.info(f"HF token: {'found' if hf_token else 'not found (gated models may fail)'}")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token  # also cover implicit Hub access
    model = args.model
    model_config = AutoConfig.from_pretrained(model, token=hf_token)
    model_type = getattr(model_config,"model_type")
    log.info(f"Config loaded: model_type={model_type} | vision={hasattr(model_config, 'vision_config')}")

    # Get the precision/dtype from config, with proper fallback
    if hasattr(model_config, "torch_dtype") and model_config.torch_dtype is not None:
        precision = model_config.torch_dtype
    elif hasattr(model_config, "dtype") and model_config.dtype is not None:
        precision = model_config.dtype
    else:
        # Fallback to bfloat16 on CUDA (if supported), otherwise float32 on MPS/CPU, float16 on CUDA
        if device == "cuda" and torch.cuda.is_bf16_supported():
            precision = torch.bfloat16
        elif device == "cuda":
            precision = torch.float16
        else:
            precision = torch.float32

    has_vision = False
    if hasattr(model_config,"vision_config"):
        has_vision = True
    model_loader = AutoModelForCausalLM
    if (has_vision):
        model_loader = AutoModelForImageTextToText

    quant_config = None
    qbit = args.quant_measure

    if device == "mps" and qbit:
        print("BitsAndBytes quantization is not supported on MPS; disabling requested quantization.")
        qbit = None

    # autodetect BitsAndBytes quant; overrides option
    if hasattr(model_config,"quantization_config"):
        bnb_config = getattr(model_config, "quantization_config")
        if (bnb_config["load_in_4bit"] == True):
            if device == "mps":
                raise RuntimeError("BitsAndBytes 4-bit models are not supported on MPS. Please use CPU/CUDA or load full-precision weights.")
            qbit = "4bit"
            # Override precision with compute dtype from quant config if available
            if "bnb_4bit_compute_dtype" in bnb_config and bnb_config["bnb_4bit_compute_dtype"]:
                precision = bnb_config["bnb_4bit_compute_dtype"]
        elif (bnb_config["load_in_8bit"] == True):
            if device == "mps":
                raise RuntimeError("BitsAndBytes 8-bit models are not supported on MPS. Please use CPU/CUDA or load full-precision weights.")
            qbit = "8bit"

    # Convert string dtype to torch dtype if needed
    if isinstance(precision, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        precision = dtype_map.get(
            precision,
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32,
        )

    if qbit == "4bit":
        print(f"Using compute dtype from quant config: {precision}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", # better for QLoRA
        )
    elif qbit == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
#            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=False,
#            llm_int8_threshold=6.0,
        )    

    if isinstance(args.data_harmful, str):
        harmful_list = load_data(args.data_harmful)
    else:
        harmful_list = load_data("./data/harmful.parquet")
    if isinstance(args.data_harmless, str):
        harmless_list = load_data(args.data_harmless)
    else:
        harmless_list = load_data("./data/harmless.parquet")

    log.info(f"Loaded {len(harmful_list)} harmful and {len(harmless_list)} harmless prompts")

    if args.deccp:
        deccp_list = load_dataset("augmxnt/deccp", split="censored", token=hf_token)
        harmful_list += deccp_list["text"]
        log.info(f"Added {len(deccp_list)} deccp prompts -> {len(harmful_list)} harmful prompts")

    attn_impl = "flash_attention_2" if args.flash_attn and device == "cuda" else None

    log.info(
        f"Loading weights: dtype={precision} | quant={qbit or 'none'} | "
        f"attn={attn_impl or 'default'} | device_map={device_map}"
    )
    load_start = time.time()

    if hasattr(model_config, "quantization_config"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
#            trust_remote_code=True,
            dtype=precision,
            device_map=device_map,
            attn_implementation=attn_impl,
            token=hf_token,
        )
    else:
        model = model_loader.from_pretrained(
            args.model,
#            trust_remote_code=True,
            dtype=precision,
            low_cpu_mem_usage=True,
            device_map=device_map,
            quantization_config=quant_config,
            attn_implementation=attn_impl,
            token=hf_token
        )
    model.requires_grad_(False)
    log.info(f"Model loaded in {fmt_duration(time.time() - load_start)} | dtype={model.dtype}")
    log_vram("after model load")
    if has_tied_weights(model_type):
        model.tie_weights()

    # point to base of language model
    if hasattr(model, "language_model"):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, "language_model"):
            layer_base = layer_base.language_model


#    for i, layer in enumerate(layer_base.layers):
#        layer.register_forward_hook(debug_hook(i))

    #print(layer_base.embed_tokens.weight.dtype)
    #print(layer_base.config.hidden_size) 

    if qbit == "4bit": # stabilize for Gemma 3, possibly other models
        layer_base.embed_tokens = layer_base.embed_tokens.to(precision)
        layer_base.norm = layer_base.norm.to(precision)
    # Gemma 3 still needs Winsorization to not explode!

    # Load processor for vision models, tokenizer for text-only models
    processor = None
    if has_vision:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model,
                device_map=device_map,
                padding=True,
                token=hf_token,
            )
            tokenizer = processor.tokenizer
            log.info("Loaded processor for vision model")
        except (IndexError, Exception) as e:
            # If processor loading fails, fall back to tokenizer only
            log.warning(f"Could not load processor ({e}), falling back to tokenizer only")
            has_vision = False
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
#                trust_remote_code=True,
                device_map=device_map,
                padding=True,
                token=hf_token,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
#            trust_remote_code=True,
            device_map=device_map,
            padding=True,
            token=hf_token,
        )
        log.info("Loaded tokenizer")

    log.info("Computing refusal information...")
    measure_start = time.time()
    results = {}
    results = compute_refusals(
        model=model,
        tokenizer=tokenizer,
        harmful_list=harmful_list,
        harmless_list=harmless_list,
        projected=args.projected,
        inference_batch_size=args.batch_size,
        clip=args.clip,
        processor=processor,
        is_vision_model=has_vision,
        token2=args.token2,
    )

    log.info(f"Measurement finished in {fmt_duration(time.time() - measure_start)}")

    log.info(f"Saving refusal information to {args.output}...")
    torch.save(results, args.output)
    if os.path.exists(args.output):
        log.info(f"Saved {os.path.getsize(args.output) / 2**20:.1f} MiB to {args.output}")

    # Release VRAM so next measurement can start immediately
    log.info("Unloading model and clearing memory...")
    del model, tokenizer, processor, results
    clean_up()
    log.info("Done.")
