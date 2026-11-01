"""Export an ablated HuggingFace model directory to a GGUF that Ollama can import.

Pipeline:
  1. locate (or shallow-clone) llama.cpp and install its converter deps
  2. convert_hf_to_gguf.py  -> f16 (or f32/bf16/q8_0) GGUF
  3. llama-quantize         -> Q4_K_M / Q5_K_M / ... (optional, needs the binary)
  4. write a Modelfile and optionally run `ollama create`

Usage:
  python export_gguf.py out_gemma3-4b-it --quant Q4_K_M --tag gemma3-4b-abliterated --create
  python export_gguf.py out_gemma3-4b-it --dry-run        # only print what would run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

try:
    from utils.logging import fmt_duration, log, setup_logging
except ImportError:  # allow running from another cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.logging import fmt_duration, log, setup_logging

REPO_URL = "https://github.com/ggml-org/llama.cpp"
# Never let the converter requirements downgrade/replace our CUDA torch build.
SKIP_DEPS = {"torch", "torchvision", "torchaudio", "transformers"}
DEFAULT_DEPS = ["numpy", "protobuf", "sentencepiece"]


def run(cmd: list[str], dry_run: bool = False, cwd: str | None = None) -> None:
    """Run a command, echoing it. In dry-run mode only log it."""
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    if dry_run:
        log.info(f"[dry-run] {printable}")
        return
    log.info(f"$ {printable}")
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}")
    log.info(f"done in {fmt_duration(time.time() - start)}")


def find_llama_cpp(explicit: str | None) -> Path:
    """Resolve the llama.cpp checkout: --llama-cpp, $LLAMA_CPP_DIR, ./llama.cpp, ./vendor/llama.cpp."""
    candidates = [
        explicit,
        os.environ.get("LLAMA_CPP_DIR"),
        str(Path(__file__).resolve().parent / "llama.cpp"),
        str(Path(__file__).resolve().parent / "vendor" / "llama.cpp"),
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand).resolve()
    return Path(__file__).resolve().parent / "llama.cpp"


def ensure_llama_cpp(target: Path, dry_run: bool) -> None:
    """Shallow-clone llama.cpp if it isn't present yet."""
    if (target / "convert_hf_to_gguf.py").exists() or any(target.rglob("convert_hf_to_gguf.py")):
        log.info(f"llama.cpp found at {target}")
        return
    if dry_run:
        log.info(f"[dry-run] would clone {REPO_URL} -> {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Cloning llama.cpp (shallow) into {target} ...")
    run(["git", "clone", "--depth", "1", REPO_URL, str(target)])


def find_converter(llama_dir: Path) -> Path:
    """Locate convert_hf_to_gguf.py (repo root in newer revisions, tools/ in others)."""
    for cand in (llama_dir / "convert_hf_to_gguf.py", llama_dir / "tools" / "convert_hf_to_gguf.py"):
        if cand.exists():
            return cand
    matches = sorted(llama_dir.rglob("convert_hf_to_gguf.py"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {llama_dir}")


def ensure_convert_deps(llama_dir: Path, dry_run: bool) -> None:
    """Install the converter's Python deps, minus torch/transformers (already installed)."""
    pip = [sys.executable, "-m", "pip", "install"]
    req = llama_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"

    if (llama_dir / "gguf-py").exists():
        # Use the repo's own gguf bindings so versions always match.
        run(pip + [str(llama_dir / "gguf-py")], dry_run)

    packages: list[str] = []
    if req.exists():
        for line in req.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            # skip comments, pip options (-r file, --extra-index-url, ...) and blank lines
            if not name or name.startswith("#") or name.startswith("-"):
                continue
            base = re.split(r"[=<>!\[;]", name)[0].strip()
            if base.lower() in SKIP_DEPS:
                log.info(f"Skipping dependency {name} (already provided by this env)")
                continue
            packages.append(name)
    if not packages:
        packages = DEFAULT_DEPS  # upstream requirements only pin torch; keep the rest working
        log.info(f"Using default converter deps: {', '.join(packages)}")

    if packages:
        run(pip + packages, dry_run)

    # Safety net: never let dependency resolution silently downgrade torch to a CPU build.
    try:
        import torch

        if not torch.cuda.is_available():
            log.warning("torch no longer reports CUDA after installing converter deps - GPU runs may break")
    except Exception:
        pass


def find_quantizer(llama_dir: Path, explicit: str | None) -> Path | None:
    """Find llama-quantize(.exe): explicit path, $LLAMA_QUANTIZE, PATH, or a llama.cpp build."""
    for cand in (explicit, os.environ.get("LLAMA_QUANTIZE"), shutil.which("llama-quantize")):
        if cand and Path(cand).exists():
            return Path(cand).resolve()
    patterns = [
        "build/bin/Release/llama-quantize.exe",
        "build/bin/llama-quantize.exe",
        "build/Release/llama-quantize.exe",
        "build/llama-quantize.exe",
        "llama-quantize.exe",
        "build/bin/llama-quantize",
    ]
    for pat in patterns:
        cand = llama_dir / pat
        if cand.exists():
            return cand
    for cand in sorted(llama_dir.rglob("llama-quantize.exe"))[:1]:
        return cand
    return None


def download_quantizer(dest: Path, dry_run: bool) -> Path | None:
    """Download a prebuilt Windows llama.cpp release and return llama-quantize.exe."""
    import urllib.request

    # `releases/latest` is only a nightly pointer these days; scan recent releases for real assets.
    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
    log.info(f"Querying {api}")
    with urllib.request.urlopen(api, timeout=60) as resp:
        releases = json.load(resp)

    asset = None
    source = None
    for release in releases:
        assets = release.get("assets", [])
        for needle in ("win-cpu-x64.zip", "win-cuda-12.4-x64.zip", "win-cuda-13.1-x64.zip"):
            hit = next((a for a in assets if needle in a["name"]), None)
            if hit:
                asset, source = hit, release.get("tag_name")
                break
        if asset:
            break
    if not asset:
        log.warning("No Windows release zip found in recent llama.cpp releases")
        return None
    log.info(f"Found {asset['name']} in release {source}")

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / asset["name"]
    if dry_run:
        log.info(f"[dry-run] would download {asset['browser_download_url']}")
        return zip_path.with_suffix(".exe")
    log.info(f"Downloading {asset['name']} ({asset['size'] / 2**20:.0f} MiB) ...")
    urllib.request.urlretrieve(asset["browser_download_url"], zip_path)
    log.info(f"Extracting to {dest} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    found = next(iter(sorted(dest.rglob("llama-quantize.exe"))), None)
    return found


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_modelfile(gguf: Path, model_dir: Path, tag: str, mmproj: Path | None, out_dir: Path) -> Path:
    """Generate an Ollama Modelfile. Ollama picks up the GGUF's embedded Jinja chat template."""
    gen = read_json(model_dir / "generation_config.json")
    lines = [
        f"# Auto-generated by export_gguf.py for {tag}",
        f'FROM "{gguf.resolve()}"',
    ]
    param_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "repetition_penalty": "repeat_penalty",
    }
    for src, dst in param_map.items():
        value = gen.get(src)
        if isinstance(value, (int, float)):
            lines.append(f"PARAMETER {dst} {value}")

    if mmproj is not None:
        lines += [
            "",
            "# Vision: Ollama cannot attach a projector to a custom GGUF import, so this build",
            "# is text-only. Use the mmproj with llama.cpp / llama-server if you need images:",
            f"#   llama-server -m {gguf.name} --mmproj {mmproj.name}",
        ]

    modelfile = out_dir / "Modelfile"
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"Wrote {modelfile}")
    for line in lines:
        log.info(f"  | {line}")
    return modelfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an ablated HF model to an Ollama-importable GGUF.")
    parser.add_argument("model_dir", type=str, help="Ablated model directory (config.json + *.safetensors)")
    parser.add_argument("--out-dir", "-o", type=str, default=None, help="Output directory (default: <model_dir>-gguf)")
    parser.add_argument("--outtype", type=str, default="f16", choices=["f32", "f16", "bf16", "q8_0"],
                        help="GGUF output type before quantization (default: f16)")
    parser.add_argument("--quant", "-q", type=str, default="Q4_K_M",
                        help="Final quantization for llama-quantize, e.g. Q4_K_M/Q5_K_M/Q8_0. Empty string to skip.")
    parser.add_argument("--tag", type=str, default=None, help="Ollama model name (default: <model_dir>-abliterated)")
    parser.add_argument("--llama-cpp", type=str, default=None, help="Path to an existing llama.cpp checkout")
    parser.add_argument("--llama-quantize", type=str, default=None, help="Path to llama-quantize binary")
    parser.add_argument("--download-quantizer", action="store_true", default=False,
                        help="Download a prebuilt llama-quantize from GitHub releases if none is found")
    parser.add_argument("--no-mmproj", action="store_true", default=False,
                        help="Skip exporting the multimodal projector (text-only models)")
    parser.add_argument("--create", action="store_true", default=False, help="Run `ollama create` after conversion")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Only print the commands that would run")
    parser.add_argument("--skip-deps", action="store_true", default=False, help="Do not install converter dependencies")
    parser.add_argument("--log-file", type=str, default=None, help="Also write logs to this file")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Debug-level logging")
    args = parser.parse_args()

    setup_logging(args.log_file, args.verbose)
    log.info("export_gguf.py starting")
    start = time.time()

    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"{model_dir} does not look like a model directory (no config.json)")
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No .safetensors found in {model_dir}")
    total_gb = sum(s.stat().st_size for s in shards) / 2**30
    log.info(f"Model: {model_dir} | {len(shards)} shard(s), {total_gb:.2f} GiB")

    config = read_json(model_dir / "config.json")
    arch = (config.get("architectures") or ["unknown"])[0]
    is_vision = "vision_config" in config
    log.info(f"architectures={arch} | model_type={config.get('model_type')} | vision={is_vision}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else model_dir.with_name(model_dir.name + "-gguf")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{model_dir.name}-abliterated"
    log.info(f"Output dir: {out_dir} | ollama tag: {tag}")

    # 1. llama.cpp + deps
    llama_dir = find_llama_cpp(args.llama_cpp)
    ensure_llama_cpp(llama_dir, args.dry_run)
    converter = find_converter(llama_dir) if (llama_dir.exists() and not args.dry_run) else llama_dir / "convert_hf_to_gguf.py"
    if not args.skip_deps and not args.dry_run:
        ensure_convert_deps(llama_dir, args.dry_run)
    elif args.skip_deps:
        log.info("Skipping dependency installation (--skip-deps)")

    # 2. convert -> GGUF
    gguf_path = out_dir / f"{model_dir.name}-{args.outtype}.gguf"
    if gguf_path.exists() and gguf_path.stat().st_size > 0:
        log.info(f"Reusing existing GGUF: {gguf_path} ({gguf_path.stat().st_size / 2**30:.2f} GiB)")
    else:
        run([sys.executable, str(converter), str(model_dir), "--outfile", str(gguf_path), "--outtype", args.outtype],
            args.dry_run)
        if gguf_path.exists():
            log.info(f"GGUF written: {gguf_path} ({gguf_path.stat().st_size / 2**30:.2f} GiB)")

    # 2b. multimodal projector (for llama.cpp; Ollama custom imports stay text-only)
    mmproj_path = None
    if is_vision and not args.no_mmproj:
        mmproj_path = out_dir / f"mmproj-{model_dir.name}-f16.gguf"
        if mmproj_path.exists() and mmproj_path.stat().st_size > 0:
            log.info(f"Reusing existing mmproj: {mmproj_path} ({mmproj_path.stat().st_size / 2**20:.0f} MiB)")
        else:
            run([sys.executable, str(converter), str(model_dir), "--mmproj",
                 "--outfile", str(mmproj_path), "--outtype", "f16"], args.dry_run)
        if mmproj_path.exists():
            log.info(f"mmproj written: {mmproj_path} ({mmproj_path.stat().st_size / 2**20:.0f} MiB)")
        else:
            mmproj_path = None

    # 3. quantize
    final_gguf = gguf_path
    quant = (args.quant or "").strip()
    if quant:
        quantizer = find_quantizer(llama_dir, args.llama_quantize)
        if quantizer is None and args.download_quantizer:
            quantizer = download_quantizer(out_dir / "llama-cpp-bin", args.dry_run)
        if quantizer is None:
            log.warning(
                f"No llama-quantize found - keeping {args.outtype} GGUF. "
                f"Re-run with --download-quantizer, or point at one with --llama-quantize <path>."
            )
        else:
            log.info(f"Using quantizer: {quantizer}")
            final_gguf = out_dir / f"{model_dir.name}-{quant.lower()}.gguf"
            run([str(quantizer), str(gguf_path), str(final_gguf), quant], args.dry_run)
            if final_gguf.exists():
                log.info(f"Quantized GGUF: {final_gguf} ({final_gguf.stat().st_size / 2**30:.2f} GiB)")

    # 4. Modelfile (+ optional ollama create)
    modelfile = write_modelfile(final_gguf, model_dir, tag, mmproj_path, out_dir)
    if args.create:
        run(["ollama", "create", tag, "-f", str(modelfile)], args.dry_run)
        log.info(f"Run it with: ollama run {tag}")
    else:
        log.info(f"Import manually:  ollama create {tag} -f \"{modelfile}\"")

    log.info(f"export_gguf.py finished in {fmt_duration(time.time() - start)}")


if __name__ == "__main__":
    main()
