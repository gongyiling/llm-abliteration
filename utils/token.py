import os


def get_hf_token() -> str | None:
    """Read HF token from env HF_TOKEN, else from a local hf_token.txt (one line, not committed).

    Search order: HF_TOKEN env var, ./hf_token.txt in the current working directory,
    then hf_token.txt next to this package and in its parent (project root).
    """
    t = os.environ.get("HF_TOKEN")
    if t:
        return t

    here = os.path.dirname(os.path.abspath(__file__))
    for fn in (
        "hf_token.txt",
        os.path.join(here, "hf_token.txt"),
        os.path.join(os.path.dirname(here), "hf_token.txt"),
    ):
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None
