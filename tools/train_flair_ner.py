#!/usr/bin/env python3
"""Train a Flair sequence tagger for citations and entities on output/flair/ (from
src/annotate/export_flair.py). GPU server.

Embeddings (default `--mode stack`): MediBerta (mschonhardt/MediBerta, XLM-R tuned on medieval Latin,
fine-tuned here) stacked with the two Latin legal Flair language models (mschonhardt/latin-legal-forward
and -backward, contextual string embeddings, frozen). Hugging Face ids are downloaded automatically;
local paths work too.

  pip install flair
  python tools/train_flair_ner.py                      # stack (default)
  python tools/train_flair_ner.py --mode transformer   # MediBerta only
  python tools/train_flair_ner.py --mode flair         # Flair LMs + BiLSTM-CRF only
  python tools/train_flair_ner.py --predict            # tag all units with the trained model

Writes output/flair/model/ (final-model.pt, training.log, test results with span-level P/R/F1 per label).
A checkpoint model_epoch_<k>.pt is saved after every epoch; a rerun after an interruption resumes from the latest one.
`--predict` writes output/annotations/spans_flair.jsonl in the record shape of spans.py
({unit, doc, spans: [{start, end, text, label, score}]}) so export and index code can use it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "flair"
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
DEFAULT_TRANSFORMER = "mschonhardt/MediBerta"
DEFAULT_FWD = "mschonhardt/latin-legal-forward/latin-legal-forward.pt"
DEFAULT_BWD = "mschonhardt/latin-legal-backward/latin-legal-backward.pt"


def resolve(spec: str) -> str:
    """Local path, or 'owner/repo/filename' on the Hugging Face Hub (downloaded to the HF cache)."""
    if Path(spec).exists():
        return spec
    parts = spec.split("/")
    if len(parts) >= 3:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id="/".join(parts[:2]), filename="/".join(parts[2:]))
    return spec


def build_embeddings(a):
    from flair.embeddings import FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
    parts = []
    if a.mode in ("stack", "transformer"):
        parts.append(TransformerWordEmbeddings(a.transformer, layers="-1", subtoken_pooling="first",
                                               fine_tune=True, use_context=True, allow_long_sentences=True))
    if a.mode in ("stack", "flair"):
        parts += [FlairEmbeddings(resolve(a.lm_forward), fine_tune=False), FlairEmbeddings(resolve(a.lm_backward), fine_tune=False)]
    return parts[0] if len(parts) == 1 else StackedEmbeddings(parts)


def train(a) -> None:
    import flair
    import torch
    from flair.datasets import ColumnCorpus
    from flair.models import SequenceTagger
    from flair.trainers import ModelTrainer

    flair.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # TF32 on Ampere and later
    corpus = ColumnCorpus(DATA, {0: "text", 1: "ner"}, train_file="train.txt", dev_file="dev.txt", test_file="test.txt")
    print(corpus)
    labels = corpus.make_label_dictionary(label_type="ner")
    # checkpoints: Flair saves model_epoch_<k>.pt after every epoch (save_model_each_k_epochs=1); if the run was
    # cut off, the latest one is loaded and training continues for the remaining epochs (the learning-rate
    # schedule restarts for that remainder, which is acceptable for a fine-tuning run of this size)
    done_epochs, tagger = 0, None
    ckpts = sorted(a.out.glob("model_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if ckpts and not (a.out / "final-model.pt").exists():
        done_epochs = int(ckpts[-1].stem.split("_")[-1])
        print(f"resuming from {ckpts[-1].name}: {done_epochs} of {a.epochs} epochs done")
        tagger = SequenceTagger.load(ckpts[-1])
    remaining = a.epochs - done_epochs
    if remaining <= 0:
        print("all epochs done; keeping the last checkpoint as final-model.pt")
        import shutil
        shutil.copy(ckpts[-1], a.out / "final-model.pt")
        return
    ckpt_kwargs = {"save_model_each_k_epochs": 1, "save_optimizer_state": True}
    if a.mode == "flair":
        if tagger is None:
            emb = build_embeddings(a)
            tagger = SequenceTagger(hidden_size=256, embeddings=emb, tag_dictionary=labels, tag_type="ner", use_crf=True, use_rnn=True)
        ModelTrainer(tagger, corpus).train(a.out, learning_rate=0.1, mini_batch_size=a.batch, max_epochs=remaining,
                                           embeddings_storage_mode="gpu", patience=3, **ckpt_kwargs)
    else:
        # fine-tuning recipe: transformer trained end to end, Flair LMs frozen on top, CRF for span consistency
        if tagger is None:
            emb = build_embeddings(a)
            tagger = SequenceTagger(hidden_size=256, embeddings=emb, tag_dictionary=labels, tag_type="ner",
                                    use_crf=True, use_rnn=False, reproject_embeddings=False)
        # chunk 0 = the whole mini-batch in one pass (large GPUs); --amp: mixed precision (autocast + grad scaler)
        ModelTrainer(tagger, corpus).fine_tune(a.out, learning_rate=a.lr, mini_batch_size=a.batch,
                                               mini_batch_chunk_size=a.chunk or None, max_epochs=remaining, use_final_model_for_eval=False,
                                               use_amp=a.amp, **ckpt_kwargs)


def predict(a) -> None:
    import flair
    import torch
    from flair.data import Sentence
    from flair.models import SequenceTagger

    flair.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = a.out / ("best-model.pt" if (a.out / "best-model.pt").exists() else "final-model.pt")
    tagger = SequenceTagger.load(model_file)
    out = a.predict_out or ROOT / "output" / "annotations" / "spans_flair.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    units = [json.loads(l) for l in (a.units or ROOT / "output" / "units.jsonl").open(encoding="utf-8")]
    group = 200                                     # units per predict call: large GPU batches, sentences sorted by length inside
    with out.open("w", encoding="utf-8") as fh:
        for g0 in range(0, len(units), group):
            work = []
            for u in units[g0:g0 + group]:
                text = u["la"]
                toks = [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(text)]
                chunks = [toks[i:i + 120] for i in range(0, len(toks), 120)]
                work.append((u, text, chunks, [Sentence([t for t, _, _ in ch]) for ch in chunks]))
            sents = [s for *_, ss in work for s in ss]
            while sents:
                try:
                    with torch.autocast(device_type=flair.device.type, enabled=a.amp and flair.device.type == "cuda"):
                        tagger.predict(sents, mini_batch_size=a.predict_batch)
                    break
                except torch.OutOfMemoryError:
                    if a.predict_batch <= 1:
                        raise
                    a.predict_batch //= 2; torch.cuda.empty_cache()
                    print(f"  out of GPU memory: batch now {a.predict_batch}", flush=True)
            for u, text, chunks, ss in work:
                spans = []
                for ch, s in zip(chunks, ss):
                    for sp in s.get_spans("ner"):
                        i0, i1 = sp.tokens[0].idx - 1, sp.tokens[-1].idx - 1
                        a0, b0 = ch[i0][1], ch[i1][2]
                        spans.append({"start": a0, "end": b0, "text": text[a0:b0], "label": sp.tag, "score": round(float(sp.score), 3)})
                fh.write(json.dumps({"unit": u["unit"], "doc": u["doc"], "task": "spans", "model": f"flair:{model_file.name}",
                                     "spans": spans, "ok": True}, ensure_ascii=False) + "\n")
            print(f"  {min(g0 + group, len(units))}/{len(units)}", flush=True)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["stack", "transformer", "flair"], default="stack")
    ap.add_argument("--transformer", default=DEFAULT_TRANSFORMER)
    ap.add_argument("--lm-forward", default=DEFAULT_FWD)
    ap.add_argument("--lm-backward", default=DEFAULT_BWD)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4, help="mini-batch chunk size for fine-tuning (memory); 0 = no chunking")
    ap.add_argument("--amp", action="store_true", help="mixed precision (torch autocast) for training and prediction")
    ap.add_argument("--predict-batch", type=int, default=32, help="sentences per GPU batch in --predict")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", type=Path, default=DATA / "model")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--units", type=Path, help="units file for --predict (default output/units.jsonl)")
    ap.add_argument("--predict-out", type=Path, help="output for --predict (default output/annotations/spans_flair.jsonl)")
    a = ap.parse_args()
    predict(a) if a.predict else train(a)


if __name__ == "__main__":
    main()
