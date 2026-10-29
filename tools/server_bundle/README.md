# Consilia annotation bundle (GPU server)

Self-contained: scripts, schemes, the units and consilia, the gold sets. Nothing else from the repo is
needed. Built by `tools/make_server_bundle.sh` from the repo; rebuild after data or prompt changes.

```bash
tar xzf consilia-server-bundle.tar.gz && cd consilia-server-bundle
./run.sh                      # setup → serve → gold → full → spans → flair → body2 → pack
./run.sh gold                 # or single stages, in any order, resumable
```

GPU queue (gpq): put the bundle tarball into `~/consilia_new/`, copy `run_consilia.sh` to
`~/.gpq/session/run_consilia.sh` and register it. It unpacks once, runs setup once, then the stages
full, spans, flair, body2 (gold was run already; `STAGES="gold full …"` repeats it) with a marker per finished stage in `state/`, packs the results on every
exit (also when the session is cut off) and stops vLLM. After an interruption register the same script
again: finished stages are skipped, classify/spans continue with the unfinished items, the Flair training resumes from its last epoch checkpoint (model_epoch_<k>.pt). Set
`GPQ_END_CMD` in the script if the queue needs a command to release the session; `DIR` and `STAGES`
can be overridden the same way. Logs in `~/consilia_new/logs/gpq_*.log`.

| Stage | What | Rough time on one RTX 6000 (96 GB) |
|---|---|---|
| setup | `.venv` (Flair + requirements) and `.venv-vllm` (vLLM from PyPI, several GB); or point `VLLM_BIN` at an existing vLLM | 10–20 min |
| serve | starts `vllm serve Qwen/Qwen3.8-27B-FP8` (bf16 KV cache, prefix caching) if `BASE_URL` does not answer; first start downloads ~30 GB | 5 min + download |
| gold | Layer A/B on the gold splits + evaluation; compare with the Gemini numbers printed | ~15 min |
| full | Layer B pass 1 (10,352 units), Layer A (2,730 consilia) | 4–6 h |
| body2 | Layer B pass 2, a second reading; optional, last | 3–4 h |
| spans | Layer C (citations/entities) over all units, silver corpus | 4–6 h |
| flair | export BIO corpus, fine-tune MediBerta stacked with the Latin legal Flair LMs, tag all units | 1–2 h |
| pack | `consilia-results.tar.gz` (annotations, eval, logs) and `consilia-flair-model.tar.gz` | — |

Failed items (`ok: false`, mostly answers truncated at the token budget) are retried simply by running the stage again.
Settings: `MODEL` (default `Qwen/Qwen3.8-27B-FP8`; `Qwen/Qwen3.6-27B` works the same way), `BASE_URL`, `WORKERS` (32),
`THINKING` (`--thinking` or `--no-thinking`; Qwen3.8 thinks by default), `EPOCHS`, `BATCH`. Output files carry the model
slug, e.g. `body_qwen-qwen3-8-27b-fp8_p1.jsonl`.
vLLM runs with Triton attention and the torch sampler because FlashInfer's JIT kernel builds failed on the Blackwell box; `ATTN_BACKEND=FLASHINFER FLASHINFER_SAMPLER=1` tries FlashInfer again. If start-up still dies, `ENFORCE_EAGER=1 ./run.sh serve` skips graph capture (roughly 20–30 % slower). `EXTRA_VLLM_ARGS` passes further flags; on a crash run.sh prints the traceback.
If gold shows no gain from thinking, run the rest with `THINKING=--no-thinking ./run.sh full spans`.

Back in the repo: unpack `consilia-results.tar.gz` in the repo root (it only adds files under
`output/annotations`, `eval/results`, `output/flair`, `logs`), then

```bash
python src/annotate/export.py --body output/annotations/body_qwen-qwen3-8-27b-fp8_p1.jsonl output/annotations/body_qwen-qwen3-8-27b-fp8_p2.jsonl --subjects output/annotations/subjects_qwen-qwen3-8-27b-fp8_p1.jsonl
python src/build_site_data.py && python src/qa/check_data.py
```

The span gold (`eval/gold/spans_gold.jsonl`) is optional for the run; if it exists when the bundle is
built or is copied in later, the spans and flair stages evaluate against it and the tagger trains on
gold + silver.
