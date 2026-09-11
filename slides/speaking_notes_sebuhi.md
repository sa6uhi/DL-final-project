# Speaking notes — Səbuhi (Member C): DAE + serving (~5 minutes)

Total: 5 slides, ~60 seconds each. Numbers below are the exact values on the slides —
do not round differently on stage.

---

## 1. "The DAE learns what legitimate traffic looks like" (~60s)

> "Fraud is 3.5% of traffic and it keeps changing shape, so a classifier trained on
> yesterday's fraud misses tomorrow's. My second detector takes the opposite approach:
> it never sees fraud at all. The denoising autoencoder trains only on legitimate
> transactions, with Gaussian noise and 10% feature dropout so it learns structure
> instead of copying input to output. At serving time, whatever it fails to
> reconstruct is suspicious — the anomaly score is MSE plus a small L1 term, gamma
> 0.4, so a few large residuals can't dominate. The model is tiny: 485 thousand
> parameters, under 2 megabytes."

Trap: if asked why 800 inputs — 400 scaled numerics plus 400 missingness flags.
If asked why only legit — otherwise it learns to reconstruct fraud too and the
residual stops discriminating.

Transition: "An anomaly score is only useful if it tells us something the
classifier doesn't — next slide."

## 2. "The DAE signal is complementary and explainable" (~60s)

> "Two pieces of evidence. Left: t-SNE of the 32-dimensional bottleneck — fraud and
> legit overlap, silhouette 0.105. That is the point: anomaly is not fraud, the DAE
> fires on different cases than the transformer, which is exactly what you want from
> a second opinion. Right: we run SHAP on the anomaly score, so every flag comes
> with drivers — address and amount dominate, and the ranking is stable across
> samples, Spearman 0.898. One honest caveat, printed on the slide: SHAP explains
> the DAE component only, not the final gated decision."

Transition: "So much for the model — now how it reaches production."

## 3. "From checkpoint to endpoint: the serving stack" (~60s)

> "Checkpoints are exported once, in two runtimes: EXIR, which stays inside PyTorch,
> and ONNX, which fuses kernels and drops the Python overhead — that's why the same
> scorer runs 0.99 milliseconds under EXIR and 0.14 under ONNX. FastAPI serves them
> statelessly: /predict for single transactions, /stream for JSON batches,
> /explain for the SHAP waterfall, plus /health and /metrics. The Streamlit
> dashboard that you'll see in the demo reads those same endpoints — there is no
> separate demo path, what the graders can curl is what the analyst sees."

Trap: /stream is a JSON batch endpoint, not SSE streaming — say "batch" if asked.
Trap: /metrics percentiles are scoring-only; the two-second SHAP explain path is
excluded by design.

Transition: "All of this rebuilds from zero with one command — next."

## 4. "One command reproduces everything" (~45s)

> "./run_all.sh from a clean checkout: it downloads the data, prepares it — that
> part always re-runs because it's cheap — trains only if checkpoints are missing,
> then evaluates, serializes, and runs the test gate. Three Docker images mirror
> the pipeline stages — init for data, train for training, infer for serving —
> published to GHCR, and CI enforces black, flake8, and 787 tests with an 80%
> coverage gate. Fresh-clone reproduction verified end to end."

Keep it short — this slide exists to pre-empt the "does it actually run?" question.

Transition: "And it runs fast — the numbers."

## 5. "Fast enough to serve on a single CPU" (~60s)

> "Full fusion pipeline: 6.56 milliseconds median, 2,746 transactions per second at
> batch 128, on a single CPU container host. The headline P99 of 26.95 milliseconds
> misses 15 — and the slide says so, because FT-CAT dominates the budget at 5 of
> those 6.5 milliseconds. The 15-millisecond objective holds where it matters: the
> DAE scorer path that gates live traffic runs 0.71 milliseconds P99, 0.14 under
> ONNX. All timings are model-side, no HTTP, no SHAP — measured over 50 timed runs
> after warm-up."

This is the slide the jury will probe. Memorize this one-liner:

> "15 milliseconds holds for the DAE scorer path that gates live traffic;
> the full fusion pipeline is batch and async."

---

## Q&A one-liners (30 seconds each, details in MEMBER_C_DEFENSE_GUIDE.md)

- **Why MSE, not BCE?** RobustScaler leaves features unbounded; clamping to [0,1]
  destroys fraud tails. MSE matches the data geometry.
- **EXIR vs ONNX?** EXIR is PyTorch-native AOT; ONNX fuses kernels and skips
  Python — hence 0.14 ms.
- **Block rate 0%?** At alpha 0.01 the threshold is conservative: auto-approve the
  confident 77%, review the rest. Blocking needs a looser operating point.
- **Demo fraud only human_review?** Honest output at PR-AUC 0.34 — our story is
  triage with SHAP drivers, not magic.
