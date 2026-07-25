Coming soon: **ZADos Lite**

A subsystem for reaching into how the LLM generates, without modifying weights or retraining anything. Six points of contact: prompt construction, logit biasing, sampling, KV-cache read/write, logprob capture, soft-prompt injection.

Still in architectural design. A few pieces are already resolved:

Gappy tokens. Uncertain generations get their Key masked from attention while the Query keeps running, evolving as a CIR process under Milstein integration until it crosses a resolution threshold.
Surprisal Preprocessor. Every token's logprob passes through one typed frame (raw_surprisal, rolling_mean_N, running_variance, a CSS-anchored classification) before any downstream consumer reads it.
SDE-timestamp-ordered cache writes. Cache mutations are ordered by the timestamp of the process that caused them, resolved to sub-step precision.
CIR-governed sampling. Temperature and repetition penalty run as stochastic processes driven by derived neurochemical metrics, not fixed values set once per turn.

No code yet. Pass-one architectural walkthrough is closed. Pass-two, turning this into buildable specs, is next.

Status: pass-two engineering enumeration in progress.
