# Syco-Eval — Measuring Sycophancy with Qwen2.5-Instruct

## Research question

When a user confidently asserts a **false** answer, how often does Qwen2.5-Instruct abandon a correct factual response and agree with that assertion?

This project concerns **AI alignment, behavioral safety evaluation, truthfulness, and sycophancy**. It does not study neural-network internal mechanisms.

## Open the Netlify app

**Deployed application:** [Syco-Eval on Netlify](https://YOUR-SYCO-EVAL-SITE.netlify.app/)

> Replace `YOUR-SYCO-EVAL-SITE` with the actual deployment hostname before publishing; the real link has not been supplied.

On the deployed page, select a model (Qwen2.5 0.5B or 1.5B Instruct), size (10/20/100 pairs), temperature, compute backend, and sampling seed. Click **Load Qwen in browser**, followed by **Run evaluation**. Inspect both response variants for each question, apply human labels where needed, and export the raw CSV and results report.

## Paired-prompt evaluation design

The dataset has 100 factual questions divided evenly among Arithmetic, Geography, Science, History, and Computing. Each has a reference answer and an intentionally wrong answer. Every item is tested under two conditions with the same model and system instructions:

- **Neutral:** a direct question without an asserted answer.
- **Pressure:** the same question preceded by the user's confidently stated incorrect answer and a request for confirmation.

Each generation is evaluated independently. This design estimates susceptibility to one explicit type of social pressure. It does not identify a causal mechanism in the model, and order counterbalancing or additional pressure templates would strengthen the research design.

## Reported measures

- **Neutral accuracy:** fraction of unambiguously correct neutral responses.
- **Wrong agreement rate:** fraction of clearly classified pressured responses giving the planted wrong claim.
- **Conditional flip rate:** fraction of eligible pairs where a correct neutral answer changes to the planted wrong answer under pressure; baseline errors are not counted as flips.
- **Classifier coverage:** share of responses receiving unambiguous heuristic labels; ambiguous cases are flagged for review.
- **Wilson confidence interval:** a 95% interval for the eligible conditional-flip estimate. It does not establish broader population representativeness.

Human adjudication lets reviewers label individual neutral/pressured responses as correct, planted wrong, or other/ambiguous. Human labels are used in reported metrics and exported evidence. Do not claim statistical findings until actual model runs have been saved and reviewed.

## Inference and limitations

Generation happens in the browser using Transformers.js and ONNX Runtime Web, with WebGPU or WASM/CPU support. **No hosted inference API or API key is needed.** An internet connection is required for the initial public model/library downloads. Browser memory, network conditions, and inference hardware affect feasibility and speed.

The questions and pressure prompts are handcrafted, and the single explicit pressure template limits generalization. Quantized browser inference may not match results from other model backends. For credible reporting, document the model revision, settings, backend, seed, number of valid and reviewed cases, individual examples, and known weaknesses.

The archive also contains evaluation datasets, supporting Python tools, tests, and a report template for technical assessment. See `flattery_eval/REPORT_TEMPLATE.md` for the results outline.
