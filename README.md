# Syco-Eval — LLM Sycophancy Evaluation

## Using the deployed app

1. Visit the **Netlify deployment** above.
2. Select **Qwen2.5 0.5B Instruct** or **Qwen2.5 1.5B Instruct**.
3. Choose **10 pairs**, **20 pairs**, or **100 pairs** and set temperature, compute backend, and seed.
4. Click **Load Qwen in browser**. Initial loading downloads public model and JavaScript assets.
5. Click **Run evaluation**. Every pair receives one neutral prompt and one prompt containing a confidently stated wrong answer.
6. Review the two model responses per pair; adjudicate ambiguous answers, then select **Export raw CSV** and **Export results report**.

## Research question

When a user confidently asserts a **false** answer, how often does Qwen2.5-Instruct abandon a correct factual response and agree with that assertion?

This project concerns **AI alignment, behavioral safety evaluation, truthfulness, and sycophancy**. It does not study neural-network internal mechanisms.

## What the project measures

- Accuracy under neutral questioning.
- Agreement with deliberately incorrect user claims.
- Conditional switches from a correct neutral answer to an incorrect pressured answer.
- Classifier coverage and uncertainty estimates for eligible observations.

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

The application executes Qwen inference within the browser through Transformers.js and ONNX Runtime Web (WebGPU or WASM/CPU). **No hosted inference API or API key is required.** Initial public model downloads need internet access; speed and feasibility depend on device resources and browser compatibility.

The dataset contains 100 hand-built pairs across arithmetic, geography, science, history, and computing. Short evaluation sizes sample across these domains. Heuristic labels are intentionally conservative and ambiguous answers need human adjudication. Report metrics from completed model runs only; this narrow pressure-template study does not prove general sycophancy behavior. The archive includes a Python evaluation implementation and tests for technical review, while Netlify is the recommended user-facing entry point.
