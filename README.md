# Syco-Eval — Local-Model Sycophancy Evaluation

## Launch the app

**Netlify deployment:** [Launch Syco-Eval](https://YOUR-SYCO-EVAL-SITE.netlify.app/)

**Deployment URL note:** Replace `YOUR-SYCO-EVAL-SITE` with the actual name of your Netlify deployment before publishing this README. The deployment URL was not provided.

Syco-Eval measures how often Qwen2.5-Instruct follows a user's confidently stated *incorrect* answer instead of maintaining a correct factual response.

## Using the deployed app

1. Visit the **Netlify deployment** above.
2. Select **Qwen2.5 0.5B Instruct** or **Qwen2.5 1.5B Instruct**.
3. Choose **10 pairs**, **20 pairs**, or **100 pairs** and set temperature, compute backend, and seed.
4. Click **Load Qwen in browser**. Initial loading downloads public model and JavaScript assets.
5. Click **Run evaluation**. Every pair receives one neutral prompt and one prompt containing a confidently stated wrong answer.
6. Review the two model responses per pair; adjudicate ambiguous answers, then select **Export raw CSV** and **Export results report**.

## What the project measures

- Accuracy under neutral questioning.
- Agreement with deliberately incorrect user claims.
- Conditional switches from a correct neutral answer to an incorrect pressured answer.
- Classifier coverage and uncertainty estimates for eligible observations.

## Technical details and limitations

The application executes Qwen inference within the browser through Transformers.js and ONNX Runtime Web (WebGPU or WASM/CPU). **No hosted inference API or API key is required.** Initial public model downloads need internet access; speed and feasibility depend on device resources and browser compatibility.

The dataset contains 100 hand-built pairs across arithmetic, geography, science, history, and computing. Short evaluation sizes sample across these domains. Heuristic labels are intentionally conservative and ambiguous answers need human adjudication. Report metrics from completed model runs only; this narrow pressure-template study does not prove general sycophancy behavior. The archive includes a Python evaluation implementation and tests for technical review, while Netlify is the recommended user-facing entry point.

Detailed methodology: `flattery_eval/README.md`.
