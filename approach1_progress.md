# Pipeline Update Summary

This update modernizes the clinical report LLM evaluation pipeline while preserving the original prediction, validation, output, and evaluation structure as much as possible.

## Major changes

### 1. Updated API usage

- Replaced the outdated `AzureOpenAI` client setup with the newer `requests`-based SHC Azure OpenAI API pattern.
- Updated the deployment to `gpt-5-nano` for GPT-5-nano Global usage.
- Switched authentication to use the `api-key` header rather than `Ocp-Apim-Subscription-Key`.
- Loads the API key from `SANDBOX_API_KEY`, matching the newer working example.

### 2. Added post-run token and cost tracking

The script now reads token usage from each API response and tracks cumulative usage across the full run, including:

- Total prompt/input tokens
- Cached input tokens
- Uncached input tokens
- Completion/output tokens
- Reasoning tokens, if reported by the API
- Total tokens
- Estimated total cost
- Estimated cost savings from cached tokens

Pricing assumptions are based on GPT-5-nano Global:

| Token type   | Price per 1M tokens |
| ------------ | ------------------: |
| Input        |               $0.05 |
| Cached input |               $0.01 |
| Output       |               $0.40 |

A cumulative cost report is printed and saved as `token_cost_report.json`.

### 3. Added a-priori full-pipeline cost estimation

Before running inference across all held-out cases, the script now:

1. Builds the full prompt for every test case.
2. Estimates total input-token usage before sending API calls.
3. Estimates possible output-token cost using the configured maximum completion-token budget.
4. Prints a full projected cost estimate for the pipeline.
5. Prompts the user from the command line to confirm before continuing.

A `--yes` / `-y` command-line flag was added to bypass this confirmation for non-interactive runs.

### 4. Simplified validation-token system

- Removed the previous two-token validation setup scattered throughout the prompt.
- Replaced it with a single deterministic validation token per case.
- The validation token is placed only once, at the very end of the report/prompt.
- Output validation now checks that the model returns this single expected token exactly.

### 5. Prompt structure adjusted for caching

The prompt was reorganized to place stable, repeated content earlier, including:

- Task instructions
- Strict rules
- Descriptor guide
- Output JSON schema
- Few-shot training examples

Case-specific content is placed later. This structure is intended to improve prompt-prefix caching across repeated calls and increase cached-token savings.

### 6. Preserved core pipeline behavior

The following core behavior remains unchanged:

- Same CSV-based case loading structure
- Same few-shot training-row setup
- Same held-out test-case loop
- Same JSON-only model output requirement
- Same schema/range validation logic, except for the simplified validation token
- Same prediction outputs to JSONL and CSV
- Same downstream evaluation metrics for dispersion and relapse prediction

## Output files added or modified

- `predictions_testing_cases.jsonl`: per-case prediction records
- `predictions_testing_cases.csv`: tabular prediction results
- `evaluation_metrics.txt`: downstream evaluation metrics
- `run.log`: full pipeline log
- `token_cost_report.json`: cumulative token and cost report

## Practical effect

The updated script should now run against the newer SHC Azure GPT-5-nano API endpoint, estimate cost before execution, track actual token/cost usage after each call, report cached-token savings, and use a cleaner validation-token design that avoids distributing sentinel tokens throughout the clinical prompt.
