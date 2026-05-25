# Project 1 Failure Analysis

A structured record of failures encountered during SFT and DPO training. Each entry covers what happened, why, what was tried, and the outcome.

---

## Summary Table

| Failure | What Happened | Hypothesis | Tried | Result | Status |
|---|---|---|---|---|---|
| Prompt leakage | Output continued with role/prompt labels after answer | Prompt boundary not stable in legacy template | Chat template + stop sequences + raw/final leakage eval | leakage = 0 | solved |
| Extra notes | Email body followed by explanation or tips | SFT data format contamination + base model habit | no_extra_notes eval + DPO email_quality pairs | Largely fixed; 7B DPO still oscillates | monitored |
| Too short | Non-email answers were one sentence | Insufficient step data + adapter insertion too narrow | has_steps eval + step-guidance data + attn+MLP target modules | Resolved after attn+MLP | solved |
| Missing international office | CPT/OPT/visa queries gave general guidance without mentioning ISEO | 1.5B capacity bottleneck | Multi-round DPO visa_safe pairs + 7B scale-up | 7B largely resolved | solved at 7B |
| steps+email incomplete | Prompt asked for steps + email; model gave only one | Format preference not stably learned | DPO steps_email pairs (chosen = steps+email, rejected = one only) | Improved | monitored |
| DPO v1 no gain | SFT-only 90% = DPO v1 90% preference win rate | Eval saturation + pair quality insufficient | Repaired pairs + non-saturated eval set + expanded data | v5 achieved meaningful improvement | improved |
| dpo_visa_safe_010 | Model scored rejected higher than chosen | Rejected near-miss too similar to chosen in surface form | Revised pair: chosen more assertive, rejected to prose | Fixed in v2 | solved |
| 7B overconfidence | no_absolute_promise failures increased after DPO | 7B absorbs DPO signal more strongly, amplifying confidence | 6 targeted pairs (v2), then reverted | Patch caused no_extra_notes to spike 1->7; reverted | open |
| DPO non-determinism | Same config, different rule pass across runs (301-309/311) | Small-data 1-epoch LoRA DPO sensitive to GPU floating-point variation | Multiple runs recorded; promoted best checkpoint | Accepted as known limitation | documented |

---

## 1. Prompt Leakage / Generation Boundary

**What happened:**

Early SFT runs produced outputs that continued generating after the answer ended. Common patterns included the model outputting "Human:", "Assistant:", "### Instruction:", "### Response:", "Note:", or "Explanation:" after the actual response body.

**Hypothesis:**

The legacy prompt template (### Instruction / ### Input / ### Response) left the model uncertain about where its output should stop. The base model's pretraining distribution included multi-turn conversation formats and instruction-following templates that the model partially reproduced.

**What was tried:**

- Switched from legacy prompt template to the model's native chat template (apply_chat_template)
- Added explicit stop sequences to truncate output at prompt/role boundary markers
- Added prompt_utils.clean_generation_with_info() to strip leakage in postprocessing
- Added two-level leakage tracking in eval: raw_prompt_leakage (before postprocessing) and postprocessed_prompt_leakage (after), so the eval could distinguish hard failures from soft ones

**Result:**

Raw prompt leakage = 0, final prompt leakage = 0, truncation = 0 across all model versions from v1 onward.

**Status: solved**

Early SFT runs showed prompt-boundary leakage. Fixed by standardizing the chat template, adding stop sequences, and explicitly evaluating both raw and final responses for leakage.

---

## 2. Extra Notes After Email Body

**What happened:**

Email drafts were correct and complete, but the model appended a commentary section after the closing:

- "Note: This email is polite because it acknowledges the professor's time."
- "Explanation: The subject line is specific so the professor knows the topic."
- "Please replace [Your Name] with your actual name before sending."

These additions are not acceptable in a campus assistant that produces ready-to-send emails. Users expect the output to end at "Best regards, [Your Name]".

**Hypothesis:**

Two sources: (1) the SFT training data may have included output examples with post-email notes that were not cleaned out before training; (2) the base model's instruction-following distribution includes self-commentary after completing a writing task, especially for formal email templates.

**What was tried:**

- SFT data: cleaned all output examples to remove post-closing commentary
- Eval: added no_extra_notes check — fails if output contains "Note:", "Explanation:", "This email", "Why this works", "Please replace", or similar markers after the email body
- DPO email_quality pairs: chosen = email ending cleanly at "Best regards, [Your Name]"; rejected = same email with post-closing note

**Result:**

Largely resolved in 1.5B models. However, 7B DPO runs show oscillation in no_extra_notes: v1 had 1 failure, v2 had 7 (caused by patch pairs with hedging-heavy chosen answers), v3 had 5 after revert. This is a monitored known limitation.

**Status: monitored**

Solved in SFT stage. DPO training can reintroduce it if chosen answers for other failure types contain verbose or hedging-heavy phrasing that the model over-generalizes.

---

## 3. Response Too Short / Missing Steps

**What happened:**

Some non-email responses were a single sentence: "Contact the housing office." or "Check with your advisor." These responses are directionally correct but not useful. The campus assistant goal requires specific steps, what to prepare, which office to contact, and optionally an email template.

**Hypothesis:**

Two contributing factors: (1) the SFT training data had insufficient step-by-step guidance examples for certain scenario types; (2) the LoRA adapter was inserted only into attention projections (qv-only), limiting its capacity to learn structured output formats.

**What was tried:**

- Eval: added has_steps check — requires numbered steps (1. / 2. / 3.) for process guidance
- Eval: added not_too_short check
- SFT data: supplemented with step-structured guidance samples for housing, enrollment, and insurance categories
- LoRA: expanded target modules from qv-only to full attn+MLP (q/k/v/o + gate/up/down_proj) — this was the decisive fix

**Result:**

not_too_short failures essentially disappeared after attn+MLP expansion. The target_modules ablation confirmed that attn+MLP consistently outperformed attn-only and qv-only on step structure metrics.

**Status: solved**

The root cause was partially model capacity (adapter insertion scope), not only data. Expanding to attn+MLP resolved the issue.

---

## 4. Missing International Office Mention

**What happened:**

High-risk queries involving CPT, OPT, F-1 status, SEVIS, or below-full-time enrollment sometimes received factually reasonable guidance about the enrollment or visa process, but did not mention the international student office (ISEO). For these queries, directing the student to ISEO is a safety requirement — not a suggestion — because errors can result in status violations.

**Hypothesis:**

The 1.5B base model's distribution treats enrollment and visa questions as general administrative problems and does not reliably associate them with international office escalation. The model has not learned the conditional rule: "if enrollment change + international student status risk → must mention ISEO."

**What was tried:**

- SFT data: added high-risk international student scenarios with explicit ISEO mentions
- DPO v1–v5: over six iterations and 6+ targeted visa_safe pairs covering CPT authorization, full-time enrollment risk, and SEVIS email scenarios
- The failure oscillated between 1–2 different eval samples across all versions, but never reached zero on 1.5B
- Scaled up to Qwen2.5-7B

**Result:**

1.5B never achieved zero failures on this check despite 6+ targeted pairs. 7B DPO v1 had 0 failures; v3 (promoted) had 1, attributed to CUDA non-determinism. The capacity ceiling at 1.5B is confirmed: this behavior pattern requires more model parameters to reliably generalize.

**Status: solved at 7B**

Repeated targeted DPO on 1.5B could not fully eliminate international-office escalation failures. Scaling to 7B largely resolved the issue, confirming this was a capacity bottleneck rather than a data coverage problem.

---

## 5. steps+email Format Incomplete

**What happened:**

Some prompts explicitly required both a numbered step-by-step plan AND an email template (output_format = steps_plus_email). The model sometimes produced only the email, or only the steps, rather than both in sequence. A user asking how to handle a housing maintenance issue should receive: (1) what to do first, (2) what to prepare, (3) an email they can send to the housing office.

**Hypothesis:**

This is a format preference problem, not a factual error. The base model's distribution does not strongly favor the steps-then-email structure over single-mode output. Without preference signal, the model defaults to whichever format it finds more likely under its learned distribution.

**What was tried:**

- DPO steps_email pairs: chosen = complete steps + email; rejected = only steps, only email, or email with missing closing
- The steps_plus_email output format was tracked as a separate category in preference eval

**Result:**

Improved across DPO versions. Not fully eliminated — the model still occasionally produces only one component, particularly for complex multi-step scenarios where the email portion is long. Tracked as an ongoing quality concern.

**Status: monitored**

---

## 6. DPO v1 No Measurable Gain

**What happened:**

DPO v1 trained successfully (loss decreased, reward margins positive during training), but the preference eval showed no improvement over SFT-only: both scored 90% win rate on the same eval set.

**Hypothesis:**

Two causes: (1) the eval set was too small (10 pairs) and likely saturated — SFT-only already handled most pairs correctly, leaving no room for DPO to demonstrate improvement; (2) at least one pair (dpo_visa_safe_010) had a preference inversion where the model scored rejected higher than chosen, actively undermining the win rate.

**What was tried:**

- Identified and repaired dpo_visa_safe_010 (see failure 7 below)
- Added a small number of targeted pairs for each rule failure observed in v1 eval
- Rebuilt eval set with near-miss rejected answers in v3, making it non-saturated (SFT-only 83.33%)
- Expanded DPO data to 151 pairs in v5

**Result:**

By v5 (151 pairs, 30-pair non-saturated eval): SFT-only 83.33% vs DPO v5 86.67% — the first version with a clear, directly comparable preference improvement. Score margin also widened consistently across versions.

**Status: improved**

The lesson is that both eval set design and pair quality matter. A saturated eval set and a single inverted pair can mask real DPO training progress.

---

## 7. dpo_visa_safe_010 Preference Inversion

**What happened:**

In DPO v1 preference eval, the model assigned a higher log-probability to the rejected answer than to the chosen answer for this specific pair. The pair covers a SEVIS/immigration status email: chosen = contact ISEO immediately; rejected = wait for a clearer reminder.

**Hypothesis:**

The v1 rejected answer was a well-formatted numbered list, structurally identical to the chosen answer. The model's learned distribution could not distinguish them based on format alone, and the rejected answer's phrasing ("many status emails are informational") was more consistent with the base model's distribution for administrative questions. The content difference (immediate action vs. waiting) was not enough signal to overcome the surface similarity.

**What was tried:**

- Revised chosen to be more specific and assertive: added "do not wait for another reminder" and "ask whether there is a deadline, required action, or document you must submit"
- Converted rejected from a numbered list to a single prose paragraph, removing the structural advantage

**Result:**

Fixed in v2. The pair became a clear win: chosen score -2.0625, rejected score -3.5312.

**Status: solved**

A failed DPO pair revealed that preference data quality is critical. The rejected answer must not look structurally identical to the chosen answer, or the model cannot learn from the pair. The fix was to make the chosen answer more specific and downgrade the rejected answer's surface credibility.

---

## 8. 7B Overconfidence After DPO

**What happened:**

After DPO training on 7B, the no_absolute_promise check began failing. The model generated phrases like "your visa will not be affected" or "you do not need to contact anyone" in responses to visa and enrollment questions. This failure type did not appear in 1.5B DPO runs.

**Hypothesis:**

The 7B model absorbs DPO preference signal more strongly than 1.5B (evidenced by larger reward margins: ~1.36 vs ~0.61). This stronger absorption appears to amplify the model's confidence in its outputs across the board, including in contexts where overconfident claims are unsafe. The chosen answers in the DPO pairs were assertive and action-oriented; the model generalized this assertive style to high-risk domains where hedging is required.

**What was tried:**

- Added 6 targeted DPO pairs for no_absolute_promise (v2 patch): chosen answers explicitly hedge visa/status conclusions; rejected answers contain absolute guarantees
- Result: no_extra_notes spiked from 1 to 7 failures. The hedging and multi-caveat style in the new chosen answers was over-generalized to all outputs.
- Reverted to 151 pairs (v3)

**Result:**

v3 (promoted): no_absolute_promise = 1 failure (improved from v1's 4). Accepted as a known limitation. Direct data patching at this scale is unreliable because chosen answer style affects the whole model, not just the targeted behavior.

**Status: open**

The 7B overconfidence/no_absolute_promise failure is an alignment tradeoff: stronger DPO optimization produces better preference win rates and score margins, but can also amplify confident phrasing in safety-sensitive domains. Resolving this cleanly would require either multi-epoch DPO with stronger KL regularization, or a separate safety-focused RLHF pass.

---

## 9. DPO Training Non-Determinism

**What happened:**

Three training runs on 7B DPO with identical data (151 pairs), config (seed=42, beta=0.1, r=32), and hardware class (A100) produced different results:

| Run | Rule pass | no_extra_notes | no_absolute_promise | intl_office |
|---|---|---|---|---|
| v1 | 306/311 = 98.39% | 1 | 4 | 0 |
| v2 (patch, reverted) | 301/311 = 96.75% | 7 | 3 | 0 |
| v3 (promoted) | 304/311 = 97.75% | 5 | 1 | 1 |

Even v1 and v3, which used the same 151 pairs, produced different failure distributions.

**Hypothesis:**

GPU floating-point non-determinism causes micro-level batch order differences even with the same seed. At 1-epoch LoRA DPO on a small dataset (~121 train pairs), the model is sensitive to these differences. The rule pass rate oscillates in a ~301–309/311 range, which is characteristic of training near a behavioral capacity boundary where small gradient differences change which samples fall on which side of a check threshold.

**What was tried:**

- Ran three separate training runs and recorded all results
- Did not attempt to average checkpoints or ensemble (not compatible with the Colab deployment workflow)
- Promoted the checkpoint with the best combination of preference metrics and rule stability (v3: no_absolute_promise=1, intl_office=1, no_extra_notes=5)

**Result:**

Accepted as a known limitation. The promoted checkpoint is v3. Run-to-run variation of 1-3 check failures is expected and does not indicate a broken pipeline.

**Status: documented**

Small-data DPO is sensitive to micro-level training variation. Multiple runs were tracked and the checkpoint was selected based on both preference metrics and rule stability, not a single number. This is characteristic of 1-epoch LoRA DPO at this data scale and should be expected in future training iterations.

---

## Cross-Cutting Observations

**1. Eval set design is as important as data quality.**
A saturated eval set (where SFT-only already wins everything) makes DPO improvement invisible. Every eval iteration should target a non-trivial pass rate for SFT-only (80-90%), with near-miss rejected answers.

**2. DPO chosen answer style is learned globally, not locally.**
Adding pairs where chosen answers have hedging language, multiple caveats, or verbose structure causes that style to appear in unrelated outputs. New pairs must be written with the same concise, assertive style as the existing dataset.

**3. Targeted micro-patches are unreliable at small data scale.**
Adding 1-2 pairs for a specific failure often oscillates the failure to a different eval sample rather than eliminating it. Volume and variety of training signal matter more than targeted patches.

**4. Capacity vs. data coverage.**
Some failures (mentions_international_office) looked like data gaps but were actually capacity ceilings. Diagnosing this requires trying targeted pairs first, confirming they do not help, and then scaling the model. Do not over-invest in data patches before testing scale-up.

**5. DPO amplifies both good and bad patterns.**
Stronger DPO signal (larger reward margin, positive logits on chosen) improves win rate and score margin but can amplify whatever confident patterns the model has learned, including overconfident phrasing. Higher beta or multi-epoch DPO with stronger regularization would be needed to separate these effects.
