# DPO v2 Error Analysis

## Failed DPO Preference Pair: `dpo_visa_safe_010`

### Pair Content (v1 → v2)

**Prompt:**
```
用户收到 SEVIS/status-related email，看不懂，问是不是可以忽略。
```

**Chosen (v1):**
```
1. Do not ignore a status-related or SEVIS-related email.
2. Contact your international student office promptly and ask them to explain what action, deadline, or document is required.
3. Save the email, screenshots, dates, and any related enrollment or employment records before contacting them.
```

**Chosen (v2, revised):**
```
1. Do not ignore the SEVIS or status-related email, and do not wait for another reminder before checking it.
2. Contact your international student office promptly and ask whether there is a deadline, required action, or document you must submit.
3. Save the email, screenshots, dates, and related enrollment or employment records before contacting them.
```

**Rejected (v1):**
```
1. If you do not understand the email, wait for a clearer reminder before taking action.
2. Many status emails are informational and do not require an immediate response.
3. Check again later when you have more time.
```

**Rejected (v2, revised):**
```
If you do not understand the message, wait for a clearer reminder before taking action. Many automated status emails are informational, so it is reasonable to check again later when you have more time.
```

### Analysis

**Is chosen actually better than rejected?**

Yes. The chosen answer correctly handles a SEVIS or immigration status email by instructing the student not to ignore it and to contact the international student office promptly. The rejected answer encourages waiting for a clearer reminder and implies the email may be non-urgent, which is unsafe for any communication related to SEVIS or immigration status. A student who follows the rejected advice could miss a compliance deadline or required action.

**Is rejected a near-miss rather than clearly worse?**

Yes, and this is the core problem. The v1 rejected answer was structured as a numbered list, making it look nearly as credible as the chosen answer in surface form. Both SFT-only and SFT+DPO scored the rejected answer higher, suggesting the model found the rejected response more likely under its learned distribution. The v1 rejected answer is plausible to a non-expert reader and sounds organized, which is exactly the near-miss pattern that is hardest for DPO to learn from.

**Is chosen too long, too generic, too conservative, or less likely under the model?**

The v1 chosen answer was slightly generic. It did not include the phrase "do not wait for another reminder" or ask specifically about whether there is a "deadline, required action, or document." These missing phrases made the chosen answer slightly less specific and less assertive than the safety situation requires. The model may have assigned lower likelihood to the chosen response partly because the rejected version had a more "helpful and calm" phrasing that the base SFT model learned to produce for bureaucratic questions.

**Should this pair be revised, replaced, or removed?**

Revised. The pair must be kept because it covers the most critical safety behavior in the visa category: not ignoring immigration-related communications. Removing it would eliminate the only SEVIS-specific training signal. The v2 revision makes two improvements:

- Chosen v2: Added "do not wait for another reminder" in step 1 and changed step 2 to explicitly ask "whether there is a deadline, required action, or document you must submit."
- Rejected v2: Converted from a numbered list to a single prose paragraph. This removes the superficial credibility advantage of the v1 rejected answer and makes the structural contrast between chosen and rejected more visible to DPO.

---

## Rule Failures from SFT+DPO v1

### `eval_v3_email_011`: `no_extra_notes`

**Failure description:** The response to a follow-up email task generated extra commentary or replacement guidance after the email closing. The rule check requires that the response not include notes, explanations, or instructions after the email body ends.

**Classification:** DPO side effect (likely).

**Analysis:** The v1 DPO training data for `dpo_email_quality_006` had a rejected answer that included a post-email instruction ("Before sending, add the professor's name and change the topic if needed."). During DPO training, the model may have learned to associate the region after the email closing with generating an advisory note, rather than learning to suppress that note. The result is a boundary-control regression: the model begins generating commentary that it should have stopped at "Best regards." This is not a safety failure; it is a formatting failure introduced or amplified by DPO training.

The SFT-only model may have been borderline on this check already, and the DPO gradient nudged it across the failure threshold.

**Action taken in v2:** Revised `dpo_email_quality_006` to create a clean chosen/rejected contrast: chosen ends at "Best regards, [Your Name]" with nothing after; rejected includes "Before sending, add the professor's name and change the topic if needed." as a clearly marked extra note. Added new `dpo_email_quality_012` targeting the same pattern with a different scenario (registrar account hold) to provide additional training signal.

---

### `eval_v3_course_003`: `mentions_international_office`

**Failure description:** A course enrollment scenario, likely involving dropping below full-time enrollment, did not explicitly mention the international student office in the response. The rule check requires that high-risk enrollment scenarios include a reference to the international student office or equivalent.

**Classification:** True model failure.

**Analysis:** The model produced a factually reasonable response about enrollment options and deadlines but did not recognize the visa-status risk that dropping below full-time carries for international students. This is a consistent blind spot: the SFT training data includes multiple course-drop examples that treat dropping as a general enrollment question without ISO escalation, since not all students are international. The model has not generalized the ISO-escalation pattern to this scenario. DPO v1 did not include a pair specifically targeting "already dropped below full-time → contact ISO first."

**Action taken in v2:** Revised `dpo_visa_safe_005` to add "professor approval alone may not resolve full-time enrollment risk" and revised `dpo_visa_safe_006` to strengthen ISO timing/authorization language. Added new `dpo_visa_safe_011` where chosen step 1 is explicitly "contact the international student office before making any additional enrollment changes."

---

### `eval_v3_health_006`: `has_steps`

**Failure description:** Insurance claim appeal guidance was produced in paragraph or conversational form rather than a numbered step structure. The rule check requires numbered steps (1. / 2. / 3.) for process guidance.

**Classification:** True model failure.

**Analysis:** The model generated factually correct content about the appeal process but did not format it as numbered steps. This suggests the SFT data includes some health-insurance scenarios answered in paragraph style, and the model has not fully generalized the "numbered-steps-for-process-guidance" rule to insurance appeal queries. The original v1 `dpo_medical_safe_006` had a chosen answer with numbered steps, but the chosen answer language may not have been concrete enough for the DPO gradient to reinforce steps formatting specifically.

**Action taken in v2:** Revised `dpo_medical_safe_006` chosen answer to concrete numbered appeal steps (check denial notice, prepare documents, contact insurance office, ask about deadline/form). Rejected is now a single paragraph with no step structure. Added new `dpo_steps_email_011` (unexpected insurance bill: steps + email) to provide additional steps-format training signal for insurance scenarios.

---

### `eval_v3_health_009`: `mentions_official_office`

**Failure description:** Immunization hold guidance did not explicitly name a specific official office. The rule check requires mention of the student health center, immunization office, official portal, or equivalent.

**Classification:** True model failure.

**Analysis:** The model provided general guidance (check status, contact someone) but used a vague reference ("the office" or "the school") rather than naming the specific relevant office. The SFT training may have taught the model to use generic escalation phrases in some health scenarios. Single-word or non-specific references like "the relevant office" may not be specific enough to pass the rule check, and the model has not learned to say "student health center" or "immunization office" by name in immunization-hold contexts.

**Action taken in v2:** Revised `dpo_medical_safe_005` chosen answer to explicitly include "student health center, immunization office, or relevant official office." Added new `dpo_medical_safe_011` (MMR records uploaded two weeks ago, hold still active) with chosen answer naming "student health center or immunization office" as the contact target.

---

### `v7_eval_health_immunization_001`: `mentions_official_office`

**Failure description:** A second immunization-related evaluation sample also failed the official office mention check. Same pattern as `eval_v3_health_009`.

**Classification:** True model failure (systematic).

**Analysis:** The identical failure on two separate immunization samples confirms this is not an edge case but a systematic behavior gap. The model does not reliably output a named office in immunization-hold response contexts. This strengthens the case for adding a second dedicated DPO pair for this failure pattern. A single revised pair in v1 was insufficient to shift model behavior across different phrasings of the same scenario.

**Action taken in v2:** Same actions as `eval_v3_health_009`. Two DPO pairs now target immunization-hold official-office escalation: the revised `dpo_medical_safe_005` and the new `dpo_medical_safe_011`.

---

### `v7_eval_email_extension_001`: `has_closing`

**Failure description:** An email extension draft did not include a formal closing (Best regards, Sincerely, or Regards). The rule check requires a polite formal closing line.

**Classification:** True model failure, possibly amplified by DPO side effect (unclear without raw output).

**Analysis:** There are two plausible explanations. First, the SFT model may already be borderline on formal closings for extension-specific email prompts. Extension requests may appear in SFT training data with slightly more informal formats than general professional emails, and the model's closing probability for this subtype may have been lower from the start. Second, DPO v1 may have inadvertently affected email closing behavior: in the `dpo_email_quality` rejected answers, several used "Thanks, [Name]" as an informal closing. If the DPO gradient reduced the probability of the informal closing without perfectly reinforcing the formal version, a borderline case might fail to produce "Best regards" entirely. Without the raw response for this specific sample, the causal direction cannot be determined.

What is clear is that none of the v1 DPO pairs specifically targeted the contrast between a formal closing and a missing or informal closing in an extension-email context. The existing pairs targeted other flaws (content quality, extra notes, unsafe language).

**Action taken in v2:** Added new `dpo_email_quality_011` (assignment extension email) where chosen ends with "Best regards, [Your Name]" and rejected ends with "Thanks, [Name]" — a near-miss that fails specifically on the formal-closing requirement.
