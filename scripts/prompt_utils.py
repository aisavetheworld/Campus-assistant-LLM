"""Shared prompt template and output-boundary helpers for SFT/inference/eval."""

from __future__ import annotations

from typing import Any


ROLE_CONSTRAINT = (
    "You are an assistant helping international students navigate campus administrative tasks. "
    "You are not the university office, professor, housing office, insurance office, or "
    "legal/medical advisor. Do not pretend to be an official office."
)

SYSTEM_MESSAGE = (
    f"{ROLE_CONSTRAINT} Give safe, practical guidance. For visa, legal, medical, "
    "insurance, academic discipline, or other high-risk questions, avoid definitive "
    "conclusions and refer the student to the relevant official office or qualified professional."
)

STOP_SEQUENCES = [
    "\n### Instruction:",
    "\n### Input:",
    "\n### Response:",
    "\n### Output:",
    "\n### Refine:",
    "\nHuman:",
    "\nAssistant:",
    "\nUser:",
    "\nNote:",
    "\nExplanation:",
    "\nAdditional Tips:",
    "\n**Note:**",
    "\n**Additional Tips:**",
    "\n**Email Draft:**",
    "\nPlease replace",
    "\nEnsure to replace",
    "\nRemember to replace",
]

ROLE_LABEL_STOP_SEQUENCES = [
    "\nAssistant:",
    "\nHuman:",
    "\nUser:",
]

PROMPT_LABEL_STOP_SEQUENCES = [
    "\n### Instruction:",
    "\n### Input:",
    "\n### Response:",
    "\n### Output:",
    "\n### Refine:",
]

PROMPT_LEAKAGE_STOP_SEQUENCES = ROLE_LABEL_STOP_SEQUENCES + PROMPT_LABEL_STOP_SEQUENCES

EXTRA_NOTE_MARKERS = [
    "Human:",
    "Assistant:",
    "User:",
    "### Instruction:",
    "### Input:",
    "### Response:",
    "### Output:",
    "### Refine:",
    "Note:",
    "Explanation:",
    "Additional Tips:",
    "This email",
    "The above draft",
    "Why this works",
    "Email Draft:",
    "Please replace",
    "Ensure to replace",
    "Remember to replace",
    "You can customize",
]


def language_requirement(response_language: str) -> str:
    """Return a plain-English response-language requirement."""
    if response_language == "en":
        return "Respond in English."
    if response_language == "zh":
        return "Respond in Chinese."
    if response_language == "bilingual":
        return "Respond bilingually using English and Chinese where helpful."
    return "Use the response language requested by the task metadata."


def has_any_keyword(text: str, keywords: list[str]) -> bool:
    """Return whether any keyword appears in text, case-insensitively."""
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def escalation_requirements(
    category: str,
    risk_level: str,
    instruction: str,
    input_text: str,
) -> list[str]:
    """Generate category/risk-aware official escalation requirements."""
    if risk_level not in {"medium", "high"}:
        return []

    combined = f"{instruction}\n{input_text}"
    requirements: list[str] = []

    if category == "housing":
        requirements.append(
            "Mention the housing office, Student Mail, mailroom team, or relevant housing team when official help is needed."
        )
    elif category == "course_enrollment":
        requirements.append("Mention an academic advisor, department, or registrar for academic enrollment issues.")
        if risk_level == "high" and has_any_keyword(
            combined,
            ["visa", "status", "full-time", "full time", "cpt", "opt", "sevis", "reduced course load"],
        ):
            requirements.append("Mention the international student office for visa/status/CPT/OPT/full-time enrollment risk.")
    elif category == "health_insurance":
        if has_any_keyword(
            combined,
            [
                "medical",
                "symptom",
                "symptoms",
                "treatment",
                "doctor",
                "chest pain",
                "dizziness",
                "fever",
                "breathing",
                "medication",
                "allergic",
                "healthcare",
                "心理",
                "头晕",
                "呼吸",
                "发烧",
            ],
        ):
            requirements.append("Mention a healthcare provider or student health center for medical or symptom-related issues.")
        if has_any_keyword(combined, ["insurance claim", "claim", "waiver", "bill", "coverage", "appeal"]):
            requirements.append("Mention the insurance office or insurance provider for insurance claim, waiver, billing, or coverage issues.")

    return requirements


def format_requirements(
    output_format: str,
    risk_level: str,
    response_language: str,
    category: str = "general",
    instruction: str = "",
    input_text: str = "",
) -> list[str]:
    """Build prompt requirements from metadata."""
    requirements = [
        language_requirement(response_language),
        "Begin directly with the answer requested by the Output Format.",
        "Never write role labels or speaker labels.",
        "Never write prompt-section labels or restart the prompt.",
        "Stop immediately after the final answer. Do not start a new turn, new prompt, note, explanation, or second draft.",
    ]

    if output_format in {"email_template", "steps_plus_email"}:
        requirements.extend(
            [
                'Only provide the email draft.',
                'The first line must start with "Subject:".',
                'Include a greeting that starts with "Dear".',
                'End immediately after the closing, such as "Best regards,\\n[Your Name]".',
                "Do not add notes, explanations, tips, analysis, replacement instructions, or another draft.",
            ]
        )
    elif output_format == "steps":
        requirements.extend(
            [
                "Use a numbered list with at least 3 steps.",
                "The first line of the response must start with `1.`.",
                "Start each step with `1.`, `2.`, `3.`.",
                "Do not answer in a single paragraph.",
                "Stop after the final numbered step.",
            ]
        )
    elif output_format == "safe_escalation":
        requirements.extend(
            [
                "Use a safe escalation style with clear boundaries.",
                "Do not give absolute promises or guaranteed outcomes.",
                "Refer the user to the relevant official office or qualified professional.",
                "Use a numbered list with at least 3 steps.",
                "The first line of the response must start with `1.`.",
                "Start each step with `1.`, `2.`, `3.`.",
                "Do not answer in a single paragraph.",
                "Stop after the final numbered step.",
            ]
        )
    else:
        requirements.append("Provide a concise campus-assistant answer.")

    requirements.extend(escalation_requirements(category, risk_level, instruction, input_text))

    if risk_level == "high":
        requirements.append("For high-risk issues, avoid definitive legal, visa, medical, insurance, or disciplinary conclusions.")

    return requirements


def resolve_prompt_template(prompt_template: str, model_name_or_path: str = "") -> str:
    """Resolve prompt template selection, defaulting Qwen instruct models to chat."""
    normalized = (prompt_template or "auto").lower()
    if normalized in {"chat", "legacy"}:
        return normalized
    if normalized != "auto":
        raise ValueError(f"Unsupported prompt_template: {prompt_template}")
    return "chat" if "qwen" in model_name_or_path.lower() else "legacy"


def build_user_content(
    instruction: str,
    input_text: str,
    *,
    category: str = "general",
    risk_level: str = "low",
    output_format: str = "plain_answer",
    user_language: str = "mixed",
    response_language: str = "en",
) -> str:
    """Build the user message content without legacy prompt labels."""
    requirements = "\n".join(
        f"- {item}"
        for item in format_requirements(
            output_format,
            risk_level,
            response_language,
            category=category,
            instruction=instruction,
            input_text=input_text,
        )
    )
    return (
        "Task metadata:\n"
        f"- Category: {category}\n"
        f"- Risk level: {risk_level}\n"
        f"- Output format: {output_format}\n"
        f"- User language: {user_language}\n"
        f"- Response language: {response_language}\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Student instruction:\n{instruction.strip()}\n\n"
        f"Student context:\n{input_text.strip()}"
    )


def build_messages(
    instruction: str,
    input_text: str,
    *,
    category: str = "general",
    risk_level: str = "low",
    output_format: str = "plain_answer",
    user_language: str = "mixed",
    response_language: str = "en",
    output: str | None = None,
) -> list[dict[str, str]]:
    """Build system/user/assistant chat messages for chat-template training or inference."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_user_content(
                instruction,
                input_text,
                category=category,
                risk_level=risk_level,
                output_format=output_format,
                user_language=user_language,
                response_language=response_language,
            ),
        },
    ]
    if output is not None:
        messages.append({"role": "assistant", "content": output.strip()})
    return messages


def ensure_text_ends_with_eos(text: str, eos_token: str | None) -> str:
    """Ensure rendered training text explicitly ends with the tokenizer EOS token."""
    if not eos_token:
        return text
    stripped = text.rstrip()
    if stripped.endswith(eos_token):
        return stripped
    return f"{stripped}{eos_token}"


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    ensure_eos: bool = False,
) -> str:
    """Render messages with tokenizer.apply_chat_template and optional EOS enforcement."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("prompt_template='chat' requires a tokenizer with apply_chat_template.")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if ensure_eos:
        rendered = ensure_text_ends_with_eos(rendered, getattr(tokenizer, "eos_token", None))
    return rendered


def build_prompt(
    instruction: str,
    input_text: str,
    *,
    category: str = "general",
    risk_level: str = "low",
    output_format: str = "plain_answer",
    user_language: str = "mixed",
    response_language: str = "en",
    output: str | None = None,
    eos_marker: str = "",
    prompt_template: str = "legacy",
    tokenizer: Any | None = None,
    model_name_or_path: str = "",
) -> str:
    """Create the shared metadata-aware SFT/inference prompt."""
    resolved_template = resolve_prompt_template(prompt_template, model_name_or_path)
    if resolved_template == "chat":
        messages = build_messages(
            instruction,
            input_text,
            category=category,
            risk_level=risk_level,
            output_format=output_format,
            user_language=user_language,
            response_language=response_language,
            output=output,
        )
        return apply_chat_template_text(
            tokenizer,
            messages,
            add_generation_prompt=output is None,
            ensure_eos=output is not None,
        )

    requirements = "\n".join(
        f"- {item}"
        for item in format_requirements(
            output_format,
            risk_level,
            response_language,
            category=category,
            instruction=instruction,
            input_text=input_text,
        )
    )
    prompt = (
        f"### Role:\n{ROLE_CONSTRAINT}\n\n"
        "### Metadata:\n"
        f"Category: {category}\n"
        f"Risk Level: {risk_level}\n"
        f"Output Format: {output_format}\n"
        f"User Language: {user_language}\n"
        f"Response Language: {response_language}\n\n"
        f"### Requirements:\n{requirements}\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Input:\n{input_text.strip()}\n\n"
        "### Response:\n"
    )
    if output is not None:
        prompt += output.strip()
        if eos_marker:
            prompt += eos_marker
    return prompt


def build_prompt_from_sample(
    sample: dict[str, Any],
    *,
    output: str | None = None,
    eos_marker: str = "",
    prompt_template: str = "legacy",
    tokenizer: Any | None = None,
    model_name_or_path: str = "",
) -> str:
    """Build a prompt from a metadata-rich sample dictionary."""
    return build_prompt(
        instruction=sample["instruction"],
        input_text=sample.get("input", ""),
        category=sample.get("category", "general"),
        risk_level=sample.get("risk_level", "low"),
        output_format=sample.get("output_format", "plain_answer"),
        user_language=sample.get("user_language", "mixed"),
        response_language=sample.get("response_language", "en"),
        output=output,
        eos_marker=eos_marker,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        model_name_or_path=model_name_or_path,
    )


def truncate_with_stop_info(
    text: str, stop_sequences: list[str] | None = None
) -> tuple[str, bool, str | None]:
    """Truncate generated text and return stop-sequence metadata."""
    stops = stop_sequences or STOP_SEQUENCES
    earliest: int | None = None
    stop_used: str | None = None
    for stop in stops:
        idx = find_stop_position(text, stop)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
            stop_used = stop
    if earliest is None:
        return text.strip(), False, None
    truncated = text[:earliest].rstrip()
    if truncated.endswith("---"):
        truncated = truncated[:-3].rstrip()
    return truncated.strip(), True, stop_used


def count_words(text: str) -> int:
    """Count rough whitespace-separated words."""
    return len(text.strip().split())


def find_stop_position(text: str, stop: str) -> int:
    """Find stop sequence position, including line-start markers at response start."""
    idx = text.find(stop)
    if idx != -1:
        return idx
    if stop.startswith("\n") and text.startswith(stop[1:]):
        return 0
    return -1


def clean_generation_with_info(
    text: str,
    stop_sequences: list[str] | None = None,
    early_truncation_word_threshold: int = 40,
) -> tuple[str, dict[str, Any]]:
    """Clean generated text while preserving boundary diagnostics.

    Extra-note markers after a complete answer are normal post-processing
    truncation. Prompt/role labels before a complete answer are tracked as early
    truncation and raw prompt leakage instead of being treated as a clean answer.
    """
    stops = stop_sequences or STOP_SEQUENCES
    raw_text = text.strip()
    cleaned = raw_text

    while True:
        earliest: int | None = None
        stop_used: str | None = None
        for stop in stops:
            idx = find_stop_position(cleaned, stop)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx
                stop_used = stop
        if earliest is None or stop_used is None:
            info = {
                "raw_response": raw_text,
                "raw_response_length": len(raw_text),
                "raw_response_word_count": count_words(raw_text),
                "response_length": len(cleaned.strip()),
                "response_word_count": count_words(cleaned),
                "was_truncated_by_stop_sequence": False,
                "stop_sequence_used": None,
                "is_prompt_leakage_stop": False,
                "is_extra_note_stop": False,
                "is_early_truncation": False,
                "raw_prompt_leakage_detected": has_prompt_leakage(raw_text),
                "postprocessed_prompt_leakage_detected": has_prompt_leakage(cleaned),
            }
            return cleaned.strip(), info

        before = cleaned[:earliest].rstrip()
        if before.endswith("---"):
            before = before[:-3].rstrip()
        response_word_count = count_words(before)
        is_prompt_leakage_stop = stop_used in PROMPT_LEAKAGE_STOP_SEQUENCES
        info = {
            "raw_response": raw_text,
            "raw_response_length": len(raw_text),
            "raw_response_word_count": count_words(raw_text),
            "response_length": len(before.strip()),
            "response_word_count": response_word_count,
            "was_truncated_by_stop_sequence": True,
            "stop_sequence_used": stop_used,
            "is_prompt_leakage_stop": is_prompt_leakage_stop,
            "is_extra_note_stop": not is_prompt_leakage_stop,
            "is_early_truncation": response_word_count < early_truncation_word_threshold,
            "raw_prompt_leakage_detected": has_prompt_leakage(raw_text),
            "postprocessed_prompt_leakage_detected": has_prompt_leakage(before),
        }
        return before.strip(), info


def truncate_at_stop_sequences(text: str, stop_sequences: list[str] | None = None) -> str:
    """Truncate generated text at the earliest configured stop sequence."""
    truncated_text, _, _ = truncate_with_stop_info(text, stop_sequences)
    return truncated_text


def has_prompt_leakage(text: str) -> bool:
    """Return whether generated text contains role or prompt-label leakage."""
    leakage_markers = [
        "Human:",
        "Assistant:",
        "User:",
        "### Instruction:",
        "### Response:",
        "### Output:",
        "### Refine:",
    ]
    lower = text.lower()
    return any(marker.lower() in lower for marker in leakage_markers)
