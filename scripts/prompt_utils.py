"""Shared prompt template and output-boundary helpers for SFT/inference/eval."""

from __future__ import annotations

from typing import Any


ROLE_CONSTRAINT = (
    "You are an assistant helping international students navigate campus administrative tasks. "
    "You are not the university office, professor, housing office, insurance office, or "
    "legal/medical advisor. Do not pretend to be an official office."
)

STOP_SEQUENCES = [
    "\n---",
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


def format_requirements(output_format: str, risk_level: str, response_language: str) -> list[str]:
    """Build prompt requirements from metadata."""
    requirements = [
        language_requirement(response_language),
        "Answer only as the assistant. Do not continue the conversation as Human, User, or Assistant.",
        "Do not include prompt labels such as ### Instruction, ### Input, ### Response, ### Output, or ### Refine in the response.",
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
        requirements.append("Use concise, actionable steps.")
    elif output_format == "safe_escalation":
        requirements.extend(
            [
                "Use a safe escalation style with clear boundaries.",
                "Do not give absolute promises or guaranteed outcomes.",
                "Refer the user to the relevant official office or qualified professional.",
                "Use concise, actionable steps when appropriate.",
            ]
        )
    else:
        requirements.append("Provide a concise campus-assistant answer.")

    if risk_level == "high":
        requirements.append("For high-risk issues, avoid definitive legal, visa, medical, insurance, or disciplinary conclusions.")

    return requirements


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
) -> str:
    """Create the shared metadata-aware SFT/inference prompt."""
    requirements = "\n".join(
        f"- {item}" for item in format_requirements(output_format, risk_level, response_language)
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
    )


def truncate_at_stop_sequences(text: str, stop_sequences: list[str] | None = None) -> str:
    """Truncate generated text at the earliest configured stop sequence."""
    stops = stop_sequences or STOP_SEQUENCES
    earliest: int | None = None
    for stop in stops:
        idx = text.find(stop)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text.strip()
    return text[:earliest].strip()
