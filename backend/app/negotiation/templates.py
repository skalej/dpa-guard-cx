TEMPLATES = {
    "breach_notification_timeline": {
        "ask": "Add a breach notification obligation without undue delay and within 72 hours where feasible.",
        "fallback": "Commit to notification without undue delay and provide a timeline in hours.",
        "rationale": "GDPR Article 33 expects prompt notification; clear timelines reduce ambiguity.",
    },
    "subprocessor_authorization": {
        "ask": "Require prior written consent or general written authorization with a right to object and advance notice.",
        "fallback": "Allow general authorization with notice and objection period for new subprocessors.",
        "rationale": "Controllers must control onward processing and be informed of subprocessor changes.",
    },
    "audit_rights": {
        "ask": "Include audit/inspection rights with reasonable notice and scope.",
        "fallback": "Allow annual audits by a qualified third-party auditor under confidentiality.",
        "rationale": "Audit rights are necessary to verify compliance with Article 28(3)(h).",
    },
    "deletion_or_return": {
        "ask": "Require return or deletion of personal data upon termination with a defined timeframe.",
        "fallback": "Commit to deletion/return within a commercially reasonable timeframe.",
        "rationale": "Article 28(3)(g) requires return or deletion at end of services.",
    },
    "assistance_with_dsar_and_security": {
        "ask": "Processor must assist with data subject requests and maintain appropriate security measures.",
        "fallback": "Provide reasonable assistance and maintain industry-standard technical and organizational measures.",
        "rationale": "Article 28(3)(e)-(f) requires assistance and security obligations.",
    },
    "international_transfers": {
        "ask": "Specify GDPR Chapter V transfer mechanism (e.g., SCCs) for cross-border transfers.",
        "fallback": "Commit to an approved transfer mechanism before any international transfer.",
        "rationale": "Transfers require a lawful mechanism under Chapter V.",
    },
    "confidentiality_of_personnel": {
        "ask": "Ensure personnel are bound by confidentiality obligations.",
        "fallback": "Limit access to authorized personnel subject to confidentiality.",
        "rationale": "Confidentiality is a core safeguard for personal data processing.",
    },
    "purpose_limitation_and_instructions": {
        "ask": "Clarify processing only on documented controller instructions.",
        "fallback": "Limit processing to documented purposes and instructions.",
        "rationale": "Article 28 requires processing only on controller instructions.",
    },
}


def get_template(check_id: str) -> dict:
    return TEMPLATES.get(
        check_id,
        {
            "ask": "Add standard DPA protections for this area.",
            "fallback": "Add reasonable safeguards aligned to GDPR Article 28.",
            "rationale": "Standard DPA safeguard recommended.",
        },
    )
