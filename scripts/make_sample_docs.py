"""Generate the bundled sample documents in ``sample_docs/``.

These small synthetic PDFs let the system (and the evaluation harness) run
out-of-the-box and mirror the examples in the assignment. Regenerate with:

    pip install reportlab        # only needed to regenerate
    python -m scripts.make_sample_docs

The generated PDFs are committed, so reportlab is NOT a runtime dependency.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_docs"
_STYLES = getSampleStyleSheet()
_H, _B = _STYLES["Heading1"], _STYLES["BodyText"]


def _build(filename: str, pages: list[tuple[str, list[str]]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    story: list = []
    for i, (title, paras) in enumerate(pages):
        story.append(Paragraph(title, _H))
        story.append(Spacer(1, 12))
        for p in paras:
            story.append(Paragraph(p, _B))
            story.append(Spacer(1, 8))
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT / filename), pagesize=LETTER).build(story)
    print(f"wrote {filename} ({len(pages)} pages)")


def main() -> None:
    _build("HR_Policy.pdf", [
        ("HR Policy — Overview", [
            "This document describes the human resources policies for all employees.",
            "It covers working hours, code of conduct, and general expectations.",
        ]),
        ("Working Hours and Remote Work", [
            "Standard working hours are 9:00 AM to 6:00 PM, Monday to Friday.",
            "Employees may work remotely up to two days per week with manager approval.",
        ]),
        ("Leave Policy", [
            "Employees are eligible for 24 paid leaves annually.",
            "Unused leaves may be carried over up to a maximum of 10 days into the next year.",
            "Sick leave of up to 12 days per year is provided separately from paid leave.",
            "Maternity leave is 26 weeks and paternity leave is 4 weeks.",
        ]),
    ])

    _build("Customer_Policy.pdf", [
        ("Customer Policy — Introduction", [
            "This policy governs customer purchases, refunds, and support.",
        ]),
        ("Refund Policy", [
            "Refunds are allowed within 30 days of purchase.",
            "To request a refund, customers must contact support with their order ID.",
            "Refunds are processed to the original payment method within 7 business days.",
        ]),
        ("Support and Warranty", [
            "Standard support is available Monday to Friday, 9 AM to 5 PM.",
            "All hardware products carry a one-year limited warranty.",
        ]),
    ])


if __name__ == "__main__":
    main()
