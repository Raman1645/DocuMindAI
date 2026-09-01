"""
generate_sample_data.py - Helper script to generate sample TXT and PDF test files.

Creates:
  1. rag-poc/sample_data/company_policy.txt
  2. rag-poc/sample_data/employee_handbook.pdf (Multi-page PDF for page metadata testing)
"""

import sys
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_txt():
    txt_path = SAMPLE_DIR / "company_policy.txt"
    content = """# ACME Corp Remote Work & Information Security Policy

1. Executive Summary & Purpose
ACME Corp is committed to maintaining operational excellence and employee flexibility.
This policy outlines guidelines for remote work, core collaboration hours, and data security requirements.

2. Core Working Hours & Availability
Employees must maintain presence on Slack and corporate email between 10:00 AM and 4:00 PM EST.
Flexible start and end times outside of these core hours are permitted with direct manager approval.

3. Home Office Equipment Stipend
All full-time employees are eligible for a one-time setup stipend of $500 to purchase ergonomic desks, monitors, or chairs.
Purchased hardware remains corporate property and must be logged with the IT asset management system.

4. Data Protection & Network Security
Remote workers accessing company repositories or customer PII must do so via the company-issued VPN.
Public Wi-Fi networks in coffee shops, hotels, or airports must strictly be accessed through an encrypted VPN tunnel.
"""
    txt_path.write_text(content, encoding="utf-8")
    print(f"[OK] Created TXT sample: {txt_path}")
    return txt_path


def create_sample_pdf():
    pdf_path = SAMPLE_DIR / "employee_handbook.pdf"
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # --- Page 1 ---
        story.append(Paragraph("ACME Corp Employee Handbook - Page 1", styles['Title']))
        story.append(Spacer(1, 18))
        story.append(Paragraph("Section 1: Code of Conduct", styles['Heading2']))
        story.append(Paragraph(
            "ACME Corp fosters an inclusive, respectful, and ethical workplace environment. "
            "All team members are expected to conduct themselves professionally and report any compliance concerns.",
            styles['Normal']
        ))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Section 2: Health and Safety", styles['Heading2']))
        story.append(Paragraph(
            "Employee wellbeing is our top priority. Office facilities provide ergonomic workstations, "
            "and health insurance coverage begins on the first day of employment.",
            styles['Normal']
        ))
        
        story.append(PageBreak())
        
        # --- Page 2 ---
        story.append(Paragraph("ACME Corp Employee Handbook - Page 2", styles['Title']))
        story.append(Spacer(1, 18))
        story.append(Paragraph("Section 3: Annual Leave Policy", styles['Heading2']))
        story.append(Paragraph(
            "Full-time employees receive 20 days of Paid Time Off (PTO) per calendar year. "
            "Unused PTO up to 5 days can be carried over into the next calendar year.",
            styles['Normal']
        ))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Section 4: Performance Reviews", styles['Heading2']))
        story.append(Paragraph(
            "Bi-annual performance evaluations take place in June and December. "
            "Merit-based salary adjustments and annual bonuses are finalized during the Q4 review cycle.",
            styles['Normal']
        ))
        
        doc.build(story)
        print(f"[OK] Created PDF sample (2 pages): {pdf_path}")
        
    except ImportError:
        # Fallback if reportlab is missing: create a minimal PDF via pypdf or PyPDF2 if possible
        print("ReportLab not found; installing reportlab to generate PDF sample...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "reportlab"], check=True)
        create_sample_pdf()
        
    return pdf_path


if __name__ == "__main__":
    print("=== Generating Sample Test Documents ===")
    create_sample_txt()
    create_sample_pdf()
    print("Sample generation complete.")
