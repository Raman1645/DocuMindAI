"""
generate_sample_data.py - Helper script to generate sample TXT and PDF test files.

Creates:
  1. rag-poc/sample_data/company_policy.txt (Remote work, security, equipment)
  2. rag-poc/sample_data/employee_handbook.pdf (Code of conduct, PTO, performance reviews)
  3. rag-poc/sample_data/engineering_handbook.txt (PR standards, on-call pay, SEV incident levels)
  4. rag-poc/sample_data/benefits_guide.pdf (Healthcare plans, 401k match, wellness, parental leave)
"""

import sys
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def create_company_policy_txt():
    txt_path = SAMPLE_DIR / "company_policy.txt"
    content = """# ACME Corp Remote Work & Information Security Policy

1. Executive Summary & Purpose
ACME Corp is committed to maintaining operational excellence and employee flexibility.
This policy outlines guidelines for remote work, core collaboration hours, and data security requirements.

2. Core Working Hours & Availability
Employees must maintain presence on Slack and corporate email between 10:00 AM and 4:00 PM EST.
Flexible start and end times outside of these core hours are permitted with direct manager approval.
Core collaboration meetings must be scheduled within this 10:00 AM - 4:00 PM EST window.

3. Home Office Equipment Stipend
All full-time employees are eligible for a one-time setup stipend of $500 to purchase ergonomic desks, monitors, or chairs.
Purchased hardware remains corporate property and must be logged with the IT asset management system.
Reimbursement claims must be submitted within 60 days of the employee's start date via the corporate expense portal.

4. Data Protection & Network Security
Remote workers accessing company repositories, internal staging environments, or customer PII must do so via the company-issued VPN.
Public Wi-Fi networks in coffee shops, hotels, or airports must strictly be accessed through an encrypted VPN tunnel.
All laptops and mobile devices accessing corporate services must have FileVault or BitLocker full-disk encryption enabled.
Screen timeouts must be set to lock automatically after a maximum of 5 minutes of inactivity.

5. Device Loss and Incident Reporting
In the event of a lost or stolen company laptop, employees must notify security@acme.com within 2 hours.
The IT Security team will initiate remote device wipe protocols immediately upon confirmation.
"""
    txt_path.write_text(content, encoding="utf-8")
    print(f"[OK] Created TXT sample: {txt_path}")
    return txt_path


def create_employee_handbook_pdf():
    pdf_path = SAMPLE_DIR / "employee_handbook.pdf"
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # --- Page 1 ---
        story.append(Paragraph("ACME Corp Employee Handbook - Page 1", styles['Title']))
        story.append(Spacer(1, 14))
        story.append(Paragraph("Section 1: Code of Conduct & Workplace Ethics", styles['Heading2']))
        story.append(Paragraph(
            "ACME Corp fosters an inclusive, respectful, and ethical workplace environment. "
            "All team members are expected to conduct themselves professionally and report any compliance concerns "
            "to the Ethics Hotline or People Operations. Discrimination or harassment of any kind is strictly prohibited.",
            styles['Normal']
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Section 2: Health, Safety & Workplace Accommodations", styles['Heading2']))
        story.append(Paragraph(
            "Employee wellbeing is our top priority. Office facilities provide ergonomic workstations, "
            "and comprehensive health insurance coverage begins on the first day of employment. "
            "Ergonomic assessment requests can be submitted to facilities@acme.com.",
            styles['Normal']
        ))
        
        story.append(PageBreak())
        
        # --- Page 2 ---
        story.append(Paragraph("ACME Corp Employee Handbook - Page 2", styles['Title']))
        story.append(Spacer(1, 14))
        story.append(Paragraph("Section 3: Annual Leave Policy & PTO Accrual", styles['Heading2']))
        story.append(Paragraph(
            "Full-time employees receive 20 days of Paid Time Off (PTO) per calendar year, accrued monthly at 1.66 days per month. "
            "Unused PTO up to a maximum of 5 days can be carried over into the next calendar year. Any remaining unused balance beyond 5 days is forfeited on December 31. "
            "Sick leave is allocated as 10 separate days per calendar year and does not roll over.",
            styles['Normal']
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Section 4: Performance Evaluations & Career Progression", styles['Heading2']))
        story.append(Paragraph(
            "Bi-annual performance evaluations take place in June and December. "
            "Mid-year reviews in June focus on goal calibration and feedback, while year-end reviews in December evaluate overall annual achievements. "
            "Merit-based salary adjustments and annual bonuses are finalized during the Q4 review cycle and take effect on February 1.",
            styles['Normal']
        ))
        
        doc.build(story)
        print(f"[OK] Created PDF sample (2 pages): {pdf_path}")
        
    except ImportError:
        print("[WARNING] ReportLab not found. Please run 'pip install reportlab' in your virtual environment.")
        
    return pdf_path


def create_engineering_handbook_txt():
    txt_path = SAMPLE_DIR / "engineering_handbook.txt"
    content = """# ACME Corp Engineering & Technical Operations Handbook

1. Code Review & Pull Request Standards
All production code changes require a minimum of two approving peer reviews before merging into the main branch.
Continuous Integration (CI) test suites and static analysis linters must pass with 100% success rate.
Pull requests should not exceed 400 lines of modified code to ensure thorough and attentive review.

2. On-Call Rotations & Compensation
Engineering on-call rotations operate on a 7-day weekly cycle starting Tuesdays at 10:00 AM EST.
Engineers on primary rotation receive a weekly base on-call stipend of $300.
If an engineer is called to resolve an incident on an official corporate holiday, an additional $100 per holiday shift is compensated.
Secondary on-call engineers receive a $150 weekly stipend for standby escalation.

3. Incident Severity Levels & Response SLAs
Incident severity classifications dictate team response times:
- SEV-1 (Critical outage affecting >25% of users): Initial response within 15 minutes, updates every 30 minutes.
- SEV-2 (Major degradation of core feature): Initial response within 30 minutes, updates every 1 hour.
- SEV-3 (Minor non-critical bug): Initial response within 4 hours during business hours.
- SEV-4 (Cosmetic / low priority issue): Handled in standard sprint planning.

4. Production Release & Deployment Freezes
Standard deployment windows are Monday through Thursday between 9:00 AM and 2:00 PM EST.
Friday deployments are prohibited except for critical SEV-1 emergency hotfixes.
An annual year-end deployment freeze is enforced from December 15 through January 3 to protect holiday system stability.
"""
    txt_path.write_text(content, encoding="utf-8")
    print(f"[OK] Created TXT sample: {txt_path}")
    return txt_path


def create_benefits_guide_pdf():
    pdf_path = SAMPLE_DIR / "benefits_guide.pdf"
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # --- Page 1 ---
        story.append(Paragraph("ACME Corp Comprehensive Benefits Guide - Page 1", styles['Title']))
        story.append(Spacer(1, 14))
        story.append(Paragraph("Section 1: Healthcare & Dental Coverage Tiers", styles['Heading2']))
        story.append(Paragraph(
            "ACME Corp offers two comprehensive healthcare plans through BlueCross: "
            "The Gold PPO Plan has a $500 individual deductible with 90% in-network coverage and a $20 copay for primary care. "
            "The Silver HDHP Plan has a $1,500 individual deductible and is paired with a Health Savings Account (HSA) with an annual company contribution of $750. "
            "Comprehensive dental and vision coverage is fully employer-paid for all full-time employees and their dependents.",
            styles['Normal']
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Section 2: 401(k) Retirement Savings & Vesting", styles['Heading2']))
        story.append(Paragraph(
            "Employees are eligible to enroll in the company 401(k) plan immediately upon hire through Fidelity. "
            "ACME Corp provides a dollar-for-dollar match on employee contributions up to 5% of annual base salary. "
            "Company matching contributions follow a 3-year graded vesting schedule: 33% vested after Year 1, 66% after Year 2, and 100% fully vested after Year 3.",
            styles['Normal']
        ))
        
        story.append(PageBreak())
        
        # --- Page 2 ---
        story.append(Paragraph("ACME Corp Comprehensive Benefits Guide - Page 2", styles['Title']))
        story.append(Spacer(1, 14))
        story.append(Paragraph("Section 3: Annual Wellness & Fitness Stipend", styles['Heading2']))
        story.append(Paragraph(
            "To promote physical and mental wellbeing, all full-time employees are eligible for an annual wellness stipend of $350 per calendar year. "
            "Eligible wellness expenses include gym memberships, fitness trackers, yoga classes, meditation apps, and mental health counseling co-pays. "
            "This wellness stipend is separate from the home office equipment setup stipend.",
            styles['Normal']
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Section 4: Parental Leave & Family Support", styles['Heading2']))
        story.append(Paragraph(
            "ACME Corp provides 16 weeks of 100% fully paid parental leave for all eligible new parents (birthing, non-birthing, and adoptive). "
            "Parental leave can be taken continuously or in two separate blocks within the first 12 months following birth or adoption. "
            "Additionally, new parents receive a one-time $1,000 baby bonding bonus upon commencement of leave.",
            styles['Normal']
        ))
        
        doc.build(story)
        print(f"[OK] Created PDF sample (2 pages): {pdf_path}")
        
    except ImportError:
        print("[WARNING] ReportLab not found. Please run 'pip install reportlab' in your virtual environment.")
        
    return pdf_path


def main():
    print("=== Generating Enriched Multi-Document Evaluation Corpus ===")
    create_company_policy_txt()
    create_employee_handbook_pdf()
    create_engineering_handbook_txt()
    create_benefits_guide_pdf()
    print("Corpus generation completed successfully.")


if __name__ == "__main__":
    main()
