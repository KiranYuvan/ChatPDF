"""Creates data/pdfs/sample_handbook.pdf (a fictional company handbook) for testing.

python -m scripts.make_sample_pdf
"""

from fpdf import FPDF

from app.config import get_settings

PAGES = {
    "1. Welcome and Working Hours": (
        "Welcome to Northwind Robotics. This handbook explains the policies that apply to all "
        "full-time employees. Our standard working hours are 9:00 AM to 5:30 PM, Monday to Friday, "
        "with a one-hour lunch break. Core collaboration hours, when everyone must be available for "
        "meetings, are 10:00 AM to 3:00 PM in the employee's local time zone.\n\n"
        "Employees may work flexible hours outside the core window if their manager approves. "
        "Overtime must be approved in advance by a department head and is compensated at 1.5 times "
        "the normal hourly rate."
    ),
    "2. Leave Policy": (
        "Every full-time employee receives 24 days of paid annual leave per calendar year. Leave "
        "accrues at 2 days per month. Up to 5 unused days may be carried over into the next year; "
        "any additional unused days are forfeited on December 31.\n\n"
        "Sick leave is separate: employees get 10 paid sick days per year. A doctor's note is "
        "required for sick leave longer than 3 consecutive days. Parental leave is 16 weeks fully "
        "paid for the primary caregiver and 6 weeks fully paid for the secondary caregiver.\n\n"
        "Leave requests must be submitted in the HR portal at least 2 weeks in advance for "
        "absences longer than 3 days."
    ),
    "3. Remote Work": (
        "Northwind Robotics follows a hybrid model. Employees may work remotely up to 3 days per "
        "week. Tuesdays and Thursdays are in-office days for all teams based at the Austin campus.\n\n"
        "The company provides a one-time home office stipend of 750 US dollars for a desk, chair "
        "or monitor. Internet costs are reimbursed up to 50 US dollars per month. Employees working "
        "remotely must use the company VPN when accessing internal systems."
    ),
    "4. Expenses and Travel": (
        "Business travel must be booked through the TravelDesk tool. Economy class is required for "
        "flights under 6 hours; premium economy is allowed for longer flights.\n\n"
        "The daily meal allowance while travelling is 75 US dollars. Hotel stays are capped at "
        "220 US dollars per night unless the travel city is on the high-cost list. Expense reports "
        "must be submitted within 30 days with itemized receipts for any purchase over 25 US dollars. "
        "Reports submitted after 60 days will not be reimbursed."
    ),
    "5. IT Security": (
        "Passwords must be at least 14 characters and are rotated every 180 days. Multi-factor "
        "authentication is mandatory for email, the HR portal and source control.\n\n"
        "Lost or stolen laptops must be reported to the IT helpdesk within 24 hours by calling "
        "extension 4357. Employees must not install unapproved software on company devices; the "
        "list of approved tools is maintained on the IT intranet page."
    ),
}


def main() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for title, body in PAGES.items():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, body)

    s = get_settings()
    s.ensure_dirs()
    out = s.pdf_dir / "sample_handbook.pdf"
    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
