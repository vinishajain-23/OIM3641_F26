def calculate_loan_payment(interest, term, present_value):
    monthly_interest = (interest / 100) / 12
    if monthly_interest == 0:
        return present_value / term
    payment = (present_value * monthly_interest) / (1 - (1 + monthly_interest) ** -term)
    return payment