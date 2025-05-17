def calculate_emi(principal, annual_rate, tenure_months):
    """
    Calculates Equated Monthly Installment (EMI) for a loan.
    Inputs:
        principal (float): Principal loan amount.
        annual_rate (float): Annual interest rate (in %).
        tenure_months (int): Loan tenure in months.
    Outputs:
        float: EMI amount or None on error.
    """
    try:
        monthly_rate = annual_rate / (12 * 100)
        if monthly_rate == 0: # Avoid division by zero if interest rate is 0
             emi = principal / tenure_months
        else:
            emi = principal * monthly_rate * (1 + monthly_rate)**tenure_months / ((1 + monthly_rate)**tenure_months - 1)
        return round(emi, 2)
    except Exception as e:
        print(f"Error in Loan EMI calculation: {e}")
        return None
