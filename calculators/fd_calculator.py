def calculate_fd(principal, rate, tenure_years):
    """
    Calculates Fixed Deposit maturity amount and interest earned.
    Inputs:
        principal (float): Principal amount.
        rate (float): Annual interest rate (in %).
        tenure_years (float): Tenure in years.
    Outputs:
        tuple: (maturity_amount, interest_earned) or (None, None) on error.
    """
    try:
        maturity_amount = principal * (1 + rate / 100) ** tenure_years
        interest_earned = maturity_amount - principal
        return round(maturity_amount, 2), round(interest_earned, 2)
    except Exception as e:
        print(f"Error in FD calculation: {e}")
        return None, None
