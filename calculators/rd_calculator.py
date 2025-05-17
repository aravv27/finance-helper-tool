def calculate_rd(monthly_installment, rate, tenure_months):
    """
    Calculates Recurring Deposit maturity amount and interest earned.
    Inputs:
        monthly_installment (float): Monthly installment amount.
        rate (float): Annual interest rate (in %).
        tenure_months (int): Tenure in months.
    Outputs:
        tuple: (maturity_amount, total_interest_earned) or (None, None) on error.
    """
    try:
        r = rate / (12 * 100)  # Monthly interest rate
        n = tenure_months       # Number of installments
        
        # M = P * [((1 + r)^n - 1) / r] * (1+r) -- This formula is for end of period payments
        # A more common RD formula where interest is compounded quarterly and payments are monthly:
        # For simplicity, using the formula where interest is compounded monthly for RD
        # M = P * [((1+i)^n - 1)/i] where i = r/12 and n = months
        
        maturity_amount = 0
        current_value = 0
        for _ in range(int(n)):
            current_value += monthly_installment
            current_value *= (1 + r) 
        
        # Alternate simpler calculation often used:
        # M = P * n + P * n * (n+1)/2 * r/12 (approx for simple interest on installments)
        # Using standard compound interest formula for RD:
        # Maturity Amount = P * [((1+R)^N - 1) / R] where R is rate per period, N is number of periods.
        # Here, period is monthly.

        # Let's use the formula: M = P * [((1 + r)^n - 1) / r] * (1 + r) if interest is paid at end,
        # or simply M = P * [((1 + r)^n - 1) / r] if interest is part of the sum.
        # Assuming interest is compounded monthly:
        maturity_amount = monthly_installment * (((1 + r)**n - 1) / r)
        
        total_invested = monthly_installment * n
        total_interest_earned = maturity_amount - total_invested
        
        return round(maturity_amount, 2), round(total_interest_earned, 2)
    except Exception as e:
        print(f"Error in RD calculation: {e}")
        return None, None
