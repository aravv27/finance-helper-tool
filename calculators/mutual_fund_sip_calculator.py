def calculate_mutual_fund_sip(monthly_sip_amount, expected_annual_return_rate, investment_duration_years):
    """
    Calculates estimated future value for a SIP investment.
    Inputs:
        monthly_sip_amount (float): Monthly SIP amount.
        expected_annual_return_rate (float): Expected annual rate of return (in %).
        investment_duration_years (int): Investment duration in years.
    Outputs:
        tuple: (estimated_future_value, total_amount_invested, estimated_gains) or (None, None, None) on error.
    """
    try:
        i = expected_annual_return_rate / (12 * 100)  # Monthly rate of return
        n = investment_duration_years * 12           # Number of months
        
        # M = P × ({[1 + i]^n – 1} / i) × (1 + i)
        estimated_future_value = monthly_sip_amount * ((((1 + i)**n) - 1) / i) * (1 + i)
        total_amount_invested = monthly_sip_amount * n
        estimated_gains = estimated_future_value - total_amount_invested
        
        return round(estimated_future_value, 2), round(total_amount_invested, 2), round(estimated_gains, 2)
    except Exception as e:
        print(f"Error in SIP calculation: {e}")
        return None, None, None
