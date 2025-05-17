def calculate_mutual_fund_lumpsum(lumpsum_amount, expected_annual_return_rate, investment_duration_years):
    """
    Calculates estimated future value for a lumpsum mutual fund investment.
    Inputs:
        lumpsum_amount (float): Lumpsum investment amount.
        expected_annual_return_rate (float): Expected annual rate of return (in %).
        investment_duration_years (int): Investment duration in years.
    Outputs:
        tuple: (estimated_future_value, estimated_gains) or (None, None) on error.
    """
    try:
        r = expected_annual_return_rate / 100 # Annual rate of return
        n = investment_duration_years
        
        # FV = P * (1 + r)^n
        estimated_future_value = lumpsum_amount * (1 + r)**n
        estimated_gains = estimated_future_value - lumpsum_amount
        
        return round(estimated_future_value, 2), round(estimated_gains, 2)
    except Exception as e:
        print(f"Error in Lumpsum calculation: {e}")
        return None, None
