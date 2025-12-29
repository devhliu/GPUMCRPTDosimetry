"""
GPU-friendly mathematical approximations for Monte Carlo simulations.
These functions provide faster alternatives to standard mathematical operations
while maintaining sufficient accuracy for physics calculations.
"""

import triton
import triton.language as tl


@triton.jit
def fast_log_approx(x: tl.tensor) -> tl.tensor:
    """
    Fast logarithm approximation using a piecewise linear approximation.
    This is significantly faster than tl.log while maintaining reasonable accuracy
    for Monte Carlo applications where exact precision is not critical.
    
    Based on the identity: log(x) = log(2^exp * mant) = exp * log(2) + log(mant)
    where mant is in [1, 2) range.
    """
    # Clamp input to avoid log(0) and negative values
    x = tl.maximum(x, 1e-10)
    
    # Extract exponent and mantissa using bit manipulation
    # Convert float to int representation
    bits = x.to(tl.int32, bitcast=True)
    
    # Extract exponent (biased by 127 for float32)
    exp = ((bits >> 23) & 0xFF) - 127
    
    # Create mantissa in range [1, 2)
    mant_bits = (bits & 0x007FFFFF) | 0x3F800000  # Set exponent to 0 (bias=127)
    mant = mant_bits.to(tl.float32, bitcast=True)
    
    # Log approximation: log(x) ≈ exp * ln(2) + log_approx(mant)
    # For mantissa in [1, 2), use linear approximation
    # log(mant) ≈ (mant - 1) * 0.8667 + (mant - 1)^2 * (-0.222) + (mant - 1)^3 * 0.067
    mant_offset = mant - 1.0
    mant_offset_sq = mant_offset * mant_offset
    mant_offset_cu = mant_offset_sq * mant_offset
    
    log_mant = mant_offset * 0.8667 + mant_offset_sq * (-0.222) + mant_offset_cu * 0.067
    
    # log(2) ≈ 0.693147
    result = exp.to(tl.float32) * 0.693147 + log_mant
    
    return result


@triton.jit
def fast_sqrt_approx(x: tl.tensor) -> tl.tensor:
    """
    Fast square root approximation using Newton-Raphson method with a good initial guess.
    This is faster than tl.sqrt for cases where high precision is not critical.
    """
    # Handle edge cases
    x = tl.maximum(x, 0.0)
    
    # Initial approximation using bit manipulation (0x19000000 creates ~0.414*x^0.5)
    x_half = 0.5 * x
    i = x.to(tl.int32, bitcast=True)  # Get bits as int
    i = 0x5f3759df - (i >> 1)  # Magic number for fast inverse sqrt
    y = i.to(tl.float32, bitcast=True)  # Convert back to float
    
    # Newton-Raphson iteration for inverse sqrt: y = y * (1.5 - x_half * y * y)
    y = y * (1.5 - x_half * y * y)
    y = y * (1.5 - x_half * y * y)  # Second iteration for better accuracy
    
    # To get sqrt(x), multiply by x: sqrt(x) = x * (1/sqrt(x))
    return x * y


@triton.jit
def fast_sin_cos_approx(theta: tl.tensor):
    """
    Fast simultaneous sine and cosine approximation using Taylor series.
    This is more efficient than calling sin and cos separately when both are needed.
    """
    # Reduce theta to [-π, π] range for better accuracy
    # Use the fact that sin/cos have period 2π
    pi = 3.141592653589793
    two_pi = 2.0 * pi

    # Normalize angle to [-π, π]
    n = (theta / two_pi).to(tl.int32)
    theta_norm = theta - n.to(tl.float32) * two_pi

    # For very large |theta|, use more careful normalization
    large_theta = tl.abs(theta_norm) > pi
    n_adj = tl.where(large_theta, n + 1, n)
    theta_norm = tl.where(large_theta, theta - n_adj.to(tl.float32) * two_pi, theta_norm)

    # Use Taylor series approximation up to 5th order
    # sin(x) ≈ x - x^3/6 + x^5/120
    # cos(x) ≈ 1 - x^2/2 + x^4/24
    theta_sq = theta_norm * theta_norm
    theta_cu = theta_sq * theta_norm
    theta_qu = theta_sq * theta_sq
    theta_5th = theta_qu * theta_norm

    sin_val = theta_norm - theta_cu / 6.0 + theta_5th / 120.0
    cos_val = 1.0 - theta_sq / 2.0 + theta_qu / 24.0

    # Apply range reduction corrections if needed
    return sin_val, cos_val


@triton.jit
def fast_sin_approx(theta: tl.tensor) -> tl.tensor:
    """
    Fast sine approximation using Taylor series.
    This is more efficient than tl.sin for cases where high precision is not critical.
    """
    # Reduce theta to [-π, π] range for better accuracy
    pi = 3.141592653589793
    two_pi = 2.0 * pi

    # Normalize angle to [-π, π]
    n = (theta / two_pi).to(tl.int32)
    theta_norm = theta - n.to(tl.float32) * two_pi

    # For very large |theta|, use more careful normalization
    large_theta = tl.abs(theta_norm) > pi
    n_adj = tl.where(large_theta, n + 1, n)
    theta_norm = tl.where(large_theta, theta - n_adj.to(tl.float32) * two_pi, theta_norm)

    # Use Taylor series approximation up to 5th order
    # sin(x) ≈ x - x^3/6 + x^5/120
    theta_sq = theta_norm * theta_norm
    theta_cu = theta_sq * theta_norm
    theta_qu = theta_sq * theta_sq
    theta_5th = theta_qu * theta_norm

    sin_val = theta_norm - theta_cu / 6.0 + theta_5th / 120.0

    return sin_val


@triton.jit
def fast_cos_approx(theta: tl.tensor) -> tl.tensor:
    """
    Fast cosine approximation using Taylor series.
    This is more efficient than tl.cos for cases where high precision is not critical.
    """
    # Reduce theta to [-π, π] range for better accuracy
    pi = 3.141592653589793
    two_pi = 2.0 * pi

    # Normalize angle to [-π, π]
    n = (theta / two_pi).to(tl.int32)
    theta_norm = theta - n.to(tl.float32) * two_pi

    # For very large |theta|, use more careful normalization
    large_theta = tl.abs(theta_norm) > pi
    n_adj = tl.where(large_theta, n + 1, n)
    theta_norm = tl.where(large_theta, theta - n_adj.to(tl.float32) * two_pi, theta_norm)

    # Use Taylor series approximation up to 4th order
    # cos(x) ≈ 1 - x^2/2 + x^4/24
    theta_sq = theta_norm * theta_norm
    theta_qu = theta_sq * theta_sq

    cos_val = 1.0 - theta_sq / 2.0 + theta_qu / 24.0

    return cos_val


@triton.jit
def fast_acos_approx(x: tl.tensor) -> tl.tensor:
    """
    Fast arccosine approximation using polynomial approximation.
    Useful for cases where high precision is not critical.
    """
    # Clamp input to valid range [-1, 1]
    x = tl.maximum(-1.0, tl.minimum(1.0, x))
    
    # Use polynomial approximation for acos
    # Based on the identity: acos(x) = π/2 - asin(x)
    # And asin(x) ≈ x + x^3/6 + 3x^5/40 + 5x^7/112 for x in [-1,1]
    x_abs = tl.abs(x)
    sign = tl.where(x >= 0, 1.0, -1.0)
    
    # For |x| close to 1, use different approximation
    near_one = x_abs > 0.95
    
    # Polynomial approximation for asin
    x_sq = x * x
    x_cu = x_sq * x
    x_5th = x_sq * x_cu
    x_7th = x_sq * x_5th
    
    asin_approx = x + x_cu / 6.0 + 3.0 * x_5th / 40.0 + 5.0 * x_7th / 112.0
    
    # Alternative for values near 1
    sqrt_term = tl.sqrt(1.0 - x_abs)
    asin_near_one = sign * (1.5707963267948966 - 2.0 * sqrt_term * (1.0 + x_abs * (1.0/6.0 + x_abs * (3.0/40.0 + x_abs * (5.0/112.0)))))
    
    asin_result = tl.where(near_one, asin_near_one, asin_approx)
    
    # acos(x) = π/2 - asin(x)
    pi_half = 1.5707963267948966
    return pi_half - asin_result


@triton.jit
def fast_exp_approx(x: tl.tensor) -> tl.tensor:
    """
    Fast exponential approximation using Pade approximation or piecewise linearization.
    This is significantly faster than tl.exp while maintaining reasonable accuracy
    for Monte Carlo applications where exact precision is not critical.
    
    For x in [-1, 1], uses Pade approximant [1,2] which is more accurate than Taylor series.
    For other ranges, uses the identity exp(x) = exp(floor(x)) * exp(x - floor(x))
    where n = round(x/ln(2)), so that |x - n*ln(2)| < ln(2)/2 ≈ 0.346
    """
    # Handle very negative values that would underflow
    x = tl.maximum(x, -700.0)  # Prevent underflow
    
    # For small values use Pade approximant [1,2]: exp(x) ≈ (1 + x/2) / (1 - x/2 + x²/12)
    x_abs = tl.abs(x)
    small_x = x_abs < 0.5
    
    # Pade approximant [1,2] for small x
    x_half = x * 0.5
    x_squared_12 = x * x * (1.0/12.0)
    pade_num = 1.0 + x_half
    pade_den = 1.0 - x_half + x_squared_12
    pade_result = pade_num / tl.maximum(pade_den, 1e-10)
    
    # For larger values, use range reduction: exp(x) = 2^n * exp(x - n*ln(2))
    # where n = round(x/ln(2)), so that |x - n*ln(2)| < ln(2)/2 ≈ 0.346
    ln2 = 0.6931471805599453
    n = (x / ln2 + 0.5).to(tl.int32)  # Round to nearest integer
    reduced_x = x - n.to(tl.float32) * ln2
    
    # Use Pade approximant for reduced x (which is small)
    reduced_x_half = reduced_x * 0.5
    reduced_x_squared_12 = reduced_x * reduced_x * (1.0/12.0)
    pade_num_red = 1.0 + reduced_x_half
    pade_den_red = 1.0 - reduced_x_half + reduced_x_squared_12
    pade_result_red = pade_num_red / tl.maximum(pade_den_red, 1e-10)
    
    # Scale by 2^n using bit manipulation
    # Convert n to float by adding to exponent of 1.0 (which has exponent bias of 127)
    exp_n_bits = (n + 127).to(tl.int32) << 23
    exp_n = exp_n_bits.to(tl.float32, bitcast=True)
    
    range_reduced_result = pade_result_red * exp_n
    
    # Use appropriate result based on input size
    result = tl.where(small_x, pade_result, range_reduced_result)
    
    return result