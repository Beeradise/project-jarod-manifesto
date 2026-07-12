def kernel_weight(z, z_mid=KERNEL_Z_MID, z_steep=KERNEL_Z_STEEP):
    """Narrow resemblance kernel: w(z) = 1/(1+exp(-(z-z_mid)/z_steep)). Global-majority band earns ~0;
    only specific resemblance accrues evidence."""
    return 1.0 / (1.0 + math.exp(-(z - z_mid) / z_steep))
