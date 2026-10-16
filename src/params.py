"""
Model parameter container for the Vasicek–Garman–Kohlhagen framework.

Domestic (USD) short rate:  dr^d = (a  - b  r^d) dt + sigma   dZ^1
Foreign (EUR) short rate:   dr^f = (c_hat - b_hat r^f) dt + sigma_hat dZ^2  [under Q_d]
FX spot (USD/EUR):          dQ   = Q(r^d - r^f + lambda^q) dt + Q sigma_bar dZ^3

Baseline parameters from Table 1 of the paper.
"""

from dataclasses import dataclass, field


@dataclass
class ModelParams:
    """All model parameters for the domestic Vasicek + foreign Vasicek + FX model."""

    # --- Domestic (USD) Vasicek parameters ---
    r_d_0: float = 0.03670      # initial domestic short rate
    a: float = 0.1101           # domestic drift constant  (long-run mean = a/b)
    b: float = 3.0              # domestic mean-reversion speed
    sigma: float = 0.012        # domestic short-rate volatility

    # --- Foreign (EUR) Vasicek parameters under Q_d ---
    r_f_0: float = 0.01933      # initial foreign short rate
    c_hat: float = 0.057915     # foreign drift constant under Q_d  (hat{c})
    b_hat: float = 3.0          # foreign mean-reversion speed
    sigma_hat: float = 0.010    # foreign short-rate volatility

    # --- FX (USD/EUR) parameters ---
    Q_0: float = 1.1782         # initial FX spot (USD per EUR)
    sigma_bar: float = 0.075    # FX volatility  (bar{sigma})

    # --- Correlation matrix (Z^1=domestic, Z^2=foreign, Z^3=FX) ---
    rho_12: float = 0.25        # Corr(dZ^1, dZ^2)
    rho_13: float = -0.25       # Corr(dZ^1, dZ^3)
    rho_23: float = 0.10        # Corr(dZ^2, dZ^3)

    # --- Collateral net-carry assumptions (default: Table 1 values) ---
    # alpha^{c,d} - alpha^d  (USD collateral net carry)
    carry_d: float = 0.0000     # alpha^{c,d} - alpha^d
    # alpha^{c,f} - alpha^f  (EUR collateral net carry)
    carry_f: float = 0.0020     # alpha^{c,f} - alpha^f

    # --- Spread normalisation ---
    # We choose alpha^d = 0 as a numeraire, so:
    #   alpha^{beta,d} = alpha^d + beta*(alpha^{c,d} - alpha^d) = beta * carry_d
    #   alpha^{beta,f} = alpha^d + beta*(alpha^{c,f} - alpha^f)   = beta * carry_f
    # (alpha^d = 0 assumed throughout; only net-carry differences matter.)

    def alpha_beta(self, beta: float, collateral: str) -> float:
        """
        Compute the effective collateral-adjusted carry rate alpha^{beta,i}.

        Parameters
        ----------
        beta : float
            Collateralisation fraction (1 = full, 0 = uncollateralised).
        collateral : str
            Collateral currency: 'USD' or 'EUR'.

        Returns
        -------
        float
            alpha^{beta,i} = alpha^d + beta*(alpha^{c,i} - alpha^i).
        """
        if collateral.upper() == "USD":
            return beta * self.carry_d          # = 0 under baseline
        elif collateral.upper() == "EUR":
            return beta * self.carry_f          # = 0.002 under baseline
        else:
            raise ValueError(f"Unknown collateral currency: {collateral}")

    def lambda_q(self) -> float:
        """FX drift adjustment lambda^q = alpha^d - alpha^f = 0 under normalisation."""
        return 0.0


@dataclass
class ContractParams:
    """Parameters describing a single USD-settled differential swap."""

    # gamma=0: fixed vs domestic floating; gamma=1: fixed vs (domestic-foreign floating)
    gamma: int = 1              # 0 or 1
    kappa: float = 0.0          # fixed margin (in decimal; set to par at inception)
    beta: float = 1.0           # collateralisation fraction
    collateral: str = "USD"     # 'USD' or 'EUR'
    notional: float = 1.0       # notional (use 1e8 for USD-value reporting)

    def __post_init__(self) -> None:
        assert self.gamma in (0, 1), "gamma must be 0 or 1"
        assert 0.0 <= self.beta <= 1.0, "beta must be in [0, 1]"
        assert self.collateral.upper() in ("USD", "EUR"), "collateral must be USD or EUR"
