"""Set-cluster bootstrap and support/sign summaries."""

import csv
import numpy as np


def summary(z):
    e, s, p = (z[k] for k in ("eligible", "supported", "pass"))
    n, ns, npass = (int(e.sum()), int(s.sum()), int(p.sum()))
    rng = np.random.default_rng(20260912)
    draws = rng.integers(0, len(s), (2000, len(s)))
    denominators = s.sum(axis=1)[draws].sum(axis=1)
    numerators = p.sum(axis=1)[draws].sum(axis=1)
    boot = numerators[denominators > 0] / denominators[denominators > 0]
    ratio = z["probe_reference_ratio"][s]
    return dict(
        n_eligible=n,
        n_supported=ns,
        n_pass=npass,
        coverage=ns / n if n else None,
        conditional_pass=npass / ns if ns else None,
        pass_fraction_all=npass / n if n else None,
        ci_low=float(np.quantile(boot, 0.025)) if len(boot) else None,
        ci_high=float(np.quantile(boot, 0.975)) if len(boot) else None,
        negative_reference=int(z["negative_reference"].sum()),
        negative_probe=int(z["negative_probe"].sum()),
        negative_probe_supported=int((z["negative_probe"] & s).sum()),
        negative_probe_supported_fraction=(
            float((z["negative_probe"] & s).sum() / ns) if ns else None
        ),
        strict_joint_pass=float(z["strict_joint_pass"].sum() / ns) if ns else None,
        invalid=int(z["invalid"].sum()),
        **{
            f"signed_ratio_q{q}": float(np.percentile(ratio, q)) if len(ratio) else None
            for q in (5, 25, 50, 75, 95)
        },
    )


def write_csv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
