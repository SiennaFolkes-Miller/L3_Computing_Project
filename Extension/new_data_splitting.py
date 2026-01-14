import numpy as np

def load_scp_data(filename):
    """
    Load Union supernova data and extract:
    redshift, distance modulus, distance modulus error.
    """

    redshift = []
    mu = []
    mu_err = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith('%'):
                continue

            # Split LaTeX table row
            parts = [p.strip() for p in line.split('&')]

            # Need enough columns
            if len(parts) < 7:
                continue

            try:
                # Redshift
                z = float(parts[1])

                # Distance modulus: e.g. "35.35(0.22)"
                mu_str = parts[5]

                value, error = mu_str.split('(')
                value = float(value)
                error = float(error.rstrip(')'))

                redshift.append(z)
                mu.append(value)
                mu_err.append(error)

            except ValueError:
                continue

    return np.array(redshift), np.array(mu), np.array(mu_err)
