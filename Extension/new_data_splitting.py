import numpy as np

def load_scp_data(filename):
    """
    Load SCP supernova data and extract:
    redshift, effective magnitude, magnitude error.
    """
    
    redshift = []
    m_eff = []
    m_err = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith('%'):
                continue

            # Split LaTeX table row
            parts = [p.strip() for p in line.split('&')]

            # Need at least 3 columns
            if len(parts) < 3:
                continue

            try:
                # Redshift
                z = float(parts[1])

                # Magnitude and error: e.g. "19.27(0.05)"
                mag_str = parts[2]

                value, error = mag_str.split('(')
                value = float(value)
                error = float(error.rstrip(')'))

                redshift.append(z)
                m_eff.append(value)
                m_err.append(error)

            except ValueError:
                # Skip malformed rows
                continue

    return np.array(redshift), np.array(m_eff), np.array(m_err)
