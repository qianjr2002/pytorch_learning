from dtw_c import dtw_c as dtw
import numpy as np

# Example data: Replace org and trg with your original and target sequences
org = [1.0, 2.5, 3.0, 4.2, 5.1]  # Original sequence
trg = [1.0, 2.0, 3.5, 4.0, 5.5]  # Target sequence

# Reshape the sequences to be 2D arrays with shape (n_samples, n_features)
org = np.array(org, dtype=np.float64).reshape(-1, 1)
trg = np.array(trg, dtype=np.float64).reshape(-1, 1)

# Additional parameters (sdim, ldim, shiftm, winlenm)
sdim = 1     # Static dimension (usually set to 1 for 1D features)
ldim = 1     # Local dimension (usually set to 1)
shiftm = 0   # Shift parameter for DTW (usually 0)
winlenm = 5  # Window length for local comparison

# Call the dtw_org_to_trg function
dtw_org, twf_mat, mcd, mcd_mat = dtw.dtw_org_to_trg(
    org, trg, sdim, ldim, shiftm, winlenm)

# Output results
print("DTW alignment:", dtw_org)
print("Time warping function matrix (TWF):", twf_mat)
print("Mel-Cepstral Distortion (MCD):", mcd)
print("MCD matrix:", mcd_mat)
