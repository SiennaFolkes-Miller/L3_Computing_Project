import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv

# --------------------------------------------------
# 1. LOAD DIGITIZED DATA
# --------------------------------------------------
data = np.array([
    [0.2520096337305684, 0.3937630978073883],
    [0.27534327984736173, 0.3613587313502864],
    [0.30636357949391646, 0.3249820149510496],
    [0.3528119233054956, 0.2765631353414033],
    [0.4049842044352696, 0.22823809076976032],
    [0.45334052735291375, 0.17985048950611504],
    [0.5054580713771856, 0.1356229082606113],
    [0.5459635294485629, 0.10350004691751913],
    [0.5805808388852397, 0.08357574051484185],
    [0.563025867192143, 0.11197647868380756],
    [0.533968283757155, 0.14428701010290612],
    [0.49918676300397247, 0.17650370648400138],
    [0.4547558725094618, 0.21675893778737],
    [0.4160489193331458, 0.2571080041287419],
    [0.37548872415626666, 0.29332832879797355],
    [0.33678177097995066, 0.33367739513934525],
    [0.29818429201463825, 0.36583153482843844]
])

x, y = data[:,0], data[:,1]

# --------------------------------------------------
# 2. CALIBRATION TO PHYSICAL AXES
# --------------------------------------------------
# Use the Planck figure edges (H0 vs OmegaM)
H0_min, H0_max = 66.0, 68.9
OmegaM_min, OmegaM_max = 0.3035, 0.3422

# Correct linear scaling
H0_phys = H0_min + (x - x.min()) * (H0_max - H0_min) / (x.max() - x.min())
OmegaM_phys = OmegaM_min + (y - y.min()) * (OmegaM_max - OmegaM_min) / (y.max() - y.min())

data_phys = np.column_stack([H0_phys, OmegaM_phys])

# --------------------------------------------------
# 3. FIT ELLIPSE IN PHYSICAL UNITS
# --------------------------------------------------
X = data_phys[:,0]
Y = data_phys[:,1]

D = np.vstack([X*X, X*Y, Y*Y, X, Y, np.ones_like(X)]).T
S = np.dot(D.T, D)

C = np.zeros([6,6])
C[0,2] = C[2,0] = 2
C[1,1] = -1

eigvals, eigvecs = eig(np.dot(inv(S), C))
cond = 4*eigvecs[0,:]*eigvecs[2,:] - eigvecs[1,:]**2
a = eigvecs[:, cond > 0][:,0]
A,B,Cc,Dc,Ec,F = a

b = np.array([[A, B/2],
              [B/2, Cc]])
d = np.array([Dc, Ec])/2
center = -np.dot(inv(b), d)

F_center = (F + Dc*center[0] + Ec*center[1] + A*center[0]**2 + B*center[0]*center[1] + Cc*center[1]**2)
Q = b / (-F_center)

delta_chi2_95 = 5.991
Sigma = inv(Q) / delta_chi2_95  # covariance in H0 vs OmegaM

# --------------------------------------------------
# 4. CONVERT TO H0 vs OmegaL
# --------------------------------------------------
J = np.array([[1,0],
              [0,-1]])
Sigma_lambda = J @ Sigma @ J.T
H0_mean = center[0]
OmegaM_mean = center[1]
OmegaL_mean = 1.0 - OmegaM_mean
cmb_center = np.array([H0_mean, OmegaL_mean])
cmb_cov = Sigma_lambda
rho = Sigma_lambda[0,1] / np.sqrt(Sigma_lambda[0,0]*Sigma_lambda[1,1])

# --------------------------------------------------
# 5. OPTIONAL: WIDEN PRIOR
# --------------------------------------------------
scale = 5.0
sigma_H0 = np.sqrt(Sigma_lambda[0,0]) * scale
sigma_OmegaL = np.sqrt(Sigma_lambda[1,1]) * scale
Sigma_lambda_scaled = np.array([
    [sigma_H0**2, rho*sigma_H0*sigma_OmegaL],
    [rho*sigma_H0*sigma_OmegaL, sigma_OmegaL**2]
])

# Final directional prior
cmb_cov = Sigma_lambda_scaled
cmb_icov = np.linalg.inv(cmb_cov)
cmb_rho = rho

# --------------------------------------------------
# 6. RESULTS
# --------------------------------------------------
print("Center H0, OmegaL:", cmb_center)
print("Covariance matrix:\n", cmb_cov)
print("Correlation coefficient:", cmb_rho)






# --------------------------------------------------
# 5. Plot check
# --------------------------------------------------

vals, vecs = eig(Sigma)
angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))

width = 2*np.sqrt(vals[0]*delta_chi2_95)
height = 2*np.sqrt(vals[1]*delta_chi2_95)

from matplotlib.patches import Ellipse

fig, ax = plt.subplots()
ax.scatter(x,y,label="95% contour points")

ellipse = Ellipse(center, width, height, angle=angle,
                  edgecolor='red', facecolor='none')
ax.add_patch(ellipse)

ax.set_aspect('equal')
plt.legend()
#plt.show()
