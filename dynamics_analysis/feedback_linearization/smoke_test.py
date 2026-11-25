"""Sanity checks on the model and the constraint-consistent decoupling matrix."""
import time
import numpy as np
import model as M
import controller as C
import ic as IC

np.set_printoptions(precision=4, suppress=True, linewidth=130)

q = IC.nominal_config()
q, res = IC.project_config(q)
print(f"constraint projection residual ||kappa|| = {res:.2e}")

D, G = M.inertia_and_gravity(q)
w = np.linalg.eigvalsh(D)
print(f"D symmetric? {np.allclose(D, D.T)}   min/max eig = {w.min():.3e} / {w.max():.3e}  (SPD={w.min()>0})")

# end-effector / contact frame agreement
for i in (0, 1):
    kin = M.robot_kinematics(q[M.ROBOTS[i]['all']])
    pcf, Rcf = M.contact_frame(q, i)
    print(f"robot{i+1}: ||p_ee - p_cf|| = {np.linalg.norm(kin['ee_pos']-pcf):.2e}")

# trajectory + controller at t=0
p0 = q[M.PAY['pos']].copy()
traj = C.make_trajectory(p0, p0 + np.array([0.4, 0.0, 0.15]))

qd = np.zeros(M.NQ)
t0 = time.time()
u, alpha0, Mu, info, diag = C.fbl_control(q, qd, traj, 0.0, Kp=25*np.eye(6), Kd=10*np.eye(6))
dt_eval = time.time() - t0
print(f"\none fbl_control eval: {dt_eval*1000:.1f} ms")
print(f"||naive Jy D^-1 B|| = {np.linalg.norm(M.output_jacobian(q,traj,0.0) @ np.linalg.solve(D, M.B)):.2e}  (report's G, ~0 as predicted)")
Ghat = diag['Ghat']
print(f"Ghat (constraint-consistent) shape {Ghat.shape}, rank {np.linalg.matrix_rank(Ghat, tol=1e-6)}, sigma_min={diag['sigma_min']:.3e}")
print("singular values of Ghat:", np.linalg.svd(Ghat, compute_uv=False))

# verify feedback linearization: realized yddot should equal v
qdd = C.closed_loop_accel(q, qd, u, alpha0, Mu)
drift_y, Jy, ydot = C.output_drift(q, qd, traj, 0.0)
yddot = drift_y + Jy @ qdd
print(f"\nyddot - v residual = {np.linalg.norm(yddot - diag['v']):.2e}  (should be ~0)")
print(f"||u|| = {np.linalg.norm(u):.3f}, max|F0| = {np.max(np.abs(diag['F0'])):.3f}")
