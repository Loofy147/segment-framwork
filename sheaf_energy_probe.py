"""
Minimal probe into an explicitly-flagged open question (arXiv 2601.21207, May 2026):
"the distribution and diffusion behavior of GDL and TDL features during training
remains an open and underexplored problem."

Narrowed, testable sub-question:
  In a cellular sheaf diffusion network trained on a heterophilic node-classification
  task, does the trajectory of feature "roughness" (Dirichlet energy under the learned
  sheaf Laplacian) show a smooth/monotonic relationship with accuracy, or a sharp
  transition -- and does it track the operator's algebraic connectivity (spectral gap)?

This is a single run, not a swept/replicated study. It's meant to establish whether
there's a signal worth a real sweep, not to settle the question.
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad

rng = np.random.RandomState(0)

# ---------- synthetic heterophilic graph ----------
n = 24
d = 4
labels = np.array([0]*12 + [1]*12)

edges = []
for i in range(n):
    for j in range(i+1, n):
        same = labels[i] == labels[j]
        p = 0.05 if same else 0.35   # heterophilic: cross-community edges favored
        if rng.rand() < p:
            edges.append((i, j))
E = len(edges)
print(f"n={n} d={d} edges={E} avg_deg={2*E/n:.2f}")

# constant selection matrices P_u: (n*d, d)
P = []
for u in range(n):
    Pu = np.zeros((n*d, d))
    Pu[u*d:(u+1)*d, :] = np.eye(d)
    P.append(Pu)

def build_L(R):
    # R: (E, d, d) restriction maps F_{e,v}; F_{e,u} fixed = I
    L = anp.zeros((n*d, n*d))
    for k, (u, v) in enumerate(edges):
        Re = R[k]
        b_uu = anp.eye(d)
        b_vv = Re.T @ Re
        b_uv = -Re
        b_vu = -Re.T
        Pu, Pv = P[u], P[v]
        L = L + Pu @ b_uu @ Pu.T + Pv @ b_vv @ Pv.T + Pu @ b_uv @ Pv.T + Pv @ b_vu @ Pu.T
    return L

def unpack(theta):
    i = 0
    R = theta[i:i+E*d*d].reshape(E, d, d); i += E*d*d
    Emb = theta[i:i+n*d].reshape(n, d); i += n*d
    W1 = theta[i:i+d*d].reshape(d, d); i += d*d
    W2 = theta[i:i+d*d].reshape(d, d); i += d*d
    W3 = theta[i:i+d*d].reshape(d, d); i += d*d
    Wout = theta[i:i+d*2].reshape(d, 2); i += d*2
    return R, Emb, W1, W2, W3, Wout

def forward(theta, alpha=0.3):
    R, Emb, W1, W2, W3, Wout = unpack(theta)
    L = build_L(R)
    H = Emb
    for W in (W1, W2, W3):
        HW = (H @ W).flatten()
        LHW = (L @ HW).reshape(n, d)
        H = anp.tanh(H - alpha * LHW)
    logits = H @ Wout
    return logits, H, L

def loss_fn(theta):
    logits, H, L = forward(theta)
    logp = logits - anp.max(logits, axis=1, keepdims=True)
    logp = logp - anp.log(anp.sum(anp.exp(logp), axis=1, keepdims=True))
    nll = -anp.mean(logp[anp.arange(n), labels])
    return nll

gloss = grad(loss_fn)

# init params
theta_len = E*d*d + n*d + 3*d*d + d*2
theta = rng.randn(theta_len) * 0.3

m = np.zeros_like(theta); v = np.zeros_like(theta)
lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8

log = []
for step in range(1, 401):
    g = gloss(theta)
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*(g*g)
    mhat = m/(1-b1**step); vhat = v/(1-b2**step)
    theta = theta - lr*mhat/(np.sqrt(vhat)+eps)

    if step % 5 == 0 or step == 1:
        logits, H, L = forward(theta)
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == labels)
        Hf = np.array(H); Lf = np.array(L)
        dirichlet = float(Hf.flatten() @ Lf @ Hf.flatten()) / n
        eigs = np.linalg.eigvalsh((Lf+Lf.T)/2)
        alg_conn = float(eigs[1])  # 2nd smallest (1st ~ 0 only if exact kernel; sheaf Laplacians often have no trivial kernel)
        s = np.linalg.svd(Hf, compute_uv=False)
        eff_rank = float((s.sum()**2) / (s**2).sum())
        loss_v = float(loss_fn(theta))
        log.append((step, loss_v, acc, dirichlet, alg_conn, eff_rank))

print(f"{'step':>5} {'loss':>8} {'acc':>6} {'dirichlet':>10} {'alg_conn':>9} {'eff_rank':>9}")
for row in log[::4]:
    print(f"{row[0]:>5} {row[1]:>8.4f} {row[2]:>6.2f} {row[3]:>10.4f} {row[4]:>9.4f} {row[5]:>9.4f}")
print("...")
for row in log[-5:]:
    print(f"{row[0]:>5} {row[1]:>8.4f} {row[2]:>6.2f} {row[3]:>10.4f} {row[4]:>9.4f} {row[5]:>9.4f}")

import os
try:
    os.makedirs('/home/claude/sheaf_probe', exist_ok=True)
    np.save('/home/claude/sheaf_probe/log.npy', np.array(log))
except Exception as e:
    print(f"Warning: Could not save to /home/claude/sheaf_probe/log.npy: {e}")
    print("Saving to local directory log.npy instead.")
    np.save('log.npy', np.array(log))
