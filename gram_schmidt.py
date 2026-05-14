import numpy as np

def gram_schmidt(M:np.ndarray):
    """Ортогонализация столбцов с сохранением знака"""
    Q = np.zeros_like(M)
    for i in range(M.shape[1]):
        q = M[:, i].copy()
        for j in range(i):
            q = q - np.dot(Q[:, j], M[:, i]) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)
    return Q

