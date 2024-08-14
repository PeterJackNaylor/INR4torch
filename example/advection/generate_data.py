import numpy as np


def get_dataset(T=1.0, L=2 * np.pi, c=80, n_t=200, n_x=128):
    t_star = np.linspace(0, T, n_t)
    x_star = np.linspace(0, L, n_x)

    tv, xv = np.meshgrid(t_star, x_star)
    xv = np.flip(xv, axis=0)
    u_exact = np.sin(np.mod(xv - c * tv, L))
    return u_exact, t_star, x_star


if __name__ == "__main__":
    u, t, x = get_dataset()
    np.savez("adv.npz", u=u, t=t, x=x)
