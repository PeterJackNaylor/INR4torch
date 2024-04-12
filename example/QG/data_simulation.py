import numpy as np
import matplotlib.pyplot as plt

import pyqg

from utils import CheckOrCreate

def slice_satelite_like(
    cube,
    L,
    tmax,
    dt,
    swath_thick=2 * np.pi,
    direct=1,
    xy_res=100,
    swath_res=45,
    shift=0.5,
):
    all_coordinates = []
    cubes = []
    x_shift_ = shift

    pos_0 = np.array([-x_shift_, 0, 0])
    pos_1 = np.array([direct * x_shift_, L, 0])
    pos_swoth = np.array([swath_thick, 0, 0])

    idx = 0
    xy_res = list(np.linspace(0, 1, xy_res))
    swath_res = list(np.linspace(0, 1, swath_res))

    while pos_0[2] < tmax - dt:

        pos_0[0] = pos_0[0] - direct * x_shift_
        pos_0[0] = pos_0[0] % L
        tmp_coords = np.array(
            [pos_0 + u * pos_1 + v * pos_swoth for u in xy_res for v in swath_res]
        )
        tmp_coords[:, 0] = tmp_coords[:, 0] % L
        all_coordinates.append(tmp_coords)
        idx += 1
        pos_0[2] += dt

        cubes.append(
            cube[
                (np.round(tmp_coords[:, 1] / L * 255, 0)).astype(int),
                (np.round(tmp_coords[:, 0] / L * 255, 0)).astype(int),
                idx,
            ]
        )

    coords = np.concatenate(all_coordinates, axis=0)

    z = np.concatenate(cubes, axis=0)
    xytz = np.column_stack([coords, z])
    return xytz


# define a quick function for plotting and visualize the initial condition


def plot_q(m, sim, qmax=40, time=0, plot=True, interval=0.1, L=np.pi):
    time_f = round(time, 1)
    time_int = int(round(time_f / interval, 0))
    CheckOrCreate("pyqg_out/plots")
    CheckOrCreate("pyqg_out/plots/simulation")
    
    sim[:, :, time_int] = m.q[0]
    if plot:
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(m.x, m.y, m.q.squeeze(), cmap="RdBu_r")
        pc.set_clim([-qmax, qmax])
        ax.set_xlim([0, L])
        ax.set_ylim([0, L])
        ax.set_aspect(1)
        plt.colorbar(pc)
        plt.title("Time = %g" % m.t)
        plt.savefig(f"pyqg_out/plots/simulation/simulation_t_{time_f}.png")
        plt.close()


def laplacian(p):
    q = np.zeros_like(p)
    q[1:-1, 1:-1] = (
        p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    )

    return q


def plot_psi(m, sim, psimax=1, time=0, plot=True, interval=0.1, L=np.pi):
    time_f = round(time, 1)
    time_int = int(round(time_f / interval, 0))
    CheckOrCreate("pyqg_out/plots")
    CheckOrCreate("pyqg_out/plots/psi_simulation")
    CheckOrCreate("pyqg_out/plots/psi_simulation_laplacian")

    qi = m.q
    qih = m.fft(qi)
    m.wv2[0, 0] = 1
    pih = -qih / m.wv2
    pi = m.ifft(pih)
    sim[:, :, time_int] = pi[0]

    if plot:
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(m.x, m.y, sim[:, :, time_int], cmap="RdBu_r")
        pc.set_clim([-psimax, psimax])
        ax.set_xlim([0, L])
        ax.set_ylim([0, L])
        ax.set_aspect(1)
        plt.colorbar(pc)
        plt.title("Time = %g" % m.t)
        plt.savefig(f"pyqg_out/plots/psi_simulation/simulation_t_{time_f}.png")
        plt.close()

    if plot:
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(m.x, m.y, laplacian(pi[0]), cmap="RdBu_r")
        # pc.set_clim([-psimax, psimax])
        ax.set_xlim([0, L])
        ax.set_ylim([0, L])
        ax.set_aspect(1)
        plt.colorbar(pc)
        plt.title("Time = %g" % m.t)
        plt.savefig(
            f"pyqg_out/plots/psi_simulation_laplacian/simulation_t_{time_f}.png"
        )
        plt.close()


def main():
    CheckOrCreate("pyqg_out")
    tmax = 6  # 40
    interval = 0.1
    L = 2.0 * np.pi
    L = np.pi
    xy_res = 100
    swath_res = 100
    shift = 0.5
    swath_thick = L / 10
    # create the model object
    m = pyqg.BTModel(
        L=L,
        nx=256,
        beta=0.0,
        H=1.0,
        rek=0.0,
        rd=None,
        tmax=tmax,
        dt=0.001,
        taveint=1,
        ntd=8,
    )
    # in this example we used ntd=4, four threads
    # if your machine has more (or fewer) cores available, you could try changing it

    # generate McWilliams 84 IC condition

    fk = m.wv != 0
    ckappa = np.zeros_like(m.wv2)
    ckappa[fk] = np.sqrt(m.wv2[fk] * (1.0 + (m.wv2[fk] / 36.0) ** 2)) ** -1

    nhx, nhy = m.wv2.shape

    Pi_hat = (
        np.random.randn(nhx, nhy) * ckappa + 1j * np.random.randn(nhx, nhy) * ckappa
    )

    Pi = m.ifft(Pi_hat[np.newaxis, :, :])
    Pi = Pi - Pi.mean()
    Pi_hat = m.fft(Pi)
    KEaux = m.spec_var(m.wv * Pi_hat)

    pih = Pi_hat / np.sqrt(KEaux)
    qih = -m.wv2 * pih
    qi = m.ifft(qih)

    # initialize the model with that initial condition
    m.set_q(qi)
    simulation = np.zeros((m.x.shape[0], m.x.shape[1], int(tmax / interval) + 1))
    psi_simulation = np.zeros((m.x.shape[0], m.x.shape[1], int(tmax / interval) + 1))

    plot_q(m, simulation, time=0)
    plot_psi(m, psi_simulation, time=0)
    times = [0]
    for _ in m.run_with_snapshots(tsnapstart=0, tsnapint=interval):
        print(_)
        plot_psi(m, psi_simulation, time=_, plot=True, interval=interval, L=L)
        plot_q(m, simulation, time=_, plot=True, interval=interval, L=L)
        times.append(round(_, 1))

    xytz_pos = slice_satelite_like(
        psi_simulation,
        L,
        tmax,
        interval,
        swath_thick=swath_thick,
        direct=1,
        xy_res=xy_res,
        swath_res=swath_res,
        shift=shift,
    )
    xytz_neg = slice_satelite_like(
        psi_simulation,
        L,
        tmax,
        interval,
        swath_thick=swath_thick,
        direct=-1,
        xy_res=swath_res,
        swath_res=swath_res,
        shift=shift,
    )
    xytz = np.concatenate([xytz_pos, xytz_neg], axis=0)

    CheckOrCreate("pyqg_out/plots/slices")

    vmin = np.round(psi_simulation.min(), 0)
    vmax = np.round(psi_simulation.max(), 0)
    for t in list(np.unique(xytz[:, 2])):

        t = round(t, 1)
        eps = interval / 2
        time_slice = (xytz[:, 2] < t + eps) & (xytz[:, 2] > t - eps)
        idx = np.where(np.array(times) == t)[0]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.imshow(
            psi_simulation[::-1, :, idx],
            extent=[0, L, 0, L],
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax2.imshow(
            psi_simulation[::-1, :, idx],
            extent=[0, L, 0, L],
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax2.scatter(
            xytz[time_slice, 0],
            xytz[time_slice, 1],
            c=xytz[time_slice, 3],
            s=2.4,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax3.scatter(
            xytz[time_slice, 0],
            xytz[time_slice, 1],
            c=xytz[time_slice, 3],
            s=2.4,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax3.set_xlim([0, L])
        ax3.set_ylim([0, L])
        plt.savefig(f"pyqg_out/plots/slices/slice_{t}.png")
        plt.close()

    folder = "pyqg_out"
    np.save(f"{folder}/x_values.npy", m.x[0, :])
    np.save(f"{folder}/y_values.npy", m.y[:, 0])
    np.save(f"{folder}/t_values.npy", times)
    np.save(f"{folder}/q_values.npy", simulation)
    np.save(f"{folder}/psi_values.npy", psi_simulation)

    np.save(f"{folder}/qg_data.npy", xytz)


if __name__ == "__main__":
    main()
