#%%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Union, Literal
from scipy.spatial import distance_matrix


class Stepper:
    """General blueprint for coupled oscillators and dynamical system."""

    def __init__(self):
        self.record = []

    def step(self, dt):
        return NotImplementedError

    def get_state(self):
        return NotImplementedError

    def plot_state(self, state: dict = None, ax=None):

        if ax is None:
            _, ax = plt.subplots()

        if state is None:
            state = self.get_state()

        return NotImplementedError

    def log(self):
        self.record.append(self.get_state())

    def make_gif(self, file_name: str, fps=30) -> None:

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            return self.plot_state(self.record[i], ax=ax)

        print("making animation...")
        ani = animation.FuncAnimation(
            fig,
            animate,
            repeat=True,
            frames=len(self.record) - 1,
            interval=1,
        )

        writer = animation.PillowWriter(fps=fps)

        print("Saving animation...")
        ani.save(file_name, writer=writer)

    @staticmethod
    def runge_kutta_step(xn, f, dt):

        k1 = f(xn) * dt
        k2 = f(xn + 0.5 * k1) * dt
        k3 = f(xn + 0.5 * k2) * dt
        k4 = f(xn + k3) * dt

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6


class PulsedCoupledOscillator(Stepper):
    def __init__(
        self,
        number_of_oscillators: Optional[int] = 10,
        pulse_discharge: Union[float, np.ndarray] = 0.01,
        charge_rate: Union[float, np.ndarray] = 0.01,
        leakage_rate: Union[float, np.ndarray] = 0.007,
        pulse_threshold: Union[float, np.ndarray] = 1,
        charge_init: Optional[np.ndarray] = None,
        coupling: Optional[Union[float, int, np.ndarray]] = None,
    ):

        self.number_of_oscillators = number_of_oscillators

        self.pulse_discharge = (
            pulse_discharge * np.ones(self.number_of_oscillators)
            if isinstance(pulse_discharge, float)
            else pulse_discharge
        )
        self.charge_rate = (
            charge_rate * np.ones(self.number_of_oscillators)
            if isinstance(charge_rate, float)
            else charge_rate
        )
        self.leakage_rate = (
            leakage_rate * np.ones(self.number_of_oscillators)
            if isinstance(leakage_rate, float)
            else leakage_rate
        )
        self.pulse_threshold = (
            pulse_threshold * np.ones(self.number_of_oscillators)
            if isinstance(pulse_threshold, float)
            else pulse_threshold
        )

        self.natural_time = (
            -np.log(1 - self.leakage_rate * self.pulse_threshold / self.charge_rate)
            / self.leakage_rate
        )

        self.natural_frequency = 2 * np.pi / self.natural_time

        if charge_init is None:
            self.charge = (
                np.random.uniform(0, 1, number_of_oscillators) * self.pulse_threshold
            )

        if coupling is None:  # note that coupling could (should?) be phase dependent
            self._coupling = np.ones(
                (number_of_oscillators, number_of_oscillators)
            ) - np.eye(number_of_oscillators, number_of_oscillators)
        elif isinstance(coupling, float) or isinstance(coupling, int):
            self._coupling = coupling * np.ones(
                (number_of_oscillators, number_of_oscillators)
            )
        elif isinstance(coupling, np.ndarray):
            self._coupling = coupling
            assert self.coupling.shape == (number_of_oscillators, number_of_oscillators)
        else:
            raise ValueError("unrecognized type for `coupling`")
        
        
        self._phase = None

        self.jittered_radius_for_plotting = None

        super().__init__()

    @property
    def phase(self):
        self._phase = (self.charge / self.pulse_threshold) * 2 * np.pi
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        self.charge = self._phase / (2 * np.pi) * self.pulse_threshold

    @property
    def coupling(self):
        return (
            self._coupling * np.abs(np.atleast_2d(self.phase) - np.atleast_2d(self.phase).T)/(2*np.pi)
        ) # to emphasize potential phase dependence

    @property
    def dimensionless_coupling(self):

        out = (
            self.natural_frequency
            * self.coupling
            / self.pulse_threshold
            / np.abs(
                np.row_stack(self.natural_frequency)
                - np.column_stack(self.natural_frequency)
            )
        )

        np.fill_diagonal(out, np.NaN)  # in place

        return out

    @property
    def natural_dt(self):
        """Rough guess at an appropriate dt - kind of a hack."""
        return np.min(self.natural_time) / 300  # need to resolve the orbits of the smallest events

    def coherence(self, band=None):

        if band is not None:
            selection = (self.natural_frequency < band[1]) & (
                self.natural_frequency > band[0]
            )

        else:
            selection = np.ones_like(self.natural_frequency, dtype=bool)

        return np.abs(np.sum(np.exp(self.phase[selection] * 1j))) / sum(selection)

    def step_charge(self, dt):
        
        def f(x):
            return self.charge_rate - self.leakage_rate * x / self.pulse_threshold
        
        self.charge += self.runge_kutta_step(self.charge, f, dt)

    def discharge(self):
        
        pulse = self.charge > self.pulse_threshold
        
        count_cascade = 0
        
        while np.any(pulse): 
            self.charge[pulse] = 0
            self.charge += (
                self.coupling @ np.vstack(pulse)
            ).squeeze() * self.pulse_discharge
            
            pulse = self.charge > self.pulse_threshold
            
            count_cascade += 1
                
            if count_cascade > 100:
                print('Exiting casade of events, consider making time step shorter')
                break 

    def step(self, dt):
        self.step_charge(dt)
        self.discharge()

    def get_state(self):
        return dict(
            phase=self.phase,
            charge=self.charge,
        )

    def plot_phase(
        self, phase: np.ndarray, jitter: float = 0.2, scatter_kwargs=None, ax=None
    ):
        if ax is None:
            _, ax = plt.subplots()

        if self.jittered_radius_for_plotting is None:
            self.jittered_radius_for_plotting = 1 + np.random.normal(
                0, jitter / 2, len(phase)
            )

        r = self.jittered_radius_for_plotting

        default_scatter_dict = dict(
            c=phase,
            cmap="magma",
            vmin=0,
            vmax=2 * np.pi,
        )
        if scatter_kwargs is not None:
            default_scatter_dict.update(scatter_kwargs)

        ax.scatter(
            r * np.cos(phase),
            r * np.sin(phase),
            **default_scatter_dict,
        )
        ax.plot([0, 1], [0, 0], c="r")
        ax.scatter(0, 0, c="r", s=10)
        ax.set(
            xlim=[-(1 + 2 * jitter), (1 + 2 * jitter)],
            ylim=[-(1 + 2 * jitter), (1 + 2 * jitter)],
            aspect="equal",
        )

        ax.axis("off")

        return ax

    def plot_state(self, state: dict = None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        if state is None:
            state = self.get_state()

        ax = self.plot_phase(state["phase"], ax=ax)

        return ax


class SpatialPulsedOscillator(PulsedCoupledOscillator):
    def __init__(
        self,
        number_of_oscilators: Optional[np.ndarray] = 10,
        space: list = [
            [0, 1],
            [0, 1],
        ],
        locations: Optional[np.ndarray] = None,
        pulse_discharge: Optional[Union[float, np.ndarray]] = None,
        charge_rate: Union[float, np.ndarray] = 0.01,
        leakage_rate: Union[float, np.ndarray] = 0.005,
        pulse_threshold: Union[float, np.ndarray] = 1,
        charge_init: Optional[np.ndarray] = None,
        coupling: Optional[np.ndarray] = None,
    ):

        self.space = space

        if locations is None:
            locations = np.column_stack(
                [np.random.uniform(*limits, number_of_oscilators) for limits in space]
            )
        self.locations = locations

        if coupling is None:
            distance = distance_matrix(self.locations, self.locations)
            coupling = distance**-1
            np.fill_diagonal(
                coupling, 0
            )  # remove coupling with itself (confusing function call)

        if pulse_discharge is None:
            pulse_discharge = 1 / (number_of_oscilators - 1) / 5

        super().__init__(
            number_of_oscillators=number_of_oscilators,
            pulse_discharge=pulse_discharge,
            charge_rate=charge_rate,
            leakage_rate=leakage_rate,
            pulse_threshold=pulse_threshold,
            charge_init=charge_init,
            coupling=coupling,
        )

    # overload plotting:
    def plot_state(self, state: dict = None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        if state is None:
            state = self.get_state()

        if len(self.space) == 10:  # visualization for 2D simulations
            ax.scatter(
                self.locations[:, 0],
                self.locations[:, 1],
                s=10 + 1 / (state["phase"] + 0.1),
                c=state["phase"],
                cmap="twilight_shifted",
                vmin=0,
                vmax=2 * np.pi,
            )

        else:
            ax = self.plot_phase(state["phase"], ax=ax)

        return ax


class Repeaters(SpatialPulsedOscillator):
    def __init__(
        self,
        number_of_repeaters: Optional[np.ndarray] = 10,
        fault_dimensions: list = [
            [0, 0.01e3],
            [0, 0.01e3],
        ],
        locations: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        pulse_discharge: Optional[Union[float, np.ndarray]] = None,
        slip_rate: Union[float, np.ndarray] = 0.005,
        leakage_rate: Union[float, np.ndarray] = 0.004,
        pulse_threshold: Optional[Union[float, np.ndarray]] = None,
        initial_slip: Optional[np.ndarray] = None,
        coupling: Optional[np.ndarray] = None,
        strain_drop: float = 0.0001,
    ):

        if sizes is None:
            sizes = np.ones(number_of_repeaters)
        self.sizes = sizes
        
        self.fault_dimensions = fault_dimensions

        if pulse_discharge is None:
            pulse_discharge = self.sizes * strain_drop

        if pulse_threshold is None:
            # pulse_threshold = np.ones(number_of_repeaters)*0.5
            pulse_threshold = (
                pulse_discharge  # the threshold is correlated with the slip
            )
            
        if locations is None:
            locations = np.column_stack(
                [np.random.uniform(*limits, number_of_repeaters) for limits in fault_dimensions]
            )
        
        distance = distance_matrix(locations, locations)
            
        if coupling is None:
            coupling = ((1-self.sizes**3/distance**3)**(-1/2) - 1)
            
            coupling[distance < self.sizes] = -1 # deal with overlap
            
            np.fill_diagonal(
                coupling, 0
            )  # remove coupling with itself (confusing function call)

        super().__init__(
            number_of_oscilators=number_of_repeaters,
            space=fault_dimensions,
            locations=locations,
            pulse_discharge=pulse_discharge,
            charge_rate=slip_rate,
            leakage_rate=leakage_rate,
            pulse_threshold=pulse_threshold,
            charge_init=initial_slip,
            coupling=coupling,
        )

    def plot_plane_view(
        self,
        state,
        ax=None,
    ):

        if ax is None:
            _, ax = plt.subplots()

        if state is None:
            state = self.get_state()

        ax.set_xlim(self.fault_dimensions[0])
        ax.set_ylim(self.fault_dimensions[1])
        M = ax.transData.get_matrix()
        xscale = M[0,0]

        ax.scatter(
            self.locations[:, 0],
            self.locations[:, 1],
            s=(xscale*self.sizes)**2,
            c=state["phase"],
            cmap="inferno_r",
            vmin=0,
            vmax=0.1 * 2 * np.pi,
        )

        ax.set_facecolor("midnightblue")
        ax.set_aspect('equal')

        return ax

    # overload plotting:
    def plot_state(
        self,
        state: Optional[dict] = None,
        ax=None,
        plot_type: Literal["plane view", "phase"] = "plane view",
    ):
        if ax is None:
            _, ax = plt.subplots()

        if state is None:
            state = self.get_state()

        if plot_type == "plane view":
            ax = self.plot_plane_view(state, ax)
        elif plot_type == "phase":
            ax = self.plot_phase(
                state["phase"],
                scatter_kwargs=dict(
                    c=1 / self.sizes,
                    edgecolors="k",
                    linewidths=0.2,
                    cmap="Greys_r",
                    vmin=0,
                    vmax=np.max(1 / self.sizes) * 1.5,
                    s=self.sizes / np.max(self.sizes) * 20,
                ),
                ax=ax,
            )

        return ax


if __name__ == "__main__":

    number_of_repeaters = 100
    density = 5/1e4 # pathes/m  <- ultimately controls the coupling
    size = np.sqrt(number_of_repeaters/density)
    L, W = size, size

    pulser = Repeaters(
        number_of_repeaters=number_of_repeaters,
        sizes=np.random.lognormal(1.5, 0.01, number_of_repeaters),
        fault_dimensions=[
            [0,L],
            [0,W],
        ],
    )

    number_of_heavy_orbits = 20
    total_time = np.max(pulser.natural_time) * number_of_heavy_orbits

    dt = pulser.natural_dt

    number_of_steps = int(total_time / dt)

    print(f"Time step: {dt}")
    print(f"Number of time steps: {number_of_steps}")
    print(f"Number of time steps: {number_of_steps}")

    print("Coupling intensity")
    print(f"{np.nanmin(pulser.dimensionless_coupling)} (min)")
    print(f"{np.nanmax(pulser.dimensionless_coupling)} (max)")
    print(f"{np.nanmean(pulser.dimensionless_coupling)} (mean)")

    t = 0
    times = []
    r = []
    event_count = []

    frequency_bins = np.linspace(
        np.min(pulser.natural_frequency), np.max(pulser.natural_frequency), 5
    )
    r_binned = []

    for i in range(number_of_steps):
        pulser.step(dt)

        if not i % 3:
            pulser.log()

        t += dt
        times.append(t)
        r.append(pulser.coherence())
        r_binned.append(
            [
                pulser.coherence([f1, f2])
                for f1, f2 in zip(frequency_bins[:-1], frequency_bins[1:])
            ]
        )
        event_count.append(np.sum(pulser.phase==0))

    fig, AX = plt.subplots(
        1, 2, figsize=(6.5, 3), gridspec_kw=dict(width_ratios=(2, 1))
    )
    AX[0].plot(times / np.max(pulser.natural_time), r, linewidth=1, alpha=0.4)

    [
        AX[0].plot(
            times / np.max(pulser.natural_time),
            np.array(r_binned)[:, i],
            linewidth=1,
            alpha=0.4,
            color="grey",
        )
        for i, _ in enumerate(frequency_bins[:-1])
    ]

    AX[0].set(
        xlabel="Number heavy orbits",
        ylabel=r"Coherence $\dfrac{1}{N} |\sum e^{i\phi}|$",
        ylim=[0,1],
    )

    pulser.plot_state(ax=AX[1])
    
    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(times, event_count, lw=0.5)

    plt.rcParams["figure.dpi"] = 150
    pulser.make_gif("ride_my_repeater_cycle.gif")

# %%
