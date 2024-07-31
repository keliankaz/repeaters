# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Callable, Union


class CoupledOscillator:
    """Bare bones implementation of a coupled oscillator"""

    def __init__(
        self,
        phase_init: np.ndarray,
        natural_frequency: np.ndarray,
        time_init: float = 0,
    ):
        self.phase_init = phase_init
        self.time_init = time_init

        self.natural_frequency = natural_frequency

        self.number_of_oscillators = len(phase_init)

        # state variables:
        self.phase = phase_init
        self.time = time_init

        self.record = []

        self.jittered_radius_for_plotting = None

        self._validate_arg()

    def get_state(self):
        return {
            "phase": self.phase.copy(),
            "time": self.time,
        }

    def _validate_arg(self):
        assert self.number_of_oscillators == len(self.phase)
        assert self.number_of_oscillators == len(self.natural_frequency)

    def coupling(self, i, j):
        """Define coupling between nodes i and j

        j: stimulus
        i: oscilator

        """
        return NotImplementedError

    def step(self, dt):
        rate = np.zeros_like(self.phase)
        for i in range(self.number_of_oscillators):
            rate[i] = self.natural_frequency[i] + np.sum(
                [
                    self.coupling(i, j)
                    for j in range(self.number_of_oscillators)
                    if j != i
                ]
            )

        self.phase += rate * dt

    def log(self):
        self.record.append(self.get_state())

    def plot_state(self, state=None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        if state is None:
            state = self.get_state()

        return NotImplementedError

    def plot_phase(self, phase: np.ndarray, jitter: float = 0.1, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        if self.jittered_radius_for_plotting is None:
            self.jittered_radius_for_plotting = 1 + np.random.uniform(
                -jitter / 2, jitter / 2, len(phase)
            )

        r = self.jittered_radius_for_plotting

        ax.scatter(r * np.cos(phase), r * np.sin(phase))
        ax.set(
            xlim=[-(1 + jitter), (1 + jitter)],
            ylim=[-(1 + jitter), (1 + jitter)],
            aspect="equal",
        )

        return ax

    def make_gif(self, file_name: str, fps=30) -> None:

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            return self.plot_state(self.record[i], ax=ax)

        print("making animation...")
        ani = animation.FuncAnimation(
            fig, animate, repeat=True, frames=len(self.record) - 1, interval=1
        )

        writer = animation.PillowWriter(fps=fps)

        print("Saving animation...")
        ani.save(file_name, writer=writer)


class Kuramoto(CoupledOscillator):
    def __init__(
        self,
        N: int = 10,
        coupling_strenght: float = 1,
        g: Optional[Callable[[int], np.ndarray]] = lambda n: np.random.uniform(
            0.1, 3, n
        ),
        natural_frequency: Optional[np.ndarray] = None,
        phase: Optional[np.ndarray] = None,
    ):

        if g is not None:
            self.g = g

        if phase is None:
            phase = np.random.uniform(0, 2 * np.pi, N)

        if natural_frequency is None:
            natural_frequency = self.g(N)

        self.coupling_strenght = coupling_strenght

        super().__init__(phase, natural_frequency)

    def coupling(self, i, j):
        return (self.coupling_strenght / self.number_of_oscillators) * np.sin(
            self.phase[j] - self.phase[i]
        )

    def plot_state(self, state=None, ax=None):

        if state is None:
            phase = self.phase
        else:
            phase = state["phase"]

        return self.plot_phase(phase, ax=ax)


class PulsedCoupledOscillator(CoupledOscillator):
    def __init__(
        self,
        pulse_discharge: Union[float, np.ndarray],
        charge_rate: Union[float, np.ndarray],
        leakage_rate: Union[float, np.ndarray],
        pulse_threshold:  Union[float, np.ndarray],
        **coupledOscillator_kwargs,
    ):

        super().__init__(**coupledOscillator_kwargs)

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

    def step(self, dt):
        rate = np.zeros_like(self.phase)
        for i in range(self.number_of_oscillators):
            rate[i] += self.natural_frequency[i]

        self.phase += rate * dt


class ReapeterNetwork(CoupledOscillator):
    """Coupled set of earthquake repeaters.

    The repeater network is similar to the Kuramoto model with some key differences:

    - coupling is variable and depends on proximity and size of the events
    - the recurrence interval/natural frequency of events is dependent on size
    - the phase dependence of the coupling need not be sinusoidal
    - the model can be noised


    """

    def __init__(
        self,
        size: np.ndarray,
        location: np.ndarray,
        size_to_slip: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        slip_rate: float = 1,
        phase_init: np.ndarray = np.random.uniform(0, 2 * np.pi),
    ):

        # repeater properties
        self.size = size
        self.location = location
        self.slip_rate = slip_rate

        if size_to_slip is None:
            self.size_to_slip = lambda L: 1e-3 * L

        phase = phase_init
        natural_frequency = slip_rate / (self.size_to_slip(self.size))

        super().__init__(phase, natural_frequency)

    def coupling(self, i, j):
        """j -> i"""

        return (
            self.size[j]
            / np.sqrt(
                ((self.location[i, :] - self.location[j, :]) ** 2).sum()
            )  # proximity and size interaction
            * (((self.phase % (2 * np.pi)) / (np.pi)))  # sawtooth phase interaction
        )


# %%
if __name__ == "__main__":
    kuramoto = Kuramoto()

    number_of_timesteps = 10000
    dt = 0.05

    for timestep in range(number_of_timesteps):
        kuramoto.step(dt)
        kuramoto.log()

    kuramoto.plot_state()

    print("making gif...")
    kuramoto.make_gif("kuramoto_example.gif")
# %%
