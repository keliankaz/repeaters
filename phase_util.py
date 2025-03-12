import numpy as np
import copy


def get_phase(tq, t: np.ndarray):
    # note: not vectorized

    if tq < min(t) or tq > max(t):
        return np.NaN

    dt = (t - tq) / np.timedelta64(1, "D")

    if np.min(np.abs(dt)) == 0:
        index = np.argmin(np.abs(dt))
        t1 = 0
        if index == 0:
            T = dt[index + 1]
        elif index == len(dt) - 1:
            T = dt[index - 1]
        else:
            T = np.mean(dt[[index - 1, index + 1]])

    else:
        dt_pos = dt[dt >= 0]
        dt_neg = dt[dt < 0]

        t1 = min(-dt_neg)
        t2 = min(dt_pos)

        T = t1 + t2

    phase = 2 * np.pi * t1 / T

    return phase

def get_order_parameter(phases):
    return np.abs(np.nanmean(np.exp(1j * phases)))

def get_phase_time_series(tq, t):
    return np.array([get_phase(itq, t) for itq in tq])


def catalog2catalog_order_parameter(cq, c):
    phases = get_phase_time_series(cq.catalog.time.values, c.catalog.time.values)
    phases = phases[~np.isnan(phases)]
    if np.any(phases):
        order_parameter = get_order_parameter(phases)
    else:
        order_parameter = np.nan

    return order_parameter


def catalog2catalog_phase(cq, c, remove_nan=True):
    phases = get_phase_time_series(cq.catalog.time.values, c.catalog.time.values)
    if remove_nan:
        phases = phases[~np.isnan(phases)]
    return phases


def normalize_order_parameter(
    order_parameter, number_of_neighboring_repeaters, number_of_trial=1000
):
    normalized_order_parameter = []
    for p, n in zip(order_parameter, number_of_neighboring_repeaters):
        if n > 0:
            normalization = np.mean(
                [
                    np.abs(np.sum(np.exp(np.random.uniform(0, 2 * np.pi, n) * 1j))) / n
                    for _ in range(number_of_trial)
                ]
            )

            normalized_order_parameter.append(p / normalization)
        else:
            normalized_order_parameter.append(0)

    return normalized_order_parameter


def phase_analysis(repeaters, earthquakes, inner_radius=0.1, outer_radius=None):

    order_parameter = []
    number_of_neighboring_repeaters = []
    number_of_neighboring_earthquakes = []
    earthquake_order_parameter = []
    raw_phases = []
    raw_times = []
    phases = []
    earthquake_phases = []
    delta_omega = []
    
    assert inner_radius > 0, "inner_radius must be greater than 0"
    assert outer_radius > inner_radius, "outer_radius must be greater than inner_radius"

    families = repeaters.get_families()

    for family in families:
        family_ID = family.catalog.family.values[0]

        earthquakes.catalog["distance"] = family.get_nearest_neighbor_distance(
            earthquakes.catalog
        )
        
        repeaters.catalog["distance"] = family.get_nearest_neighbor_distance(
            repeaters.catalog
        )

        # make a deep copy of the catalog
        neighboring_earthquakes = copy.deepcopy(earthquakes)
        neighboring_repeaters = copy.deepcopy(repeaters)

        neighboring_repeaters.catalog = neighboring_repeaters.catalog.loc[
            (neighboring_repeaters.catalog.distance > inner_radius)
            & (neighboring_repeaters.catalog.distance < outer_radius)
        ]

        neighboring_earthquakes.catalog = neighboring_earthquakes.catalog.loc[
            (neighboring_earthquakes.catalog.distance > inner_radius)
            & (neighboring_earthquakes.catalog.distance < outer_radius)
        ]

        neighboring_repeaters.catalog = neighboring_repeaters.catalog.loc[
            neighboring_repeaters.catalog.family != family_ID
        ]

        order_parameter.append(
            catalog2catalog_order_parameter(neighboring_repeaters, family)
        )

        earthquake_order_parameter.append(
            catalog2catalog_order_parameter(neighboring_earthquakes, family)
        )

        number_of_neighboring_repeaters.append(len(neighboring_repeaters))
        number_of_neighboring_earthquakes.append(len(neighboring_earthquakes))

        phases.append(catalog2catalog_phase(neighboring_repeaters, family))
        raw_phases.append(
            catalog2catalog_phase(neighboring_repeaters, family, remove_nan=False)
        )
        raw_times.append(neighboring_repeaters.catalog.time.values)
        earthquake_phases.append(catalog2catalog_phase(neighboring_earthquakes, family))

        delta_omega.append(
            (
                1 / family.catalog.RCm.values[0]
                - 1 / neighboring_repeaters.catalog.RCm.values
            )
            * family.catalog.RCm.values[0]
        )

    order_parameter = np.array(order_parameter)
    earthquake_order_parameter = np.array(earthquake_order_parameter)
    raw_phases = np.concatenate(raw_phases)
    raw_times = np.concatenate(raw_times)
    earthquake_phases = np.concatenate(earthquake_phases)

    return (
        order_parameter,
        earthquake_order_parameter,
        phases,
        raw_phases,
        raw_times,
        earthquake_phases,
        number_of_neighboring_repeaters,
        number_of_neighboring_earthquakes,
        delta_omega,
    )
