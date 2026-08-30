import numpy as np


class ConfigurationRNN:
    dt_simulation = 0.01
    dt_data = 0.5
    tau_neuron = 0.1
    clamp_weights_min = 1e-2

    cell_label_list = [ {"label": "iMI",
                         "index4": 0,
                         "index3": 0},
                        {"label": "cMI",
                         "index4": 1,
                         "index3": 0},
                        {"label": "MON",
                         "index4": 2,
                         "index3": 1},
                        {"label": "sMI",
                         "index4": 3,
                         "index3": 2},]
    SIDE_LEFT  = "L"
    SIDE_RIGHT = "R"
    side_list = [SIDE_LEFT, SIDE_RIGHT]
    cell_list = ["iMI", "cMI", "MON", "sMI"]


    time_structure_simulation_train = {"rest_start": 20,
                                 "stimulus":   40,
                                 "rest_end":   20}
    time_structure_simulation_test = {"rest_start": 16,
                                 "stimulus": 32,
                                 "rest_end": 16}

    time_structure_simulation_train["time_list"] = [ts for ts in time_structure_simulation_train.values()]
    time_structure_simulation_test["time_list"]  = [ts for ts in time_structure_simulation_test.values()]

    time_structure_simulation_train["duration"] = np.sum(time_structure_simulation_train["time_list"])
    time_structure_simulation_test["duration"] = np.sum(time_structure_simulation_test["time_list"])

    classifier_to_pop_map = {"motion_integrator": {"ipsilateral": "iMI",
                                                   "contralateral": "cMI"},
                             "motion_onset": {"ipsilateral": "MON",
                                              "contralateral": "MON"},
                             "slow_motion_integrator": {"ipsilateral": "sMI",
                                                        "contralateral": "sMI"},
                             "myelinated": {"ipsilateral": "myelinated",
                                            "contralateral": "myelinated"}}


class ConfigurationNeural():
    N_POPS = 8
    POP_META = {
        'name': ['L_iMI', 'L_cMI', 'L_MON', 'L_sMI',
                 'R_iMI', 'R_cMI', 'R_MON', 'R_sMI', ],
        'hemisphere': ['L', 'L', 'L', 'L', 'R', 'R', 'R', 'R'],
    }

    # Fixed anatomical parameters (known from data)
    F_I_FIXED = np.array([0.3, 0.9, 0.9, 2 / 3, 0.3, 0.9, 0.9, 2 / 3])
    N_REL_FIXED = np.array([1.00, 0.80, 0.50, 0.65, 1.00, 0.80, 0.50, 0.65])
    N_CELLS = [15, 15, 2, 11, 15, 15, 2, 11]  # recorded cells per population
    N_CELLS_FREE = 16
    P = np.array([
        [0.1125, 0.18, 0.475,   0.065, 0, 0, 0, 0, ],
        [0, 0, 0, 0,         0.2, 0.3, 0.04, 0.06, ],
        [0.08, 0.45, 0.09, 0.04, 0.03, 0.15, 0, 0, ],
        [0.02, 0, 0, 0,       0.05, 0.05, 0.05, 0, ],

        [0, 0, 0, 0,     0.1125, 0.18, 0.475, 0.065],
        [0.2, 0.3, 0.04, 0.06,         0, 0, 0, 0, ],
        [0.03, 0.15, 0, 0, 0.08, 0.45, 0.09, 0.04, ],
        [0.05, 0.05, 0.05, 0,       0.02, 0, 0, 0, ]
    ]).T  # reported as in the Fig. 5b and then transposed

