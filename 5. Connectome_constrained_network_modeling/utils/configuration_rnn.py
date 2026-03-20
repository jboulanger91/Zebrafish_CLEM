import numpy as np


class ConfigurationRNN:
    dt_simulation = 0.01
    dt_data = 0.5

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
    side_list = ["L", "R"]

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
