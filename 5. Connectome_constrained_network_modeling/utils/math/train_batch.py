class TrainSignal():
    def __init__(self, input_signal, output_signal, initial_value=None, label=None):
        self.input_signal = input_signal
        self.output_signal = output_signal
        self.label = label
        self.initial_value = initial_value

class TrainBatch():
    def __init__(self, train_signal_list):
        self.train_signal_list = train_signal_list
