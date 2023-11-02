import numpy as np
import pandas as pd


class TrainSlidingWindowGeneratorAttentionPoint:

    def __init__(self, appliance_threshold, predict_mode, appliance_count, file_name, raw_file_name, shuffle, window_size, batch_size, offset):
        self.__predict_mode = predict_mode
        self.__appliance_threshold = appliance_threshold
        self.__appliance_count = appliance_count
        self.__file_name = file_name
        self.__raw_file_name = raw_file_name
        self.__batch_size = batch_size
        self.__chunk_size = 10 ** 8
        self.__shuffle = shuffle
        self.__window_size = window_size
        self.__offset = offset

        self.data_array = np.array(pd.read_csv(self.__file_name, header=None))
        self.raw_data_array = np.array(pd.read_csv(self.__raw_file_name, header=None))
        self.total_size = len(self.data_array)
        self.maximum_batch_size = len(self.data_array) - window_size
        np.random.seed(120)

    def load_dataset(self):
        print("The dataset contains ", self.total_size, " rows")
        inputs, outputs = self.generate_train_data(self.data_array)
        outputs_classification = self.generate_train_data_classification(self.raw_data_array)
        indices = np.arange(self.maximum_batch_size)
        if self.__shuffle:
            np.random.shuffle(indices)
        while True:
            for start_index in range(0, self.maximum_batch_size, self.__batch_size):
                splice = indices[start_index: start_index + self.__batch_size]
                input_data = np.array([inputs[index: index + self.__window_size] for index in splice])
                input1 = input_data[:, :, [0]]
                input2 = input_data[:, :, [1, 2, 3]]
                input_all = [input1, input2]
                output_data = np.array([outputs[index + self.__offset] for index in splice])
                outputs_class = np.array([outputs_classification[index + self.__offset] for index in splice])
                yield input_all, [output_data, outputs_class]
                # yield input_data, [output_data, outputs_class]

    def generate_train_data(self, data_array):
        inputs = data_array[:, 0:4]
        # inputs = data_array[:, 0]
        inputs = np.reshape(inputs, (-1, 4))
        # inputs = np.reshape(inputs, (-1, 1))
        outputs = data_array[:, -self.__appliance_count:]
        outputs = np.reshape(outputs, (-1, self.__appliance_count))
        return inputs, outputs

    def generate_train_data_classification(self, raw_data_array):
        outputs = raw_data_array[:, -self.__appliance_count:]
        outputs[outputs <= self.__appliance_threshold] = 0
        outputs[outputs > self.__appliance_threshold] = 1
        outputs = np.reshape(outputs, (-1, self.__appliance_count))
        return outputs
