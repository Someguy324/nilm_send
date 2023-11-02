import numpy as np
import pandas as pd


class TestSlidingWindowGeneratorAttentionOrigin(object):
    def __init__(self, appliance_threshold, number_of_windows, input_window_length, predict_mode, appliance_name_list, test_directory, raw_file_name, appliance_count):
        self.__appliance_threshold = appliance_threshold
        self.__number_of_windows = number_of_windows
        self.__test_directory = test_directory
        self.__raw_file_name = raw_file_name
        self.__predict_mode = predict_mode
        self.__appliance_name_list = appliance_name_list

        self.data_array = np.array(pd.read_csv(test_directory, skiprows=0, header=None))
        self.raw_data_array = np.array(pd.read_csv(self.__raw_file_name, header=None))
        self.total_size = len(self.data_array)
        self.window_size = input_window_length
        self.max_number_of_windows = self.total_size - input_window_length
        self.__appliance_count = appliance_count

    def load_dataset(self):
        inputs, outputs = self.generate_test_data()
        outputs_classification = self.generate_test_data_classification(self.raw_data_array)
        indices = np.arange(self.max_number_of_windows, dtype=int)
        for start_index in range(0, self.max_number_of_windows, self.__number_of_windows):
            splice = indices[start_index: start_index + self.__number_of_windows]
            # 生成input
            input_data_list = []
            for index in splice:
                input_data_temp = inputs[index: index + self.window_size]
                input_data_list.append(input_data_temp)
            input_data = np.array(input_data_list)
            # 生成output
            output_data_list = []
            output_class_list = []
            for index in splice:
                output_data_temp = outputs[index: index + self.window_size]
                output_data_list.append(output_data_temp)
                output_class_temp = outputs_classification[index: index + self.window_size]
                output_class_list.append(output_class_temp)
            target_data = np.array(output_data_list)
            target_data = np.reshape(target_data, (target_data.shape[0], target_data.shape[1]))
            target_class = np.array(output_class_list)
            target_class = np.reshape(target_class, (target_class.shape[0], target_class.shape[1]))
            yield input_data, [target_data, target_class]

    def generate_test_data(self):
        data_array = self.data_array
        inputs = data_array[:, 0]
        inputs = np.reshape(inputs, (-1, 1))
        outputs = data_array[:, -self.__appliance_count:]
        outputs = np.reshape(outputs, (-1, self.__appliance_count))
        return inputs, outputs

    def generate_test_data_classification(self, raw_data_array):
        outputs = raw_data_array[:, -self.__appliance_count:]
        outputs[outputs <= self.__appliance_threshold] = 0
        outputs[outputs > self.__appliance_threshold] = 1
        outputs = np.reshape(outputs, (-1, self.__appliance_count))
        return outputs
