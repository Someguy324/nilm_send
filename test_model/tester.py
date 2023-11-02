import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from data_process.database.common.data_utils import get_appliance_list, get_appliance_name
from test_model.metrics import recall_precision_accuracy_f1, relative_error_total_energy, mean_absolute_error, \
    recall_precision_accuracy_f1_attention
from test_generator.attention.test_generator_attention_seq import TestSlidingWindowGeneratorAttentionSeq
from test_generator.attention.test_generator_attention_point import TestSlidingWindowGeneratorAttentionPoint
from test_generator.attention.test_generator_attention_origin import \
    TestSlidingWindowGeneratorAttentionOrigin
from test_generator.concat.test_generator_concat import TestSlidingWindowGenerator
from test_generator.common.test_generator_common import TestSlidingWindowGeneratorCommon
from train_model.network.model import load_model
from utils.common_utils import get_engine
import configparser

cf = configparser.ConfigParser()
engine = get_engine()


class Tester:
    def __init__(self, meter_name, appliance, batch_size, model_type, predict_mode, meter_name_list,
                 test_directory, raw_test_directory, saved_model_dir, log_file_dir,
                 input_window_length, appliance_count, plot_to_file, fig_length, appliance_threshold):
        self.__meter_name = meter_name
        self.__appliance = appliance
        self.__model_type = model_type
        self.__predict_mode = predict_mode
        self.__meter_name_list = meter_name_list
        self.__batch_size = batch_size
        self._input_window_length = input_window_length
        self.__window_offset = int(0.5 * (self._input_window_length + 2) - 1)
        self.__number_of_windows = batch_size
        self.__test_directory = test_directory
        self.__raw_test_directory = raw_test_directory
        self.__saved_model_dir = saved_model_dir
        self.__plot_to_file = plot_to_file
        if self.__predict_mode == 'single':
            self.__appliance_count = 1
            cf.read('temp.conf', encoding='gbk')
        elif self.__predict_mode == 'single_file':
            self.__appliance_count = 1
            cf.read('temp_file.conf', encoding='gbk')
        else:
            self.__appliance_count = appliance_count
            cf.read('temp.conf', encoding='gbk')
        self.__log_file = log_file_dir
        self.__fig_length = fig_length
        self.__threshold = appliance_threshold

    def test_model(self):
        model = load_model(self.__saved_model_dir, self.__model_type, self._input_window_length)
        test_generator, test_input, test_target = self.get_test_generator()
        steps_per_test_epoch = np.round(int(test_generator.max_number_of_windows / self.__batch_size), decimals=0)
        testing_history = model.predict(x=test_generator.load_dataset(), steps=steps_per_test_epoch, verbose=2)
        self.plot_results(testing_history, test_input, test_target)

    def plot_results(self, testing_history, test_input, test_target):
        if self.__predict_mode == 'single' or self.__predict_mode == 'single_file':
            appliance_min, appliance_max = generate_min_max(self.__meter_name, self.__appliance)
            self.process_single(testing_history, test_input, test_target, appliance_min, appliance_max)
        elif self.__predict_mode == 'multiple':
            self.process_multiple(testing_history, test_input, test_target)
        elif self.__predict_mode == 'multi_label':
            self.process_multi_label(testing_history, test_input, test_target)

    def process_single(self, testing_history, test_input, test_target, appliance_min, appliance_max):
        if self.__model_type == 'attention_seq' or self.__model_type == 'attention_origin':
            self.plot_results_single_attention(testing_history, test_input, test_target, appliance_min,
                                               appliance_max)
        elif self.__model_type == 'attention_point':
            self.plot_results_single_attention_point(testing_history, test_input, test_target, appliance_min,
                                                     appliance_max)
        else:
            self.plot_results_single(testing_history, test_input, test_target, appliance_min, appliance_max)

    def process_multiple(self, testing_history, test_input, test_target):
        count = 0
        for meter_name in self.__meter_name_list:
            appliance_id_list = get_appliance_list(meter_name, engine)
            for appliance_id in appliance_id_list:
                if appliance_id is None or '-' in appliance_id:
                    continue
                appliance_name = get_appliance_name(appliance_id, engine)
                appliance_min, appliance_max = generate_min_max(self.__meter_name, appliance_name)
                self.plot_results_multiple(testing_history[:, count:count + 1], test_input,
                                           test_target[:, count:count + 1], appliance_name, count + 1,
                                           appliance_min,
                                           appliance_max)
                count = count + 1

    def process_multi_label(self, testing_history, test_input, test_target):
        count = 0
        for appliance_name in self.__meter_name_list:
            self.plot_results_multiple_label(testing_history[:, count:count + 1], test_input,
                                             test_target[:, count:count + 1], appliance_name)
            count = count + 1

    # 处理预测类型为single， 模型类型为attention_origin或attention_seq的数据（结果结果为序列），并绘制结果
    def plot_results_single_attention(self, testing_history, test_input, test_target, appliance_min, appliance_max):
        predicted_output = testing_history[0]
        predicted_on_off = testing_history[1]
        predicted_output, test_target, test_agg = self.testing_data_process(predicted_output, test_target, test_input,
                                                                            appliance_min, appliance_max)

        predicted_on_off[predicted_on_off < 0.5] = 0
        predicted_on_off[predicted_on_off >= 0.5] = 1
        prediction = build_overall_sequence(predicted_output)
        prediction_on_off = build_overall_sequence(predicted_on_off)
        rpaf, rete, mae = self.calculate_metrics_attention(prediction, test_target, test_agg, prediction_on_off,
                                                           self.__threshold)
        self.print_metrics(self.__appliance, rpaf, rete, mae)
        appliance_name = "pic"
        self.print_plots(test_agg, test_target, prediction, 1, appliance_name)
        # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(50, 40))
        # axes[0].set_title("Real")
        # axes[0].plot(np.arange(len(test_target)), test_target, color='blue')
        # axes[1].set_title("Prediction")
        # axes[1].plot(np.arange(len(prediction)), prediction, color='orange')
        # axes[2].set_title("Real vs prediction")
        # axes[2].plot(np.arange(len(test_target)), test_target, color='blue')
        # axes[2].plot(np.arange(len(prediction)), prediction, color='orange')
        # fig.tight_layout()
        # plt.show()

    # 处理预测类型为single， 模型类型为attention_point的数据（预测结果为点），并绘制结果
    def plot_results_single_attention_point(self, testing_history, test_input, test_target, appliance_min,
                                            appliance_max):
        predicted_output = testing_history[0]
        predicted_on_off = testing_history[1]
        predicted_output, test_target, test_agg = self.testing_data_process(predicted_output, test_target, test_input,
                                                                            appliance_min, appliance_max)

        predicted_on_off[predicted_on_off < 0.5] = 0
        predicted_on_off[predicted_on_off >= 0.5] = 1
        rpaf, rete, mae = self.calculate_metrics_attention(predicted_output, test_target, test_agg, predicted_on_off,
                                                           self.__threshold)
        self.print_metrics(self.__appliance, rpaf, rete, mae)
        appliance_name = "pic"
        self.print_plots(test_agg, test_target, predicted_output, 1, appliance_name)

    # 处理预测类型为single， 模型类型采用了普通序列到点结构的数据，并绘制结果
    def plot_results_single(self, testing_history, test_input, test_target, appliance_min, appliance_max):
        testing_history, test_target, test_agg = self.testing_data_process(testing_history, test_target, test_input,
                                                                           appliance_min, appliance_max)
        rpaf, rete, mae = self.calculate_metrics(testing_history, test_agg, test_target, self.__threshold)
        self.print_metrics(self.__appliance, rpaf, rete, mae)
        appliance_name = "pic"
        self.print_plots(test_agg, test_target, testing_history, 1, appliance_name)

    # 处理预测类型为multiple， 模型类型采用了普通序列到点结构的数据，并绘制结果
    def plot_results_multiple(self, testing_history, test_input, test_target, appliance_name, count, appliance_min,
                              appliance_max):
        testing_history, test_target, test_agg = self.testing_data_process(testing_history, test_target, test_input,
                                                                           appliance_min, appliance_max)
        rpaf, rete, mae = self.calculate_metrics(testing_history, test_agg, test_target, self.__threshold)
        self.print_metrics(appliance_name, rpaf, rete, mae)
        self.print_plots(test_agg, test_target, testing_history, count, appliance_name)

    # 处理预测类型为multiple_label， 模型类型采用了普通序列到点结构的数据，并绘制结果
    def plot_results_multiple_label(self, testing_history, test_input, test_target, appliance_name):
        test_agg = test_input[:, 0].flatten()
        test_agg = test_agg[:testing_history.size]
        rpaf, rete, mae = self.calculate_metrics(testing_history, test_agg, test_target, 0.5)
        self.print_metrics(appliance_name, rpaf, rete, mae)

    # 对数据做反标准化，恢复正常数据，并对内容进行切分
    def testing_data_process(self, testing_history, test_target, test_input, appliance_min, appliance_max):
        testing_history = ((testing_history * (appliance_max - appliance_min)) + appliance_min)
        test_target = ((test_target * (appliance_max - appliance_min)) + appliance_min)
        mean = cf.getfloat(self.__meter_name + '_aggregate_' + self.__appliance, 'mean')
        std = cf.getfloat(self.__meter_name + '_aggregate_' + self.__appliance, 'std')
        test_agg = (((test_input[:, 0].flatten()) * std) + mean)

        test_agg = test_agg[:testing_history.shape[0]]
        test_target[test_target < 0] = 0
        testing_history[testing_history < 0] = 0
        return testing_history, test_target, test_agg

    # 针对普通序列到点结构数据，计算本次分解结果的精度
    def calculate_metrics(self, testing_history, test_agg, test_target, threshold):
        rpaf = recall_precision_accuracy_f1(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                            test_target[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                            threshold)
        rete = relative_error_total_energy(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                           test_target[:test_agg.size - (2 * self.__window_offset)].flatten())
        mae = mean_absolute_error(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                  test_target[:test_agg.size - (2 * self.__window_offset)].flatten())
        return rpaf, rete, mae

    # 针对采用了attention模型的数据，计算本次分解结果的精度
    def calculate_metrics_attention(self, testing_history, test_target, test_agg, predicted_on_off, threshold):
        rpaf = recall_precision_accuracy_f1_attention(
            predicted_on_off[:test_agg.size - (2 * self.__window_offset)].flatten(),
            test_target[:test_agg.size - (2 * self.__window_offset)].flatten(), threshold)
        rete = relative_error_total_energy(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                           test_target[:test_agg.size - (2 * self.__window_offset)].flatten())
        mae = mean_absolute_error(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                  test_target[:test_agg.size - (2 * self.__window_offset)].flatten())
        return rpaf, rete, mae

    # 绘制总功率，实际功率，分解功率对比图
    def print_plots(self, test_agg, test_target, testing_history, count, appliance_name):
        plt.figure(count, figsize=(self.__fig_length, self.__fig_length))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        # plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="总功率")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="实际电器功率")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="分解电器功率")
        plt.title(self.__meter_name + " " + self.__appliance + " " + appliance_name + " " + self.__model_type)
        plt.ylabel("功率/瓦", fontdict={'size': 25})
        plt.xlabel("时间/分钟", fontdict={'size': 25})
        plt.yticks(fontproperties='Times New Roman', size=20)
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.legend(fontsize=25)
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        # axes.set_title("Real Vs Prediction")
        # axes.plot(np.arange(test_target.size), test_target, color='blue')
        # axes.plot(np.arange(testing_history.size), testing_history, color='orange')
        # fig.tight_layout()
        psave_dir="plot_results/" + self.__predict_mode + "/"
        csave_dir="csv_results/" + self.__predict_mode + "/"
        psave_name=self.__meter_name + "_" + self.__appliance + "_" + appliance_name
        csave_name = self.__meter_name + "_" + self.__appliance

        print("writing scv file {}".format(csave_dir+csave_name+".csv"))
        temp=np.array(testing_history)

        #如果目录不存在，创建目录
        directory = os.path.dirname(csave_dir)
        if not os.path.exists(directory):
            print("Creating directory".format(csave_dir))
            os.makedirs(directory)

        #保存csv结果
        with open(csave_dir+csave_name+".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['time', 'power'])
            for index in range(len(testing_history)):
                value = testing_history[index][0]
                writer.writerow([index, value])

            #绘制结果图
        if self.__plot_to_file:
            plt.savefig(
                psave_dir + psave_name + ".png")
        plt.show()

    # 输出本次分解结果的准确率等数据
    def print_metrics(self, appliance_name, rpaf, rete, mae):

        print("======================================Appliance: {}-{}".format(self.__meter_name, appliance_name))
        print("============ Recall: {}".format(rpaf[0]))
        print("============ Precision: {}".format(rpaf[1]))
        print("============ Accuracy: {}".format(rpaf[2]))
        print("============ F1 Score: {}".format(rpaf[3]))
        print("============ Relative error in total energy: {}".format(rete))
        print("============ Mean absolute error(in Watts): {}".format(mae))
        print("                                                  ")

        file_path = "plot_results/" + self.__predict_mode + "/" + self.__meter_name + "_" + self.__appliance + ".log"
        file = open(file_path, mode="w+")
        print("======================================Appliance: {}-{}".format(self.__meter_name, appliance_name),
              file=file)
        print("============ Recall: {}".format(rpaf[0]), file=file)
        print("============ Precision: {}".format(rpaf[1]), file=file)
        print("============ Accuracy: {}".format(rpaf[2]), file=file)
        print("============ F1 Score: {}".format(rpaf[3]), file=file)
        print("============ Relative error in total energy: {}".format(rete), file=file)
        print("============ Mean absolute error(in Watts): {}".format(mae), file=file)
        print("                                                  ", file=file)
        file.close()

    # 数据生成器
    def get_test_generator(self):
        if self.__model_type == 'concat':
            test_generator = TestSlidingWindowGenerator(number_of_windows=self.__number_of_windows,
                                                        appliance_name_list=self.__meter_name_list,
                                                        offset=self.__window_offset,

                                                        predict_mode=self.__predict_mode,
                                                        test_directory=self.__test_directory,
                                                        appliance_count=self.__appliance_count)
            test_input, test_target = test_generator.generate_dataset_concat()
        elif self.__model_type == 'attention_seq':
            test_generator = TestSlidingWindowGeneratorAttentionSeq(number_of_windows=self.__number_of_windows,
                                                                    appliance_threshold=self.__threshold,
                                                                    appliance_name_list=self.__meter_name_list,
                                                                    input_window_length=self._input_window_length,
                                                                    predict_mode=self.__predict_mode,
                                                                    test_directory=self.__test_directory,
                                                                    raw_file_name=self.__raw_test_directory,
                                                                    appliance_count=self.__appliance_count)
            test_input, test_target = test_generator.generate_test_data()
        elif self.__model_type == 'attention_point':
            test_generator = TestSlidingWindowGeneratorAttentionPoint(number_of_windows=self.__number_of_windows,
                                                                      appliance_threshold=self.__threshold,
                                                                      appliance_name_list=self.__meter_name_list,
                                                                      input_window_length=self._input_window_length,
                                                                      predict_mode=self.__predict_mode,
                                                                      test_directory=self.__test_directory,
                                                                      raw_file_name=self.__raw_test_directory,
                                                                      appliance_count=self.__appliance_count,
                                                                      offset=self.__window_offset)
            test_input, test_target = test_generator.generate_test_data()
        elif self.__model_type == 'attention_origin':
            test_generator = TestSlidingWindowGeneratorAttentionOrigin(number_of_windows=self.__number_of_windows,
                                                                       appliance_threshold=self.__threshold,
                                                                       appliance_name_list=self.__meter_name_list,
                                                                       input_window_length=self._input_window_length,
                                                                       predict_mode=self.__predict_mode,
                                                                       test_directory=self.__test_directory,
                                                                       raw_file_name=self.__raw_test_directory,
                                                                       appliance_count=self.__appliance_count,
                                                                       )
            test_input, test_target = test_generator.generate_test_data()
        else:
            test_generator = TestSlidingWindowGeneratorCommon(number_of_windows=self.__number_of_windows,
                                                              appliance_name_list=self.__meter_name_list,
                                                              offset=self.__window_offset,
                                                              predict_mode=self.__predict_mode,
                                                              test_directory=self.__test_directory,
                                                              appliance_count=self.__appliance_count)
            test_input, test_target = test_generator.generate_test_data()
        return test_generator, test_input, test_target


def generate_min_max(meter_name, appliance_name):
    section_name = meter_name + '_' + appliance_name
    appliance_min = cf.getfloat(section_name, 'min')
    appliance_max = cf.getfloat(section_name, 'max')
    return appliance_min, appliance_max


def build_overall_sequence(sequences):
    unique_sequence = []
    matrix = [sequences[::-1, :].diagonal(i) for i in range(-sequences.shape[0] + 1, sequences.shape[1])]
    for i in range(len(matrix)):
        unique_sequence.append(np.median(matrix[i]))
    unique_sequence = np.array(unique_sequence)
    return unique_sequence
