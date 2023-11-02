import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import csv

from train_model.network.model import model_select, save_model, load_model
from train_model.train_generator.attention.train_generator_attention_seq import TrainSlidingWindowGeneratorAttentionSeq
from train_model.train_generator.attention.train_generator_attention_origin import \
    TrainSlidingWindowGeneratorAttentionPure
from train_model.train_generator.attention.train_generator_attention_point import \
    TrainSlidingWindowGeneratorAttentionPoint
from train_model.train_generator.concat.train_generator_concat import TrainSlidingWindowGenerator
from train_model.train_generator.common.train_generator_common import TrainSlidingWindowGeneratorCommon


class Trainer:

    def __init__(self, scv_name, appliance, appliance_threshold, batch_size, model_type,
                 training_directory, training_directory_raw, validation_directory, validation_directory_raw,
                 save_model_dir, predict_mode, appliance_count,
                 epochs=100, input_window_length=50, validation_frequency=1,
                 patience=3, min_delta=1e-6, verbose=1, learning_rate=0.001, is_load_model=False, plot=False):
        self.__scv_name = scv_name
        self.__appliance = appliance
        self.__appliance_threshold = appliance_threshold
        self.__model_type = model_type
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__patience = patience
        self.__min_delta = min_delta
        self.__verbose = verbose
        self.__loss = "mse"
        self.__metrics = ["mse", "msle", "mae"]
        self.__learning_rate = learning_rate
        self.__beta_1 = 0.9
        self.__beta_2 = 0.999
        self.__save_model_dir = save_model_dir
        self.__predict_mode = predict_mode
        if self.__predict_mode == 'single' or self.__predict_mode == 'single_file':
            self.__appliance_count = 1
        else:
            self.__appliance_count = appliance_count
        self.__input_window_length = input_window_length
        self.__window_offset = int((0.5 * (self.__input_window_length + 2)) - 1)
        self.__validation_frequency = validation_frequency
        self.__validation_steps = 100
        self.__training_directory = training_directory
        self.__training_directory_raw = training_directory_raw
        self.__validation_directory = validation_directory
        self.__validation_directory_raw = validation_directory_raw
        self.__is_load_model = is_load_model
        self.__plot = plot
        np.random.seed(120)
        tf.random.set_seed(120)
        self.__training_generator, self.__validation_generator = self.get_generator()

    def train_model(self):
        steps_per_training_epoch = np.round(int(self.__training_generator.maximum_batch_size / self.__batch_size), decimals=0)
        if os.path.exists(self.__save_model_dir) and self.__is_load_model:
            model = load_model(self.__save_model_dir, self.__model_type, self.__input_window_length)
        else:
            model= model_select(self.__input_window_length, self.__model_type, self.__appliance_count, self.__predict_mode)
            self.compile_model(model)

        training_history = self.train_process(model, steps_per_training_epoch)
        training_history.history["val_loss"] = np.repeat(training_history.history["val_loss"], self.__validation_frequency)
        model.summary()
        save_model(model, self.__save_model_dir, self.__model_type)
        if self.__plot:
            self.plot_training_results(training_history)

    def train_process(self, model, steps_per_training_epoch):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.__min_delta,
                                                          patience=self.__patience, verbose=self.__verbose, mode="auto")
        callbacks = [early_stopping]
        training_history = model.fit(self.__training_generator.load_dataset(),
                                     steps_per_epoch=steps_per_training_epoch,
                                     epochs=self.__epochs,
                                     verbose=self.__verbose,
                                     callbacks=callbacks,
                                     validation_data=self.__validation_generator.load_dataset(),
                                     validation_freq=self.__validation_frequency,
                                     validation_steps=self.__validation_steps)
        return training_history

    def plot_training_results(self, training_history):
        save_dir = "csv_train/" + self.__predict_mode + "/"
        save_name = self.__scv_name
        # 如果目录不存在，创建目录
        directory = os.path.dirname(save_dir)
        if not os.path.exists(directory):
            print("Creating directory".format(save_dir))
            os.makedirs(directory)

        # 保存csv结果
        with open(save_dir + save_name + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Epoch', 'MSE (Training Loss)', 'MSE (Validation Loss)'])
            for index in range(len(training_history.history["val_loss"])):
                loss = training_history.history["loss"][index]
                val_loss = training_history.history["val_loss"][index]
                writer.writerow([index, loss,val_loss])


        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title('Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def compile_model(self, model):
        if self.__model_type == 'attention_origin':
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss={
                "output": tf.keras.losses.MeanSquaredError(),
                "classification_output": tf.keras.losses.BinaryCrossentropy()})
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.__learning_rate, beta_1=self.__beta_1,
                                                             beta_2=self.__beta_2),
                          loss=self.__loss, metrics=self.__metrics)

    def get_generator(self):
        if self.__model_type == 'concat':
            training_generator = TrainSlidingWindowGenerator(file_name=self.__training_directory,
                                                             batch_size=self.__batch_size,
                                                             shuffle=True,
                                                             offset=self.__window_offset,
                                                             predict_mode=self.__predict_mode,
                                                             appliance_count=self.__appliance_count)
            validation_generator = TrainSlidingWindowGenerator(file_name=self.__validation_directory,
                                                               batch_size=self.__batch_size,
                                                               shuffle=True,
                                                               offset=self.__window_offset,
                                                               predict_mode=self.__predict_mode,
                                                               appliance_count=self.__appliance_count)
        elif self.__model_type == 'attention_seq':
            training_generator = TrainSlidingWindowGeneratorAttentionSeq(file_name=self.__training_directory,
                                                                         appliance_threshold=self.__appliance_threshold,
                                                                         raw_file_name=self.__training_directory_raw,
                                                                         batch_size=self.__batch_size,
                                                                         shuffle=True,
                                                                         window_size=self.__input_window_length,
                                                                         predict_mode=self.__predict_mode,
                                                                         appliance_count=self.__appliance_count)
            validation_generator = TrainSlidingWindowGeneratorAttentionSeq(file_name=self.__validation_directory,
                                                                           appliance_threshold=self.__appliance_threshold,
                                                                           raw_file_name=self.__validation_directory_raw,
                                                                           batch_size=self.__batch_size,
                                                                           shuffle=True,
                                                                           window_size=self.__input_window_length,
                                                                           predict_mode=self.__predict_mode,
                                                                           appliance_count=self.__appliance_count)
        elif self.__model_type == 'attention_point':
            training_generator = TrainSlidingWindowGeneratorAttentionPoint(file_name=self.__training_directory,
                                                                           appliance_threshold=self.__appliance_threshold,
                                                                           raw_file_name=self.__training_directory_raw,
                                                                           batch_size=self.__batch_size,
                                                                           shuffle=True,
                                                                           window_size=self.__input_window_length,
                                                                           predict_mode=self.__predict_mode,
                                                                           appliance_count=self.__appliance_count,
                                                                           offset=self.__window_offset)
            validation_generator = TrainSlidingWindowGeneratorAttentionPoint(file_name=self.__validation_directory,
                                                                             appliance_threshold=self.__appliance_threshold,
                                                                             raw_file_name=self.__validation_directory_raw,
                                                                             batch_size=self.__batch_size,
                                                                             shuffle=True,
                                                                             window_size=self.__input_window_length,
                                                                             predict_mode=self.__predict_mode,
                                                                             appliance_count=self.__appliance_count,
                                                                             offset=self.__window_offset)
        elif self.__model_type == 'attention_origin':
            training_generator = TrainSlidingWindowGeneratorAttentionPure(file_name=self.__training_directory,
                                                                          appliance_threshold=self.__appliance_threshold,
                                                                          raw_file_name=self.__training_directory_raw,
                                                                          batch_size=self.__batch_size,
                                                                          shuffle=True,
                                                                          window_size=self.__input_window_length,
                                                                          predict_mode=self.__predict_mode,
                                                                          appliance_count=self.__appliance_count)
            validation_generator = TrainSlidingWindowGeneratorAttentionPure(file_name=self.__validation_directory,
                                                                            appliance_threshold=self.__appliance_threshold,
                                                                            raw_file_name=self.__validation_directory_raw,
                                                                            batch_size=self.__batch_size,
                                                                            shuffle=True,
                                                                            window_size=self.__input_window_length,
                                                                            predict_mode=self.__predict_mode,
                                                                            appliance_count=self.__appliance_count)
        else:
            training_generator = TrainSlidingWindowGeneratorCommon(file_name=self.__training_directory,
                                                                   batch_size=self.__batch_size,
                                                                   shuffle=True,
                                                                   offset=self.__window_offset,
                                                                   predict_mode=self.__predict_mode,
                                                                   appliance_count=self.__appliance_count)
            validation_generator = TrainSlidingWindowGeneratorCommon(file_name=self.__validation_directory,
                                                                     batch_size=self.__batch_size,
                                                                     shuffle=True,
                                                                     offset=self.__window_offset,
                                                                     predict_mode=self.__predict_mode,
                                                                     appliance_count=self.__appliance_count)
        return training_generator, validation_generator
