# -*- coding: utf-8 -*-
"""
@Description :
@Time : 2020/3/8 19:45 
@Author : Kunsong Zhao
@Version：1.0
"""

import time
import json
from param import Config
from warnings import filterwarnings
from data_loader import DataLoader
from model import Model
from evaluator import Evaluator

__author__ = 'kszhao'


filterwarnings('ignore')

FEATURES_NUM = 6

STEPS = 50

config = Config()

measure_names = ['MCC']

algo_names = ['WCE']

dataset_names = ['aFall', 'Alfresco', 'androidSync', 'androidWalpaper', 'anySoftkeyboard',
                 'Apg', 'chatSecure',  'facebook', 'kiwis', 'owncloudandroid', 'Pageturner',
                 'reddit']


def main():
    s = time.time()
    dl = DataLoader()
    print(time.time() - s)

    for project in dataset_names:

        result = {project: {i: {algo: {measure: []
                                       for measure in measure_names}
                                for algo in algo_names}
                            for i in range(STEPS)}}

        counter = 0

        while counter != STEPS:

            try:

                x_train_scaled, x_test_scaled, y_train, y_test = dl.build_data(project)

                pos_count = DataLoader.get_positive_count(y_train)
                all_count = len(y_train)

                for algo in algo_names:

                    net = Model(FEATURES_NUM,
                                hidden_shape=config.hidden_layer,
                                classes=config.classes,
                                positive_num=pos_count,
                                all_num=all_count)

                    # net.train(x_train_scaled, y_train, config)
                    net.train_batch(x_train_scaled, y_train, config)

                    y_pred, y = net.test(x_test_scaled, y_test, config)

                    result = Evaluator.evaluate(measure_names, y_pred, y, result, project, algo, counter)

                print('This is the ' + str(counter + 1) + 'times')

            except BaseException as err:
                print('The' + str(counter) + 'times loop error')
                print(err)
                continue

            counter += 1

            try:
                # reverse
                reverse_x_train = x_test_scaled
                reverse_y_train = y_test
                reverse_x_test = x_train_scaled
                reverse_y_test = y_train

                pos_count = DataLoader.get_positive_count(reverse_y_train)
                all_count = len(reverse_y_train)

                net = Model(FEATURES_NUM,
                            hidden_shape=config.hidden_layer,
                            classes=config.classes,
                            positive_num=pos_count,
                            all_num=all_count)

                net.train_batch(reverse_x_train, reverse_y_train, config)

                y_pred, y = net.test(reverse_x_test, reverse_y_test, config)

                result = Evaluator.evaluate(measure_names, y_pred, y, result, project, algo, counter)

            except BaseException as err:
                print('The' + str(counter) + 'times loop error')
                print(err)
                counter -= 2
            finally:
                counter += 1

            continue

        with open('./results/algo--final--' + str(STEPS) + '--' + project + '.json', 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    main()








