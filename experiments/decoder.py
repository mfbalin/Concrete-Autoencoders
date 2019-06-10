#!/usr/bin/env python

import sys

import numpy as np
import tensorflow as tf

from datetime import datetime
from os.path import join, exists
from os import makedirs
import json

from concrete_estimator import dataset_input_fn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

figure_dir = 'figures'

def MSELR(indices, train, val, test):
    #train = train.astype('float64')
    #val = val.astype('float64')
    #test = test.astype('float64')
    not_indices = list(set(range(train.shape[1])) - set(indices))
    LR = LinearRegression(n_jobs = 6)
    LR.fit(train[:, indices], train[:, not_indices])
    mse_val = mean_squared_error(LR.predict(val[:, indices]), val[:, not_indices])
    mse_test = mean_squared_error(LR.predict(test[:, indices]), test[:, not_indices])
    return float(mse_val), float(mse_test)

def test_GEO(name, indices, hidden_units):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    dataset_dir = 'datasets'
    
    train = np.load(join(dataset_dir, 'bgedv2_XY_tr_float32.npy'), mmap_mode = 'r')
    val = np.load(join(dataset_dir, 'bgedv2_XY_va_float32.npy'), mmap_mode = 'r')
    test = np.load(join(dataset_dir, 'bgedv2_XY_te_float32.npy'), mmap_mode = 'r')
    
    model_dir = (name + '_decoder_dir_' + str(datetime.now())).replace(' ', '_').replace(':', '.')
    
    if not exists(model_dir):
        makedirs(model_dir)
    
    indices = list(indices)
    not_indices = list(set(range(train.shape[1])) - set(indices))
    
    print(indices[: 10])
    with open(join(model_dir, 'indices.txt'), 'w') as file:
        file.write(str(list(indices)))
    
    mse_val = 0
    mse_test = 0
    
    if len(hidden_units) == 0:
        mse_val, mse_test = MSELR(indices, train, val, test)
    else:
        num_epochs = 100
        batch_size = 256
        steps_per_epoch = (train.shape[0] + batch_size - 1) // batch_size
        epochs_per_evaluation = 50
        dropout = 0.1
        learning_rate = 1e-3
        # optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        optimizer = lambda: tf.train.AdamOptimizer(learning_rate = tf.train.exponential_decay(learning_rate = learning_rate, global_step = tf.train.get_global_step(), decay_steps = steps_per_epoch, decay_rate = 0.95, staircase = True))
        
        train_input_fn = lambda: dataset_input_fn((train[:, indices], train[:, not_indices]), batch_size, -1)
        eval_input_fn = lambda: dataset_input_fn((val[:, indices], val[:, not_indices]), batch_size, seed = 1)
        test_input_fn = lambda: dataset_input_fn((test[:, indices], test[:, not_indices]), batch_size, seed = 1)

        feature_columns = [tf.feature_column.numeric_column(key = 'features', shape = [len(indices)])]

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        regressor = tf.estimator.DNNRegressor(hidden_units = hidden_units, feature_columns = feature_columns, label_dimension = len(not_indices), optimizer = optimizer, loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE, activation_fn = tf.nn.leaky_relu, dropout = dropout, model_dir = model_dir, config = tf.estimator.RunConfig(save_checkpoints_steps = epochs_per_evaluation * steps_per_epoch, save_summary_steps = steps_per_epoch, log_step_count_steps = steps_per_epoch, session_config = session_config))

        train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, max_steps = num_epochs * steps_per_epoch)
        eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn, steps = None, throttle_secs = 0)

        tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)

        eval_evaluation_result = regressor.evaluate(eval_input_fn)

        test_evaluation_result = regressor.evaluate(test_input_fn)
        
        mse_val = float(eval_evaluation_result['average_loss'])
        
        mse_test = float(test_evaluation_result['average_loss'])
    
    results = (mse_val, mse_test)
    print(name, results)
    with open(join(figure_dir, name), 'w') as f:
        json.dump(results, f)
        
    return results
    
if __name__ == "__main__":
    sz = 9000
    cae_indices = [5232, 6009, 5312, 7518, 6647, 2336, 7747, 6841, 9276, 3342, 6154, 4480, 2396, 1236, 1185, 4215, 9410, 7130, 414, 9395, 7062, 8143, 2715, 7454, 5911, 4314, 9358, 1657, 3851, 3495, 4366, 1347, 1239, 203, 4057, 4859, 9687, 3631, 5071, 9897, 7288, 4994, 3816, 3483, 7051, 4788, 6184, 4182, 307, 6128, 4245, 4079, 3466, 1217, 2806, 3312, 6054, 1891, 6798, 5606, 4414, 4955, 6753, 4957, 1362, 657, 7188, 8148, 8561, 9867, 10002, 5971, 5415, 3928, 2288, 3361, 5728, 6524, 4372, 8385, 5601, 10331, 1341, 8234, 6504, 8569, 9189, 1242, 602, 4618, 4190, 5265, 2693, 2212, 4852, 2204, 882, 8995, 794, 180, 433, 7955, 4444, 8767, 8926, 3822, 1116, 3431, 7353, 6917, 10401, 568, 5711, 9620, 9053, 3857, 7362, 772, 9692, 9433, 7673, 7486, 237, 1502, 6206, 10307, 7608, 1322, 7528, 1067, 5136, 167, 9961, 7210, 2512, 3997, 1276, 1757, 2408, 10254, 7196, 5388, 5922, 6047, 8261, 5591, 9695, 2264, 3294, 3009, 6018, 2965, 3321, 690, 3629, 10073, 3421, 2558, 6656, 95, 6114, 2682, 2443, 5497, 3818, 6061, 8866, 7508, 9306, 4272, 68, 7378, 5831, 2540, 1434, 6674, 3650, 620, 6745, 1359, 1181, 348, 7560, 7798, 5043, 8472, 9762, 235, 6112, 7425, 10356, 6323, 1080, 7494, 2735, 6337, 2233, 10065, 6309, 6074, 3705, 5785, 4132, 9278, 4711, 5673, 1404, 6937, 9216, 3904, 7304, 7932, 3303, 4700, 358, 8081, 6664, 9928, 3641, 1336, 8770, 3730, 1677, 6597, 6502, 7225, 4932, 2101, 7770, 1010, 7897, 8188, 5031, 2520, 8448, 4755, 5077, 7477, 4861, 6304, 8386, 2410, 5223, 7813, 6087, 4890, 8914, 6733, 6916, 5867, 7764, 6269, 3132, 3637, 5730, 9812, 4007, 6972, 10104, 7450, 6440, 7546, 1921, 9500, 6369, 6260, 6077, 3835, 10428, 6390, 1011, 3869, 1886, 292, 8516, 6385, 67, 4123, 2953, 6060, 254, 1436, 6278, 641, 8036, 8604, 5501, 4398, 273, 2263, 3796, 9133, 7866, 3427, 6422, 2294, 971, 2811, 9517, 8209, 4695, 5956, 7685, 2240, 10014, 7921, 6386, 5836, 4784, 1174, 1349, 7148, 9761, 6134, 7480, 3765, 4972, 7356, 6697, 17, 6058, 4538, 8817, 4153, 5309, 8051, 7579, 2814, 4515, 6985, 2068, 7068, 5326, 5634, 6930, 5123, 9068, 9663, 9519, 3946, 5073, 3208, 5146, 5423, 2185, 4554, 3335, 7146, 2816, 3027, 6752, 113, 4858, 6119, 2052, 5746, 2429, 1182, 8501, 10385, 9397, 321, 7593, 1943, 7969, 8543, 3506, 5384, 3883, 8389, 8196, 2353, 8266, 5468, 3570, 8717, 5421, 9021, 2345, 4623, 7162, 9578, 144, 519, 5674, 3512, 2990, 8285, 7769, 2949, 6631, 9239, 8937, 8220, 4261, 1091, 2932, 715, 9906, 5530, 4576, 2598, 8704, 1950, 6340, 6248, 1711, 7366, 9219, 8929, 9787, 1741, 7086, 8270, 26, 3110, 10235, 10301, 139, 957, 5457, 1614, 6071, 3772, 8994, 805, 3980, 4411, 2703, 7034, 8859, 3390, 6933, 8521, 1069, 3250, 2685, 7468, 2592, 4893, 8260, 5003, 3625, 9151, 6163, 1503, 9774, 9399, 5184, 3408, 174, 2532, 7132, 973, 2754, 2043, 642, 624, 6766, 9649, 9610, 9754, 3315, 5462, 721, 6931, 2579, 2808, 6246, 3420, 4019, 5855, 2376, 8875, 897, 8341, 5665, 1840, 9736, 4520, 4774, 3429, 5151, 2993, 670, 2195, 2251, 933, 6126, 1648, 3063, 2290, 1701, 8991, 1514, 730, 9836, 6720, 10382, 5044, 7273, 6737, 4705, 6707, 975, 10184, 8845, 1051, 2351, 8842, 217, 8005, 5337, 553, 3545, 9709, 1910, 2238, 6147, 7127, 9359, 2089, 2976, 2030, 8017, 9089, 6006, 9571, 9394, 6461, 9808, 7559, 10374, 10279, 1637, 6804, 1808, 8399, 7185, 215, 6683, 5139, 8540, 326, 1376, 5935, 2214, 3686, 3260, 5910, 4645, 4582, 7303, 6231, 9472, 9355, 5954, 7741, 2955, 5409, 8117, 9830, 4655, 1508, 8882, 4629, 8396, 3873, 5953, 98, 6050, 7325, 8059, 1792, 5754, 2847, 8474, 10078, 2488, 360, 8634, 2923, 9567, 6176, 9010, 625, 5571, 6041, 242, 7670, 7209, 10330, 728, 706, 738, 1202, 7550, 47, 2437, 2613, 9122, 5845, 1396, 7720, 5541, 7155, 921, 8582, 789, 9265, 5453, 5417, 2057, 8119, 3590, 3201, 76, 5813, 166, 3649, 6585, 3296, 4917, 6158, 52, 5869, 5212, 10143, 2256, 2853, 5872, 5702, 4531, 1160, 4793, 8870, 10181, 9487, 1871, 5032, 508, 3991, 5833, 7145, 4130, 4136, 3748, 10297, 9299, 2594, 9683, 250, 5182, 3743, 7936, 9282, 5733, 929, 4812, 2833, 4458, 3712, 1372, 8267, 5745, 2774, 7776, 958, 3931, 4780, 5576, 6729, 979, 8520, 6328, 7503, 6262, 5806, 5040, 9061, 3217, 3660, 7037, 6403, 1275, 4820, 9100, 6718, 10227, 8892, 10207, 7704, 4733, 9405, 2404, 8466, 618, 7662, 7700, 134, 9374, 3984, 8193, 6739, 2643, 6282, 4619, 4297, 8088, 1663, 169, 2098, 8881, 6979, 1777, 920, 10203, 9518, 9644, 5117, 8787, 5256, 8055, 7604, 2025, 4082, 6467, 6420, 4269, 1472, 2584, 6588, 6455, 222, 3858, 6695, 6565, 335, 4563, 7207, 6821, 8861, 8053, 6781, 5955, 1675, 2589, 6399, 6938, 3981, 5082, 3682, 3850, 771, 775, 5224, 8674, 7895, 2541, 8355, 4822, 8591, 1742, 4635, 3852, 9765, 6266, 1064, 3901, 9160, 6410, 7152, 9400, 1285, 3350, 1065, 830, 7099, 9261, 10350, 3057, 6583, 7104, 3048, 8100, 1841, 9249, 9044, 2591, 7933, 2534, 7335, 448, 294, 9544, 3643, 3676, 5946, 7726, 2045, 8830, 8083, 9758, 10362, 7018, 411, 3594, 1839, 1579, 3030, 5641, 5446, 4064, 10211, 4311, 2026, 6466, 5339, 2487, 5747, 4204, 6027, 4804, 8510, 8300, 4124, 784, 9420, 8822, 109, 9526, 2639, 823, 1019, 2960, 6207, 2508, 434, 4840, 4952, 9340, 7855, 5608, 312, 7240, 279, 4819, 6242, 3380, 8893, 5447, 2012, 9562, 2489, 4696, 4900, 4833, 403, 9543, 4088, 9188, 9251, 2662, 4241, 4176, 8441, 2686, 2967, 8712, 8894, 3912, 1554, 4298, 133, 5829, 7950, 4431, 7004, 6135, 5277, 2418, 3153, 27, 8430, 3706, 2363, 1126, 1647, 9817, 497, 342, 10460, 3083, 1377, 4634, 2770, 349, 5817, 6651, 2948, 4792, 615, 6603, 3415, 585, 1857, 7406, 304, 4456, 3467, 3336, 10121, 2969, 7402, 4760, 4278, 7828, 2491, 8885, 6668, 1685, 4425, 8611, 5122, 3060, 2818, 2593, 10100, 9283, 6370, 1572, 367, 1573, 5448, 2027, 6645, 8828, 1752, 7251, 9014, 1522, 3245, 8827]
    for hidden_units in [[], [sz], [sz, sz], [sz, sz, sz]]:
        for i in range(3):
            test_GEO('%d_GEO_landmarks_%d_hidden_layers' % (i, len(hidden_units)), np.arange(943), hidden_units)
            test_GEO('%d_GEO_CAE_%d_hidden_layers' % (i, len(hidden_units)), cae_indices, hidden_units)
