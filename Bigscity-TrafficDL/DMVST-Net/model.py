import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
from keras.models import Model
from keras import activations
from keras.engine.topology import Layer
from keras.layers import LSTM, Dense, Input, Flatten, concatenate, Reshape
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import metrics
from keras.layers.normalization import BatchNormalization

batch_size = 64
mean_label = 0.0
label_max = 0
label_min = 0
max_epoch = 200
num_feature = 100
seq_len = 8
hidden_dim = 512
threshold = 10.0
maxtruey = 0
mintruey = 0
eps = 1e-5
loss_lambda = 10.0
feature_len = 0
local_image_size = 9
cnn_hidden_dim_first = 32
len_valid_id = 0
toponet_len = 32


# sess = tf.Session()
# K.set_session(sess)


# def sample_get(datasource, cnt):
#     X, Y = [], []
#     for i in range(datasource.shape[0]):
#         if i % cnt < seq_len - 1:
#             continue
#         tmpx, tmpy = [], []
#         for j in range(seq_len):
#             tmpx.append(datasource[i - seq_len - 1 + j, :-1])
#             if j == seq_len - 1:
#                 tmpy.append(datasource[i - seq_len - 1 + j, -1])
#         X.append(tmpx)
#         Y.append(tmpy)
#     return np.array(X), np.array(Y)
#
#
# def sample_get_network(datasource, cnt):
#     X = []
#     for i in range(datasource.shape[0]):
#         if i % cnt < seq_len - 1:
#             continue
#         tmpx = []
#         for j in range(seq_len):
#             tmpx.append(datasource[i - seq_len - 1 + j, :, :])
#         X.append(tmpx)
#     return np.array(X)
#
#
# def sample_get_static(datasource, cnt):
#     X = []
#     for i in range(datasource.shape[0]):
#         if i % cnt < seq_len - 1:
#             continue
#         X.append(datasource[i - seq_len - 1 + (seq_len - 1), :])
#     return np.array(X)
#
#
# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_percentage_error_revise(y_true, y_pred):
    ma = label_max
    mi = label_min
    y_true = y_true * (ma - mi) + mi
    y_pred = y_pred * (ma - mi) + mi
    diff = diff = K.square(y_true - y_pred) / \
                  K.clip(K.square(y_true), K.epsilon(), None)
    mean_lable_float32 = mean_label.astype(np.float32)
    return 10. * K.mean(diff, axis=-1) + loss_lambda / K.square(mean_lable_float32) * K.mean(K.square(y_pred - y_true),
                                                                                             axis=-1)
    # return 10. * K.mean(diff, axis=-1)


def get_mape(y_true, y_pred, max_value, min_value):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df * (max_value - min_value) + min_value
    df_new = df[df.y_true > 10 - 1e-10]
    y_true = np.array(df_new.y_true)
    y_pred = np.array(df_new.y_pred)
    y_true_nofilter = np.array(df.y_true)
    y_pred_nofilter = np.array(df.y_pred)
    print('Number of sample whose label beyond 10: %d\n' % df_new.shape[0])
    res = sum(abs(2 * (y_true - y_pred) / (y_true + y_pred))) / len(y_true)
    res_2 = np.sqrt(np.mean((y_true - y_pred) * (y_true - y_pred)))
    res_3 = sum(abs((y_true - y_pred) / (y_true + 10))) / len(y_true)
    fw = open('dmvst.npz', 'w')
    np.savez(fw, true=y_true, pred=y_pred)
    return res, res_2, res_3


class Local_Seq_Conv(Layer):
    def __init__(self, output_dim, seq_len, feature_size, kernel_size, activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', padding='same', strides=(1, 1), **kwargs):
        super(Local_Seq_Conv, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bias_initializer = bias_initializer
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.strides = strides
        self.activation = activations.get(activation)

    def build(self, input_shape):
        batch_size = input_shape[0]
        self.kernel = []
        self.bias = []
        for eachlen in range(self.seq_len):
            self.kernel += [self.add_weight(shape=self.kernel_size,
                                            initializer=self.kernel_initializer,
                                            trainable=True, name='kernel_{0}'.format(eachlen))]

            self.bias += [self.add_weight(shape=(self.kernel_size[-1],),
                                          initializer=self.bias_initializer,
                                          trainable=True, name='bias_{0}'.format(eachlen))]
        self.build = True

    def call(self, inputs):
        output = []
        for eachlen in range(self.seq_len):

            tmp = K.bias_add(K.conv2d(inputs[:, eachlen, :, :, :], self.kernel[eachlen],
                                      strides=self.strides, padding=self.padding), self.bias[eachlen])

            if self.activation is not None:
                output += [self.activation(tmp)]

        output = tf.stack(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)


# def build_model(trainX, trainY, testX, testY, trainimage, testimage, traintopo, testtopo, feature_len):
def build_model():
    # X_train, Y_train, X_test, Y_test = Featureset_get()

    image_input = Input(shape=(seq_len, local_image_size,
                               local_image_size, None), name='cnn_input')
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, 1, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(image_input)
    spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    # spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len, kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(spatial)
    # spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    # spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len, kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(spatial)
    spatial = Flatten()(spatial)
    spatial = Reshape(target_shape=(seq_len, -1))(spatial)
    spatial_out = Dense(units=64, activation='relu')(spatial)

    lstm_input = Input(shape=(seq_len, feature_len),
                       dtype='float32', name='lstm_input')

    x = concatenate([lstm_input, spatial_out], axis=-1)
    # lstm_out = Dense(units=128, activation=relu)(x)
    lstm_out = LSTM(units=hidden_dim, return_sequences=False, dropout=0)(x)

    topo_input = Input(shape=(toponet_len,), dtype='float32', name='topo_input')
    topo_emb = Dense(units=6, activation='tanh')(topo_input)
    static_dynamic_concate = concatenate([lstm_out, topo_emb], axis=-1)

    res = Dense(units=1, activation='sigmoid')(static_dynamic_concate)
    model = Model(inputs=[image_input, lstm_input, topo_input],
                  outputs=res)
    sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss=mean_absolute_percentage_error_revise, optimizer=sgd,
                  metrics=[metrics.mae])
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=0, mode='min')
    # model.fit([trainimage, trainX, traintopo], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1,
    #           callbacks=[earlyStopping])
    # model.save('local_conv_lstm_total_embed.h5')
    return model


if __name__ == '__main__':
    build_model()
