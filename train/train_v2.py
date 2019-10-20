import pandas as pd
import logging
import argparse
from pathlib import Path
import numpy as np
#from keras_self_attention import SeqSelfAttention
from tensorflow.python import keras as tf_k
from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import backend as K
import tensorflow as tf
from wide_resnet import WideResNet
from utils import load_data, load_train_data
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
import sys, os

logging.basicConfig(level=logging.DEBUG)
sys.path.append('module')


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--validation-split", type=float, default=0.1,
                        help="validation split ratio")
    # parser.add_argument("--aug", action="store_true",
    #                     help="use data augmentation if set true")
    # parser.add_argument("--output_path", type=str, default="checkpoints",
    #                    help="checkpoint dir")    
    parser.add_argument('--container-log-level', type=int, default=logging.INFO)
    parser.add_argument('--model-version', type=str, default='')
    parser.add_argument("--weight-file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    

    # 環境変数として渡されるパラメータ
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test-dir', type=str,
                        default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--output-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")

def _run_prepare_commands():
    '''
    事前準備としてOSコマンドを実行します。
    '''

    commands = '''
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
'''

    for command in commands.split('\n'):
        if command == '':
            continue
        ret = subprocess.call(command.split(' '))
        logging.info(ret)

def _save_model(checkpoint_filename, history, model_dir, output_dir, model):
    '''
       モデルや実行履歴を保存します。
    '''
    # Checkpointからモデルをロードし直し(ベストなモデルのロード)
    # model = load_model(checkpoint_filename,
    #                    custom_objects=SeqSelfAttention.get_custom_objects())
    # model = load_model(checkpoint_filename)
    # Historyの保存
    # history_df = pd.DataFrame(history.history)
    # history_df.to_csv(output_dir + f'/history.csv')

    # Endpoint用のモデル保存
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            model_dir + '/1',
            inputs={'inputs': model.input},
            outputs={t.name: t for t in model.outputs})
    
def main():
    
    args = get_args()
    batch_size = args.batch_size
    #batch_size = 1
    epochs = args.epochs
    #nb_epochs = 1
    lr = args.lr
    #lr = float(0.1)
    opt_name = args.opt
    #opt_name = "sgd"
    depth = args.depth
    #depth = 16
    k = args.width
    #k = 8
    validation_split = args.validation_split
    #validation_split = 0.9
    # use_augmentation = args.aug
    use_augmentation = False
    # output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    # output_path = Path(__file__).resolve().parent.joinpath("/output")
    # output_path.mkdir(parents=True, exist_ok=True)
    
    weight = args.weight_file
    
    # ログレベルを引数で渡されたコンテナのログレベルと合わせます。
    logging.basicConfig(level=args.container_log_level)
                        
    # 事前準備を実行します。実行は任意です。
    # _run_prepare_commands()
    
    
    # データのロードを行います。
    logging.debug("Loading data...")    
    image_size = 64
    img, gndr, age = [], [], []
    img, gndr, age = load_train_data("./", "./data/", image_size)

    # X_data = image
    # y_data_g = np_utils.to_categorical(gender, 2)
    # y_data_a = np_utils.to_categorical(age, 101)
    X_data = img
    y_data_g = np_utils.to_categorical(gndr, 2)
    y_data_a = np_utils.to_categorical(age, 101)

    # モデルの定義を行います。
    model = WideResNet(image_size, depth=depth, k=k)()
    model.load_weights(weight)
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

#    callbacks = [LearningRateScheduler(schedule=Schedule(epochs, lr)),
#                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
#                                 monitor="val_loss",
#                                 verbose=1,
#                                 save_best_only=True,
#                                 mode="auto")
#                 ]
    
    # checkpoint_filename = str(args.output_dir) + "/weights.{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint_filename = f'model_{args.model_version}.h5'
    callbacks = [LearningRateScheduler(schedule=Schedule(epochs, lr)),
                 ModelCheckpoint(checkpoint_filename,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]
    
    # 学習を実行します。
    logging.debug("Running training...")

    data_num = len(X_data)
    # indexes = np.arange(data_num)
    # np.random.shuffle(indexes)
    # X_data = X_data[indexes]
    # y_data_g = y_data_g[indexes]
    # y_data_a = y_data_a[indexes]
    # train_num = int(data_num * (1 - validation_split))
    train_num = 3
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    #if use_augmentation:
    #    datagen = ImageDataGenerator(
    #        width_shift_range=0.1,
    #        heig#ht_shift_range=0.1,
    #        ho#ri#zontal_flip=True,
    #        pr#ep#rocessing_function=get_random_eraser(v_l=0, v_h=255))
    #    training#_generator = MixupGenerator(X_train, [y_train_g, y_train_a], batch_size=batch_size, alpha=0.2,
    #          #  #                            datagen=datagen)()
    #    hist = m#odel.fit_generator(generator=training_generator,
    #            #                   steps_per_epoch=train_num // batch_size,
    #            #                   validation_data=(X_test, [y_test_g, y_test_a]),
    #            #                   epochs=nb_epochs, verbose=1,
    #            #                   callbacks=callbacks)
    #else#:
    #    #hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=epochs, callbacks=callbacks,
    #    #                 validation_data=(X_test, [y_test_g, y_test_a]))##

    #logging.debug("Saving history...")
    
    #_save_model(model, hist, args.model_dir, args.output_dir)
    _save_model(checkpoint_filename, None, args.model_dir, args.output_dir, model)
    # pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history_{}_{}.h5".format(depth, k)), "history")


if __name__ == '__main__':
    main()