from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9), device_count = {'GPU': 0}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

def resnet50_training(train_directory, validation_directory):
    name = 'ResNet50_' + train_directory.split('/')[-1] +'_'
    model_path = name + '{val_loss:.4f}.h5'
    print('#####################################  '+name+'  ###############################################')

    input = Input(shape=(224, 224, 3))
    model = ResNet50(input_tensor=input, include_top=False, weights='imagenet', pooling='max')

    x = model.output
    x = Dense(1024, name='fully', init='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(512, init='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #Dense[0] 자리에는 클래스 개수가 들어가야됨
    x = Dense(4, activation='softmax', name='softmax')(x)
    model = Model(model.input, x)
    #model.summary()

    train_datagen = ImageDataGenerator(rescale=1./255)#rgb값 reduce
    train_generator = train_datagen.flow_from_directory(
            train_directory,
            target_size=(224, 224),
            batch_size=30,
            class_mode='categorical')

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
            validation_directory,
            target_size=(224, 224),
            batch_size=30,
            class_mode='categorical')

    model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4),
                      #optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])

    print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: model.layers[3].get_weights())
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=10 , mode='auto', monitor='val_loss')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=50,
                                  epochs=200,
                                  validation_data=val_generator,
                                  validation_steps=5,
                                  callbacks=[early_stopping, print_weights, cb_checkpoint])#, cb_checkpoint])



    #모델 평가
    print("-- Evaluate --")
    scores = model.evaluate_generator(val_generator, steps=5)
    print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

    name = name + '_' + str(int(scores[1]*100))

    model.save(name + '.h5')

    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(name + '.png')
    #plt.show()
    plt.clf()


    """
    # Store the fully connected layers
    fc1 = model.layers[-3]
    fc2 = model.layers[-2]
    predictions = model.layers[-1]
    # Create the dropout layers
    dropout1 = Dropout(0.85)
    dropout2 = Dropout(0.85)
    # Reconnect the layers
    x = dropout1(fc1.output)
    x = fc2(x)
    x = dropout2(x)
    predictors = predictions(x)
    # Create a new model
    model2 = Model(inputs=model.input, outputs=predictors)
    model2.summary()
    """

train_dir = 'G:/gcf/train'
validation_dir = 'G:/gcf/test'
resnet50_training(train_dir, validation_dir)

"""
train_dir = 'G:/gcf/train_1'
validation_dir = 'G:/gcf/test'
resnet50_training(train_dir, validation_dir)

train_dir = 'G:/gcf/train_2'
validation_dir = 'G:/gcf/test'
resnet50_training(train_dir, validation_dir)

train_dir = 'G:/gcf/train_1t'
validation_dir = 'G:/gcf/test'
resnet50_training(train_dir, validation_dir)

train_dir = 'G:/gcf/train_2t'
validation_dir = 'G:/gcf/test'
resnet50_training(train_dir, validation_dir)
"""
