''' Credit to https://github.com/brianvanleeuwen/Halite-ML-starter-bot for the starter code '''

import datetime
import os

import numpy as np
import json

from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

REPLAY_FOLDER = './replays'
training_input = []
training_target = []
np.random.seed(1223) # for reproducibility

VISIBLE_DISTANCE = 5
input_shape=((2*VISIBLE_DISTANCE+1),(2*VISIBLE_DISTANCE+1), 4)

nb_filters = 256

from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten

model = Sequential([Convolution2D(nb_filters, 6, 6, border_mode='valid', input_shape = input_shape, dim_ordering='tf'),
                   LeakyReLU(),
                   Convolution2D(nb_filters, 3, 3, border_mode='valid', dim_ordering='tf'),
                   LeakyReLU(),
                   Flatten(),
                   Dense(128),
                   LeakyReLU(),
                   Dense(32),
                   LeakyReLU(),
                   Dense(5, activation='softmax')])


model.compile('rmsprop','categorical_crossentropy', metrics=['accuracy'])

model_name = 'model.h5'


def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

size = len(os.listdir(REPLAY_FOLDER))
for index, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    if replay_name[-4:]!='.hlt':continue
    print('Loading {} ({}/{})'.format(replay_name, index, size))
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))

    frames=np.array(replay['frames'])
    player=frames[:,:,:,0]
    players,counts = np.unique(player[-1],return_counts=True)
    target_id = players[counts.argmax()]
    if target_id == 0: continue

    prod = np.repeat(np.array(replay['productions'])[np.newaxis],replay['num_frames'],axis=0)
    strength = frames[:,:,:,1]

    moves = (np.arange(5) == np.array(replay['moves'])[:,:,:,None]).astype(int)[:128]
    stacks = np.array([player==target_id,(player!=target_id) & (player!=0),prod/20,strength/255])
    stacks = stacks.transpose(1,0,2,3)[:len(moves)].astype(np.float32)

    position_indices = stacks[:,0].nonzero()
    sampling_rate = 1/stacks[:,0].mean(axis=(1,2))[position_indices[0]]
    sampling_rate *= moves[position_indices].dot(np.array([1,10,10,10,10])) # weight moves 10 times higher than still
    sampling_rate /= sampling_rate.sum()
    sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                    min(len(sampling_rate),2048),p=sampling_rate,replace=False)]

    replay_input = np.array([stack_to_input(stacks[i],[j,k]) for i,j,k in sample_indices])
    replay_target = moves[tuple(sample_indices.T)]
    
    training_input.append(replay_input.astype(np.float32))
    training_target.append(replay_target.astype(np.float32))

now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
training_input = np.concatenate(training_input,axis=0)
training_target = np.concatenate(training_target,axis=0)
indices = np.arange(len(training_input))
np.random.shuffle(indices) #shuffle training samples
training_input = training_input[indices]
training_target = training_target[indices]

# reformat input for convolutional neural net (input needs to be need to be (# examples, 9, 9, 4))
training_input = np.array(training_input).reshape(len(training_input), (2*VISIBLE_DISTANCE+1), (2*VISIBLE_DISTANCE+1), 4)

model.fit(training_input,training_target,validation_split=0.1,
          callbacks=[EarlyStopping(patience=10),
                     ModelCheckpoint(model_name,verbose=1,save_best_only=True),
                     tensorboard],
          batch_size=1024, nb_epoch=100)

model = load_model(model_name)
# load previous model

still_mask = training_target[:,0].astype(bool)
print('STILL accuracy:',model.evaluate(training_input[still_mask],training_target[still_mask],verbose=0)[1])
print('MOVE accuracy:',model.evaluate(training_input[~still_mask],training_target[~still_mask],verbose=0)[1])




'''
tpot_training_target = []
import progressbar

bar = progressbar.ProgressBar()

for sample in bar(training_target):
    tpot_training_target.append([idx for idx, x in enumerate(sample.astype(bool)) if x][0])


from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=50, verbosity=2)
tpot.fit(training_input, tpot_training_target)

tpot.export('tpot_halite_pipeline.py')

'''