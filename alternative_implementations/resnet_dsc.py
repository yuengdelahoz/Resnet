#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 yuengdelahoz <yuengdelahoz@tensorbook>
#
# Distributed under terms of the MIT license.

"""

"""
from tensorflow.keras import (
		Input,
		Model
		)

from tensorflow.keras.layers import (
		Conv2D,
		MaxPool2D,
		ReLU,
		GlobalAveragePooling2D,
		Flatten,
		Dense,
		SeparableConv2D,
		Softmax,
		BatchNormalization,
		Concatenate
		)

from custom_blocks import BasicBlockDSC, BottleNeckDSC
import numpy as np
from time import time

def _get_resnet(input_shape, ResidualLayer, num_class, name, stages):
	input_ = Input(shape=input_shape)
	#conv1
	t1 = time()
	x = SeparableConv2D(filters=64,kernel_size=(7,7),strides=2,padding='same')(input_)
	print('conv1 output size', x.shape,'time:',time()-t1)
	x = BatchNormalization()(x) # mine
	x = MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x) 
	x = ReLU()(x)

	#conv2_x
	t1 = time()
	for _ in range(stages[0]):
		x = ResidualLayer(filters=64)(x)
	print('conv2_x output size',x.shape,'time:',time()-t1)

	#conv3_x
	t1 = time()
	for i in range(stages[1]):
		if i == 0:
			x = ResidualLayer(filters=128,strides=2)(x)
		else:
			x = ResidualLayer(filters=128)(x)
	print('conv3_x output size',x.shape,'time:',time()-t1)

	#conv4_x
	t1 = time()
	for i in range(stages[2]):
		if i == 0:
			x = ResidualLayer(filters=256,strides=2)(x)
		else:
			x = ResidualLayer(filters=256)(x)
	print('conv4_x output size',x.shape,'time:',time()-t1)

	#conv5_x
	t1 = time()
	for i in range(stages[3]):
		if i == 0:
			x = ResidualLayer(filters=512,strides=2)(x)
		else:
			x = ResidualLayer(filters=512)(x)
	print('conv5_x output size',x.shape,'time:',time()-t1)
	
	x = GlobalAveragePooling2D()(x)
	x = Flatten()(x)
	x = Dense(units=num_class)(x)
	x = Softmax()(x)
	return Model(inputs=input_,outputs=x,name=name)

def get_resnet_34(input_shape, num_class=1000):
	return _get_resnet(input_shape,BasicBlockDSC,num_class,name='resnet_dsc_34',stages=(3,4,6,3))

def get_resnet_50(input_shape, num_class=1000):
	return _get_resnet(input_shape,BottleNeckDSC,num_class,name='resnet_dsc_50',stages=(3,4,6,3))

def get_resnet_101(input_shape, num_class=1000):
	return _get_resnet(input_shape,BottleNeckDSC,num_class,name='resnet_dsc_101',stages=(3,4,23,3))

def get_resnet_152(input_shape, num_class=1000):
	return _get_resnet(input_shape,BottleNeckDSC,num_class,name='resnet_dsc_152',stages=(3,8,36,3))

if __name__ == '__main__':
	import os
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	model = get_resnet_34(input_shape=(224,224,3))
	model.summary()
	model = get_resnet_50(input_shape=(224,224,3))
	model.summary()
	model = get_resnet_101(input_shape=(224,224,3))
	model.summary()
	model = get_resnet_152(input_shape=(224,224,3))
	model.summary()

