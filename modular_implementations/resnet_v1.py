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
		BatchNormalization,
		Flatten,
		Dense,
		Softmax
		)

from custom_blocks import BasicBlock, BottleNeck
import numpy as np

def _get_resnet(input_shape, ResidualLayer, num_class, name, stages):
	input_ = Input(shape=input_shape)
	#conv1
	x = Conv2D(filters=64,kernel_size=(7,7), strides=2,padding='same')(input_)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	print('conv1 output size',x.shape)
	x = MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x) 

	#conv2_x
	for i in range(stages[0]):
		if i == 0 and not isinstance(ResidualLayer,BasicBlock):
			x = ResidualLayer(filters=64,name='stage_1_{}'.format(i),is_shortcut=True)(x)
		else:
			x = ResidualLayer(filters=64,name='stage_1_{}'.format(i))(x)
	print('conv2_x output size',x.shape)

	#conv3_x
	for i in range(stages[1]):
		if i == 0:
			x = ResidualLayer(filters=128,strides=2, is_shortcut=True,name='stage_2_{}'.format(i))(x)
		else:
			x = ResidualLayer(filters=128,name='stage_2_{}'.format(i))(x)
	print('conv3_x output size',x.shape)

	#conv4_x
	for i in range(stages[2]):
		if i == 0:
			x = ResidualLayer(filters=256,strides=2, is_shortcut=True,name='stage_3_{}'.format(i))(x)
		else:
			x = ResidualLayer(filters=256,name='stage_3_{}'.format(i))(x)
	print('conv4_x output size',x.shape)

	#conv5_x
	for i in range(stages[3]):
		if i == 0:
			x = ResidualLayer(filters=512,strides=2, is_shortcut=True,name='stage_4_{}'.format(i))(x)
		else:
			x = ResidualLayer(filters=512,name='stage_4_{}'.format(i))(x)
	print('conv5_x output size',x.shape)
	
	x = GlobalAveragePooling2D()(x)
	return Model(inputs=input_,outputs=x,name=name)

	x = Flatten()(x)
	x = Dense(units=num_class)(x)
	x = Softmax()(x)
	return Model(inputs=input_,outputs=x,name=name)

def get_resnet_34(input_shape, num_class=1000):
	'''
	After GlobalAveragePooling2D:
	Total params: 21,310,208
	Trainable params: 21,293,184
	Non-trainable params: 17,024

	After Softmax:
	Total params: 21,823,208
	Trainable params: 21,806,184
	Non-trainable params: 17,024

	'''
	return _get_resnet(input_shape,BasicBlock,num_class,name='resnet_34',stages=(3,4,6,3))

def get_resnet_50(input_shape, num_class=1000):
	'''
	After GlobalAveragePooling2D:
	Total params: 23,587,712
	Trainable params: 23,534,592
	Non-trainable params: 53,120

	After Softmax:
	Total params: 25,636,712
	Trainable params: 25,583,592
	Non-trainable params: 53,120
	'''
	return _get_resnet(input_shape,BottleNeck,num_class,name='resnet_50',stages=(3,4,6,3))

def get_resnet_101(input_shape, num_class=1000):
	'''
	After GlobalAveragePooling2D:
	Total params: 42,658,176
	Trainable params: 42,552,832
	Non-trainable params: 105,344


	After Softmax:
	Total params: 44,707,176
	Trainable params: 44,601,832
	Non-trainable params: 105,344
	'''
	return _get_resnet(input_shape,BottleNeck,num_class,name='resnet_101',stages=(3,4,23,3))

def get_resnet_152(input_shape, num_class=1000):
	'''
	After GlobalAveragePooling2D:
	Total params: 58,370,944
	Trainable params: 58,219,520
	Non-trainable params: 151,424


	After Softmax:
	Total params: 60,419,944
	Trainable params: 60,268,520
	Non-trainable params: 151,424
	'''
	return _get_resnet(input_shape,BottleNeck,num_class,name='resnet_152',stages=(3,8,36,3))

if __name__ == '__main__':
	import os
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	model = get_resnet_34(input_shape=(224,224,3))
	# model = get_resnet_50(input_shape=(224,224,3))
	# model = get_resnet_101(input_shape=(224,224,3))
	# model = get_resnet_152(input_shape=(224,224,3))
	model.summary()


