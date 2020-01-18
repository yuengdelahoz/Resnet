#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 yuengdelahoz <yuengdelahoz@tensorbook>
#
# Distributed under terms of the MIT license.

"""
ResNet-101 Implementation

"""

from tensorflow.keras import (
		Input,
		Model
		)

from tensorflow.keras.layers import (
		Conv2D,
		MaxPool2D,
		ReLU,
		BatchNormalization,
		AveragePooling2D,
		Flatten,
		Dense,
		Add,
		Softmax
		)

def _residual_block(inputs,filters,strides=1): # bottleneck approach
	# padding ='same' is important to keep the size of the feature maps the same
	# This way the ouput size only depends on the number of filters used
	cnt = 0
	x = Conv2D(filters=filters,kernel_size=(1,1), strides=strides,padding='same')(inputs)
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=filters,kernel_size=(3,3), strides=1,padding='same')(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=(filters * 4),kernel_size=(1,1), strides=1,padding='same')(x)
	x = BatchNormalization()(x)

	if inputs.shape == x.shape:
		res = Add()([x,inputs])
	else:
		# apply projection to make input shape match output shape
		res = Conv2D(filters=filters * 4,kernel_size=(1,1), strides=strides,padding='same')(inputs)
		res = BatchNormalization()(res)
		res = Add()([x,res])
	x = ReLU()(res)
	return x

def get_model(input_shape,num_class=1000):
	num_layers = 0
	input_ = Input(shape=input_shape)
	#conv1
	x = Conv2D(filters=64,kernel_size=(7,7), strides=2,padding='same')(input_)
	x = MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x) 
	x = ReLU()(x)
	num_layers += 1
	print('conv1',x.shape,'num_layers',num_layers)


	#conv2_x
	for _ in range(3):
		x = _residual_block(x,filters=64)
		num_layers += 3
	print('conv2_x',x.shape,'num_layers',num_layers)

	#conv3_x
	for i in range(4):
		num_layers += 3
		if i == 0:
			x = _residual_block(x,filters=128,strides=2)
		else:
			x = _residual_block(x,filters=128)
	print('conv3_x',x.shape,'num_layers',num_layers)

	#conv4_x
	for i in range(23):
		num_layers += 3
		if i == 0:
			x = _residual_block(x,filters=256,strides=2)
		else:
			x = _residual_block(x,filters=256)
	print('conv4_x',x.shape,'num_layers',num_layers)

	#conv5_x
	for i in range(3):
		num_layers += 3
		if i == 0:
			x = _residual_block(x,filters=512,strides=2)
		else:
			x = _residual_block(x,filters=512)
	print('conv5_x',x.shape,'num_layers',num_layers)
	
	x = AveragePooling2D()(x) # defaults to pool_size=strides=(2, 2),
	x = Flatten()(x)
	x = Dense(units=num_class)(x)
	x = Softmax()(x)
	num_layers += 1
	print('total number of layers:', num_layers)
	return Model(inputs=input_,outputs=x,name='resnet_101')

if __name__ == '__main__':
	import os
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	model = get_model((224,224,3))
	# model.summary()
