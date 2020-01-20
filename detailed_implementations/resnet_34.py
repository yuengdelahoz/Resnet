#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 yuengdelahoz <yuengdelahoz@tensorbook>
#
# Distributed under terms of the MIT license.

"""
ResNet-34 Implementation

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
		GlobalAveragePooling2D,
		Flatten,
		Dense,
		Add,
		Softmax
		)

def _residual_block_identity(inputs,filters,strides=1):
	# padding ='same' is important to keep the size of the feature maps the same
	# This way the ouput size only depends on the number of filters used
	x = Conv2D(filters=filters,kernel_size=(3,3), strides=strides,padding='same')(inputs)
	# print(x.shape,inputs.shape) if strides==2 else None
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=filters,kernel_size=(3,3), strides=1,padding='same')(x)
	x = BatchNormalization()(x)

	res = Add()([x,inputs])
	x = ReLU()(res)
	return x

def _residual_block_shortcut(inputs,filters,strides=1):
	# padding ='same' is important to keep the size of the feature maps the same
	# This way the ouput size only depends on the number of filters used
	x = Conv2D(filters=filters,kernel_size=(3,3), strides=strides,padding='same')(inputs)
	# print(x.shape,inputs.shape) if strides==2 else None
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=filters,kernel_size=(3,3), strides=1,padding='same')(x)
	x = BatchNormalization()(x)

	# apply projection to make input shape match output shape
	res = Conv2D(filters=filters,kernel_size=(1,1), strides=strides,padding='same')(inputs)
	res = BatchNormalization()(res)
	res = Add()([x,res])
	x = ReLU()(res)
	return x

def get_model(input_shape,num_class=1000):
	input_ = Input(shape=input_shape)
	#conv1
	x = Conv2D(filters=64,kernel_size=(7,7), strides=2,padding='same')(input_)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	print('conv1_x output size',x.shape)

	x = MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x) 

	#conv2_x - stage 1 (3)
	x = _residual_block_identity(x,filters=64)
	x = _residual_block_identity(x,filters=64)
	x = _residual_block_identity(x,filters=64)
	print('conv2_x output size',x.shape)


	#conv3_x - stage 2 (4)
	x = _residual_block_shortcut(x,filters=128,strides=2)
	x = _residual_block_identity(x,filters=128)
	x = _residual_block_identity(x,filters=128)
	x = _residual_block_identity(x,filters=128)
	print('conv3_x output size',x.shape)

	#conv4_x - stage 3 (6)
	x = _residual_block_shortcut(x,filters=256,strides=2)
	x = _residual_block_identity(x,filters=256)
	x = _residual_block_identity(x,filters=256)
	x = _residual_block_identity(x,filters=256)
	x = _residual_block_identity(x,filters=256)
	x = _residual_block_identity(x,filters=256)
	print('conv4_x output size',x.shape)

	#conv5_x - stage 4 (3)
	x = _residual_block_shortcut(x,filters=512,strides=2)
	x = _residual_block_identity(x,filters=512)
	x = _residual_block_identity(x,filters=512)
	print('conv5_x output size',x.shape)
	
	x = GlobalAveragePooling2D()(x)
	return Model(inputs=input_,outputs=x,name='resnet_34')
	'''
	After GlobalAveragePooling2D:
	Total params: 21,310,208
	Trainable params: 21,293,184
	Non-trainable params: 17,024
	'''
	x = Flatten()(x)
	x = Dense(units=num_class)(x)
	x = Softmax()(x)
	'''
	After Softmax:
	Total params: 21,823,208
	Trainable params: 21,806,184
	Non-trainable params: 17,024
	'''
	return Model(inputs=input_,outputs=x,name='resnet_34')

if __name__ == '__main__':
	import os
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	model = get_model((224,224,3))
	model.summary()
