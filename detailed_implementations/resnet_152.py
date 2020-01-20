#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 yuengdelahoz <yuengdelahoz@tensorbook>
#
# Distributed under terms of the MIT license.

"""
ResNet-50 Implementation

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

def _residual_block_identity(inputs,filters,strides=1): # bottleneck approach
	# padding ='same' is important to keep the size of the feature maps the same
	# This way the ouput size only depends on the number of filters used
	x = Conv2D(filters=filters,kernel_size=(1,1), strides=strides,padding='same')(inputs)
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=filters,kernel_size=(3,3), strides=1,padding='same')(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=(filters * 4),kernel_size=(1,1), strides=1,padding='same')(x)
	x = BatchNormalization()(x)

	res = Add()([x,inputs])
	return x

def _residual_block_shortcut(inputs,filters,strides=1): # bottleneck approach
	# padding ='same' is important to keep the size of the feature maps the same
	# This way the ouput size only depends on the number of filters used
	x = Conv2D(filters=filters,kernel_size=(1,1), strides=strides,padding='same')(inputs)
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=filters,kernel_size=(3,3), strides=1,padding='same')(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=(filters * 4),kernel_size=(1,1), strides=1,padding='same')(x)
	x = BatchNormalization()(x)

	# apply projection to make input shape match output shape
	res = Conv2D(filters=filters * 4,kernel_size=(1,1), strides=strides,padding='same')(inputs)
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
	# shortcut x 1
	x = _residual_block_shortcut(x,filters=64)
	# identity x 2
	x = _residual_block_identity(x,filters=64)
	x = _residual_block_identity(x,filters=64)
	print('conv2_x output size',x.shape)

	#conv3_x - stage 2 (4)
	# shortcut x 1
	x = _residual_block_shortcut(x,filters=128,strides=2)
	# identity x 8
	x = _residual_block_identity(x,filters=128) #1
	x = _residual_block_identity(x,filters=128) #2
	x = _residual_block_identity(x,filters=128) #3
	x = _residual_block_identity(x,filters=128) #4
	x = _residual_block_identity(x,filters=128) #5
	x = _residual_block_identity(x,filters=128) #6
	x = _residual_block_identity(x,filters=128) #7
	print('conv3_x output size',x.shape)

	#conv4_x - stage 3 (36)
	# shortcut x 1
	x = _residual_block_shortcut(x,filters=256,strides=2)
	# identity x 22
	x = _residual_block_identity(x,filters=256) #1
	x = _residual_block_identity(x,filters=256) #2
	x = _residual_block_identity(x,filters=256) #3
	x = _residual_block_identity(x,filters=256) #4
	x = _residual_block_identity(x,filters=256) #5
	x = _residual_block_identity(x,filters=256) #6
	x = _residual_block_identity(x,filters=256) #7
	x = _residual_block_identity(x,filters=256) #8
	x = _residual_block_identity(x,filters=256) #9
	x = _residual_block_identity(x,filters=256) #10
	x = _residual_block_identity(x,filters=256) #11
	x = _residual_block_identity(x,filters=256) #12
	x = _residual_block_identity(x,filters=256) #13
	x = _residual_block_identity(x,filters=256) #14
	x = _residual_block_identity(x,filters=256) #15
	x = _residual_block_identity(x,filters=256) #16
	x = _residual_block_identity(x,filters=256) #17
	x = _residual_block_identity(x,filters=256) #18
	x = _residual_block_identity(x,filters=256) #19
	x = _residual_block_identity(x,filters=256) #20
	x = _residual_block_identity(x,filters=256) #21
	x = _residual_block_identity(x,filters=256) #22
	x = _residual_block_identity(x,filters=256) #23
	x = _residual_block_identity(x,filters=256) #24
	x = _residual_block_identity(x,filters=256) #25
	x = _residual_block_identity(x,filters=256) #26
	x = _residual_block_identity(x,filters=256) #27
	x = _residual_block_identity(x,filters=256) #28
	x = _residual_block_identity(x,filters=256) #29
	x = _residual_block_identity(x,filters=256) #30
	x = _residual_block_identity(x,filters=256) #31
	x = _residual_block_identity(x,filters=256) #32
	x = _residual_block_identity(x,filters=256) #33
	x = _residual_block_identity(x,filters=256) #34
	x = _residual_block_identity(x,filters=256) #35
	print('conv4_x output size',x.shape)

	#conv5_x - stage 4 (3)
	# shortcut x 1
	x = _residual_block_shortcut(x,filters=512,strides=2)
	# identity x 2
	x = _residual_block_identity(x,filters=512)
	x = _residual_block_identity(x,filters=512)
	print('conv5_x output size',x.shape)
	
	x = GlobalAveragePooling2D()(x)

	'''
	Total params: 58,370,944
	Trainable params: 58,219,520
	Non-trainable params: 151,424
	'''

	x = Flatten()(x)
	x = Dense(units=num_class)(x)
	x = Softmax()(x)
	'''
	Total params: 60,419,944
	Trainable params: 60,268,520
	Non-trainable params: 151,424
	'''
	return Model(inputs=input_,outputs=x,name='resnet_152')

if __name__ == '__main__':
	import os
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	model = get_model((224,224,3))
	model.summary()
