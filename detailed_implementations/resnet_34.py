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
		BatchNormalization,
		AveragePooling2D,
		Flatten,
		Dense,
		Add
		)

def _residual_block(inputs,filters,strides=1):
	# padding ='same' is important to keep the size of the feature maps the same
	# This way the ouput size only depends on the number of filters used
	x = Conv2D(filters=filters,kernel_size=(3,3), strides=strides,padding='same')(inputs)
	# print(x.shape,inputs.shape) if strides==2 else None
	x = BatchNormalization()(x)
	x = ReLU()(x)

	x = Conv2D(filters=filters,kernel_size=(3,3), strides=1,padding='same')(x)
	x = BatchNormalization()(x)

	if inputs.shape == x.shape:
		res = Add()([x,inputs])
	else:
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
	print('conv1',x.shape)
	x = MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x) 
	x = ReLU()(x)


	#conv2_x
	for _ in range(3):
		x = _residual_block(x,filters=64)
	print('conv2_x',x.shape)

	#conv3_x
	for i in range(4):
		if i == 0:
			x = _residual_block(x,filters=128,strides=2)
		else:
			x = _residual_block(x,filters=128)
	print('conv3_x',x.shape)

	#conv4_x
	for i in range(6):
		if i == 0:
			x = _residual_block(x,filters=256,strides=2)
		else:
			x = _residual_block(x,filters=256)
	print('conv4_x',x.shape)

	#conv5_x
	for i in range(3):
		if i == 0:
			x = _residual_block(x,filters=512,strides=2)
		else:
			x = _residual_block(x,filters=512)
	print('conv5_x',x.shape)
	
	x = AveragePooling2D()(x) # defaults to pool_size=strides=(2, 2),
	x = Flatten()(x)
	x = Dense(units=num_class)(x)
	return Model(inputs=input_,outputs=x,name='resnet_34')

if __name__ == '__main__':
	model = get_model((224,224,3))
	model.summary()
