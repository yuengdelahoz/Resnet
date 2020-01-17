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
		Flatten
		Dense,
		Add
		)

def _residual_block(inputs):
	x = ReLU()(x)

def get_model(shape):
	input_ = Input(shape=shape)
	x = Conv2D(filters=64,kernel_size=(7,7), strides=(2,2))(input_)
	x = MaxPool2D(strides=(2,2))(x) 
	x = ReLU()(x)
