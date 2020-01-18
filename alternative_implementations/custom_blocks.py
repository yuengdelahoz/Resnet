#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 yuengdelahoz <yuengdelahoz@tensorbook>
#
# Distributed under terms of the MIT license.

"""
Custom Layers for Resnet architecture using Depth Separable Convolution Layers

"""

from tensorflow.keras.layers import (
		Conv2D,
		MaxPool2D,
		ReLU,
		BatchNormalization,
		AveragePooling2D,
		Flatten,
		Dense,
		Add,
		Softmax,
		SeparableConv2D,
		Concatenate,
		Layer
		)

from tensorflow.keras import Input

class BasicBlockDSC(Layer):
	def __init__(self,filters,strides=1):
		super(BasicBlockDSC, self).__init__()
		self.filters = filters
		self.strides = strides
	
	def build(self,input_shape):
		self.sepconv1 = SeparableConv2D(filters=self.filters,kernel_size=(3,3),strides=self.strides,padding='same')
		self.bn = BatchNormalization()

		self.sepconv2 = SeparableConv2D(filters=self.filters,kernel_size=(3,3),padding='same')
		self.bn2 = BatchNormalization()

		# Projection shortcut
		self.projection = SeparableConv2D(filters=self.filters,kernel_size=(1,1),strides=self.strides,padding='same')
		self.bn3 = BatchNormalization()

		self.activation = ReLU()
		self.add = Add()
		self.concatenate = Concatenate()

	def call(self,inputs):
		x = self.sepconv1(inputs)
		x = self.bn(x)
		x = self.activation(x)

		x = self.sepconv2(x)
		x = self.bn2(x)

		if x.shape == inputs.shape:
			res = self.add([x,inputs])
		else:
			res = self.projection(inputs)
			res = self.bn3(res)
			res = self.add([x,res])
		return self.activation(res)

	def get_config(self):
		config = super(BasicBlockDSC, self).get_config()
		config.update({
			'filters':self.filters,
			'strides ':self.strides,
			})
		return config

class BottleNeckDSC(Layer):
	def __init__(self, filters, strides=1):
		super(BottleNeckDSC, self).__init__()
		self.filters = filters
		self.strides = strides
	
	def build(self,input_shape):
		self.sepconv1 = SeparableConv2D(filters=self.filters,kernel_size=(1,1),strides=self.strides,padding='same')
		self.bn = BatchNormalization()

		self.sepconv2 = SeparableConv2D(filters=self.filters,kernel_size=(3,3),padding='same')
		self.bn2 = BatchNormalization()

		self.sepconv3 = SeparableConv2D(filters=self.filters * 4,kernel_size=(1,1),padding='same')
		self.bn3 = BatchNormalization()

		# Projection shortcut
		self.projection = SeparableConv2D(filters=self.filters * 4,kernel_size=(1,1),strides=self.strides,padding='same')
		self.bn4 = BatchNormalization()

		self.add = Add()
		self.activation = ReLU()

	def call(self,inputs):
		x = self.sepconv1(inputs)
		x = self.bn(x)
		x = self.activation(x)

		x = self.sepconv2(x)
		x = self.bn2(x)
		x = self.activation(x)

		x = self.sepconv3(x)
		x = self.bn3(x)

		if inputs.shape == x.shape:
			res = self.add([x,inputs])
		else:
			res = self.projection(inputs)
			res = self.bn4(res)
			res = self.add([x,res])
		return self.activation(res)

	def get_config(self):
		config = super(BottleNeckDSC, self).get_config()
		config.update({
			'filters':self.filters,
			'strides ':self.strides,
			})
		return config
