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
		Layer
		)

import tensorflow as tf
class BasicBlockDSC(Layer):
	def __init__(self,filters,name,strides=1,is_shortcut=False):
		super(BasicBlockDSC, self).__init__()
		self.filters = filters
		self.strides = strides
		self.is_shortcut = is_shortcut
		self._name = name
	
	def build(self,input_shape):
		with tf.name_scope(self._name) as scope:
			self.sepconv1 = SeparableConv2D(filters=self.filters,kernel_size=(3,3),strides=self.strides,padding='same')
			self.bn = BatchNormalization()

			self.sepconv2 = SeparableConv2D(filters=self.filters,kernel_size=(3,3),padding='same')
			self.bn2 = BatchNormalization()

			# Projection shortcut
			if self.is_shortcut:
				self.projection = SeparableConv2D(filters=self.filters,kernel_size=(1,1),strides=self.strides,padding='same')
				self.bn3 = BatchNormalization()

			self.activation = ReLU()
			self.add = Add()

	def call(self,inputs):
		x = self.sepconv1(inputs)
		x = self.bn(x)
		x = self.activation(x)

		x = self.sepconv2(x)
		x = self.bn2(x)

		if x.shape.as_list() == inputs.shape.as_list():
			print('identity')
			res = self.add([x,inputs])
		else:
			print('shortcut')
			res = self.projection(inputs)
			res = self.bn3(res)
			res = self.add([x,res])
		return self.activation(res)

	def get_config(self):
		config = super(BasicBlockDSC, self).get_config()
		config.update({
			'filters':self.filters,
			'strides ':self.strides,
			'_name':self._name,
			'is_shortcut':self.is_shortcut
			})
		return config

class BottleNeckDSC(Layer):
	def __init__(self, filters, name, strides=1, is_shortcut=False):
		super(BottleNeckDSC, self).__init__()
		self.filters = filters
		self.strides = strides
		self.is_shortcut = is_shortcut
		self._name = name
	
	def build(self,input_shape):
		with tf.name_scope(self._name) as scope:
			self.sepconv1 = SeparableConv2D(filters=self.filters,kernel_size=(1,1),strides=self.strides,padding='same')
			self.bn = BatchNormalization()

			self.sepconv2 = SeparableConv2D(filters=self.filters,kernel_size=(3,3),padding='same')
			self.bn2 = BatchNormalization()

			self.sepconv3 = SeparableConv2D(filters=self.filters * 4,kernel_size=(1,1),padding='same')
			self.bn3 = BatchNormalization()

			# Projection shortcut
			if self.is_shortcut:
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

		if inputs.shape.as_list() == x.shape.as_list():
			# print('identity')
			res = self.add([x,inputs])
		else:
			# print('shortcut')
			res = self.projection(inputs)
			res = self.bn4(res)
			res = self.add([x,res])
		return self.activation(res)

	def get_config(self):
		config = super(BottleNeckDSC, self).get_config()
		config.update({
			'filters':self.filters,
			'strides ':self.strides,
			'_name':self._name,
			'is_shortcut':self.is_shortcut
			})
		return config
