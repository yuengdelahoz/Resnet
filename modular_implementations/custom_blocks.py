#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 yuengdelahoz <yuengdelahoz@tensorbook>
#
# Distributed under terms of the MIT license.

"""
ResNet backbone

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
		Layer
		)

import tensorflow as tf
class BasicBlock(Layer):
	def __init__(self,filters,name,strides=1, is_shortcut=False):
		super(BasicBlock, self).__init__()
		self.filters = filters
		self.strides = strides
		self.is_shortcut = is_shortcut
		self._name = name
	
	def build(self,input_shape):
		with tf.name_scope(self._name) as scope:
			self.conv1 = Conv2D(filters=self.filters,kernel_size=(3,3), strides=self.strides,padding='same')
			self.bn = BatchNormalization()

			self.conv2 = Conv2D(filters=self.filters,kernel_size=(3,3), strides=1,padding='same')
			self.bn2 = BatchNormalization()

			# Projection shortcut
			if self.is_shortcut:
				self.projection = Conv2D(filters=self.filters,kernel_size=(1,1), strides=self.strides,padding='same')
				self.bn3 = BatchNormalization()

			self.activation = ReLU()
			self.add = Add()

	def call(self,inputs):
		x = self.conv1(inputs)
		x = self.bn(x)
		x = self.activation(x)

		x = self.conv2(x)
		x = self.bn2(x)

		if inputs.shape.as_list() == x.shape.as_list():
			res = self.add([x,inputs])
		else:
			res = self.projection(inputs)
			res = self.bn3(res)
			res = self.add([x,res])
		x = self.activation(res)
		return x

	def get_config(self):
		config = super(BasicBlock, self).get_config()
		config.update({
			'filters':self.filters,
			'strides ':self.strides,
			'_name':self._name,
			'is_shortcut':self.is_shortcut
			})
		return config

class BottleNeck(Layer):
	def __init__(self, filters, name, strides=1,is_shortcut=False):
		super(BottleNeck, self).__init__()
		self.filters = filters
		self.strides = strides
		self._name = name
		self.is_shortcut = is_shortcut
	
	def build(self,input_shape):
		with tf.name_scope(self._name) as scope:
			self.conv1 = Conv2D(filters=self.filters, kernel_size=(1,1), strides=self.strides,padding='same')
			self.bn = BatchNormalization()

			self.conv2 = Conv2D(filters=self.filters, kernel_size=(3,3), strides=1,padding='same')
			self.bn2 = BatchNormalization()

			self.conv3 = Conv2D(filters=self.filters * 4,kernel_size=(1,1), strides=1,padding='same')
			self.bn3 = BatchNormalization()

			# Projection shortcut
			if self.is_shortcut:
				self.projection = Conv2D(filters=self.filters*4,kernel_size=(1,1), strides=self.strides,padding='same')
				self.bn4 = BatchNormalization()

			self.add = Add()
			self.activation = ReLU()

	def call(self,inputs):
		x = self.conv1(inputs)
		x = self.bn(x)
		x = self.activation(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.activation(x)

		x = self.conv3(x)
		x = self.bn3(x)

		if inputs.shape.as_list() == x.shape.as_list():
			res = self.add([x,inputs])
		else:
			res = self.projection(inputs)
			res = self.bn4(res)
			res = self.add([x,res])
		x = self.activation(res)
		return x

	def get_config(self):
		config = super(BottleNeck, self).get_config()
		config.update({
			'filters':self.filters,
			'strides ':self.strides,
			'_name':self._name,
			'is_shortcut':self.is_shortcut
			})
		return config
