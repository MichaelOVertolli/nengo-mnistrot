import numpy as np
import nengo
import nengo_dl
import tensorflow as tf
from nengo_dl import tensor_layer
slim = tf.contrib.slim

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse, neuron_type, ens_params): 
		with tf.variable_scope("G", reuse=reuse):
				with nengo.Network() as net:
						nengo_dl.configure_settings(trainable=False)
						num_output=int(np.prod([7,7,hidden_num]))
						x=tensor_layer(z,layer_func=slim.fully_connected,num_outputs=num_output,activation_fn=None)#inpt=z in tensorflow code, used z here too
						#x=reshape2(x,7,7, hidden_num)

						for idx in range(1, repeat_num+1):
                                                                print idx
                                                                shape_in=(hidden_num, 7*idx, 7*idx)
								x = nengo_dl.tensor_layer(x, layer_func=slim.conv2d, shape_in=shape_in, num_outputs=hidden_num, kernel_size=3, 
												stride=1, activation_fn=None, data_format=data_format)
								x = nengo_dl.tensor_layer(x, neuron_type, **ens_params) #add these lines to the other functions
								x = tensor_layer(x, layer_func=slim.conv2d, shape_in=shape_in, num_outputs=hidden_num, kernel_size=3, 
                                                 stride=1, activation_fn=None, data_format=data_format)
								#if idx < repeat_num - 1:
								x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)
								x = upscale2(x,2,shape_in)


						out=tensor_layer(x,layer_func=slim.conv2d,shape_in=(hidden_num, 28, 28), num_outputs=1,kernel_size=3,stride=1,
                                         activation_fn=None,data_format=data_format)
		return net,out
	
def GeneratorRCNN(x,input_channel,z_num,repeat_num,hidden_num,data_format,neuron_type, ens_params):
	with nengo.Network() as net:
		nengo_dl.configure_settings(trainable=False)
		x = tensor_layer(x, layer_func=slim.conv2d, num_outputs=hidden_num, kernel_size=3,
		stride=1, activation_fn=None, normalizer_fn=slim.batch_norm,data_format=data_format)
		
		prev_channel_num=hidden_num
		for idx in range(repeat_num):
			channel_num=hidden_num * (idx+1)
			x=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,
				stride=1,activation_fn=None,normalizer_fn=slim.batch_norm,data_format=data_format)
			x = nengo_dl.tensor_layer(x, neuron_type, **ens_params) 
			x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,
				stride=1,activation_fn=None,normalizer_fn=slim.batch_norm,data_format=data_format)
			#if idx < repeat_num + 2:
			x = nengo_dl.tensor_layer(x, neuron_type, **ens_params) 
			x=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,
				stride=2,activation_fn=None,normalizer_fn=slim.batch_norm,data_format=data_format)
		
		x=reshape2(x,7,7, hidden_num)
		z = tensor_layer(x,layer_func=slim.fully_connected,num_outputs=z_num,activation_fn=None)#inpt=x in tensorflow code, used x here too
		#z = tf.nn.softsign(z)
		#z = tf.sigmoid(z) -- worse
			
	return net,z
		
def DiscriminatorCNN(x,input_channel,z_num,repeat_num,hidden_num,data_format,neuron_type, ens_params):
	with nengo.Network() as net:
		nengo_dl.configure_settings(trainable=False)
		
		x=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=hidden_num,kernel_size=3,stride=1,
		activation_fn=tf.nn.elu,data_format=data_format)

		prev_channel_num=hidden_num
		for idx in range(repeat_num):
			channel_num=hidden_num * (idx+1)
			x=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,stride=1,
				activation_fn=None,data_format=data_format)
			x = nengo_dl.tensor_layer(x, neuron_type, **ens_params) 
			x=tensor_layer(x,layer_func=sli.conv2d,num_outputs=channel_num,kernel_size=3,stride=1,
				activation_fn=None,data_format=data_format)
			if idx < repeat_num + 2:
				x = nengo_dl.tensor_layer(x, neuron_type, **ens_params) 
				x=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,stride=2,
					activation_fn=None,data_format=data_format)
				#x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
				
		x=reshape2(x,7,7, hidden_num)
		z=x=nengo_dl.tensor_layer(x,layer_func=slim.fully_connected,num_outputs=z,activation_fn=None)
		
		#decoder
		num_output=int(np.prod([7,7,channel_num]))
		x=nengo_dl.tensor_layer(x,layer_func=slim.fully_connected,num_outputs=num_output,activation_fn=None)
		x=reshape2(x,7,7, hidden_num)
		
		for idx in range(repeat_num):
			x=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=hidden_num,kernel_size=3,stride=1,
			activation_fn=None,data_format=data_format)
			x=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=hidden_num,kernel_size=3,stride=1,
			activation_fn=None,data_format=data_format)
			#if idx < repeat_num - 1:
			x = upscale2(x,2)
		
		out=nengo_dl.tensor_layer(x,layer_func=slim.conv2d,num_outputs=input_channel,kernel_size=3,stride=1,
		activation_fn=None,data_format=data_format)
	return net,out
	
def int_shape(tensor):
	shape=tensor.get_shape().as_list()
	#shape = nengo_dl.tensor_layer(tensor,layer_func=tf.shape,name=None,out_type=tf.int32)
	return [num if num is not None else -1 for num in shape]
 

def get_conv_shape(tensor, data_format):
	shape = int_shape(tensor)
	# always return [N, H, W, C]
	if data_format == 'NCHW':
		return [shape[0], shape[2], shape[3], shape[1]]
	elif data_format == 'NHWC':
		return shape

def nchw_to_nhwc(x):
	return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
	return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
	if data_format == 'NCHW':
		x = tf.reshape(x, [-1, c, h, w])
	else:
		x = tf.reshape(x, [-1, h, w, c])
	return x

def resize_nearest_neighbor(x, new_size, data_format):
	if data_format == 'NCHW':
		x = nchw_to_nhwc(x)
		x = tf.image.resize_nearest_neighbor(x, new_size)
		x = nhwc_to_nchw(x)
	else:
		x = tf.image.resize_nearest_neighbor(x, new_size)
	return x

def upscale(x, scale, data_format):
	_, h, w, _ = get_conv_shape(x, data_format)
	return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)
		
def upscale2(x, scale, shape_in):
	#shape = nengo_dl.tensor_layer(x,layer_func=tf.shape,shape_in=shape_in,name=None,out_type=tf.int32)
	#shape = [num if num is not None else -1 for num in shape]
        c = shape_in[0]
        h = shape_in[1]
	w = shape_in[2]
        shape_in_flipped = (h, w, c)
	x = nengo_dl.tensor_layer(x, layer_func=tf.transpose, shape_in=shape_in,perm=[0, 2, 3, 1])
	new_size = (h*scale,w*scale)
        
	x = nengo_dl.tensor_layer(x, layer_func=tf.image.resize_nearest_neighbor, shape_in=shape_in_flipped, size=new_size)
	x = nengo_dl.tensor_layer(x, layer_func=tf.transpose, shape_in=(h*scale, w*scale, shape_in[0]), perm=[0, 3, 1, 2])
	return x

def get_conv_shape2(tensor,data_format):
	shape = nengo_dl.tensor_layer(tensor,layer_func=tf.shape,name=None,out_type=tf.int32)
	shape = [num if num is not None else -1 for num in shape]
	if data_format == 'NCHW':
		return [shape[0], shape[2], shape[3], shape[1]]
	elif data_format == 'NHWC':
		return shape	
				
def reshape2(x, h, w, c):
	x = tensor_layer(x, layer_func=tf.reshape, shape=[-1, c, h, w])
	return x
				
				
				
				
				
				
				
				
				
				
				
				
				
				
			
