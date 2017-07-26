import numpy as np
import nengo
import nengo_dl
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
slim = tf.contrib.slim

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse, neuron_type, ens_params): #add the last two parameters to the other functions
        with tf.variable_scope("G", reuse=reuse):
                with nengo.Network() as net:
                        nengo_dl.configure_settings(trainable=False)
                        num_output=int(np.prod[7,7,hidden_num])
                        x=tensor_layer(z,layer_func=slim.fully_connected,num_outputs=num_output,activation_fn=None)#inpt=z in tensorflow code, used z here too
                        x=tensor_layer(z,layer_func=reshape,h=7,w=7,c=hidden_num,data_format=data_format)

                        for idx in range(repeat_num):
                                x = tensor_layer(x, layer_func=slim.conv2d, num_outputs=hidden_num, kernel_size=3, 
                                                 stride=1, activation_fn=None, data_format=data_format)
                                x = nengo_dl.tensor_layer(x, neuron_type, **ens_params) #add these lines to the other functions
                                x = tensor_layer(x, layer_func=slim.conv2d, num_outputs=hidden_num, kernel_size=3, 
                                                 stride=1, activation_fn=None, data_format=data_format)
                                #if idx < repeat_num - 1:
                                x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)
                                x = tensor_layer(x,layer_func=upscale,scale=2,data_format=data_format)


                        out=tensor_layer(x,layer_func=slim.conv2d,num_outputs=1,kernel_size=3,stride=1,
                                         activation_fn=None,data_format=data_format)
	return net, out
	
def GeneratorRCNN(x,input_channel,z_num,repeat_num,hidden_num,data_format):
	with nengo.Network() as net:
		nengo_dl.configure_settings(trainable=False)
		x = tensor_layer(x, layer_func=slim.conv2d, num_outputs=hidden_num, kernel_size=3,
		stride=1, activation_fn=None, normalizer_fn=slim.batch_norm,data_format=data_format)
		
		prev_channel_num=hidden_num
		for idx in range(repeat_num):
			channel_num=hidden_num * (idx+1)
			x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,
			stride=1,activation_fn=None,normalizer_fn=slim.batch_norm,data_format=data_format)
			x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,
			stride=1,activation_fn=None,normalizer_fn=slim.batch_norm,data_format=data_format)
			#if idx < repeat_num + 2:
			x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,
			stride=2,activation_fn=None,normalizer_fn=slim.batch_norm,data_format=data_format)
		
		x = tensor_layer(x,layer_func=reshape,shape=[-1,np.prod([7,7,channel_num])],data_format=data_format)
		z = tensor_layer(x,layer_func=slim.fully_connected,num_outputs=z_num,activation_fn=None)#inpt=x in tensorflow code, used x here too
		#z = tf.nn.softsign(z)
		#z = tf.sigmoid(z) -- worse
			
	return net,z
		
def DiscriminatorCNN(x,input_channel,z_num,repeat_num,hidden_num,data_format):
	with nengo.Network() as net:
		nengo_dl.configure_settings(trainable=False)
		
		x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=hidden_num,kernel_size=3,stride=1,
		activation_fn=tf.nn.elu,data_format=data_format)

		prev_channel_num=hidden_num
		for idx in range(repeat_num)
			channel_num=hidden_num * (idx+1)
			x=tensor_layer(x,layer_func,num_outputs=channel_num,kernel_size=3,stride=1,
			activation_fn=None,data_format=data_format)
			x=tensor_layer(x,layer_func,num_outputs=channel_num,kernel_size=3,stride=1,
			activation_fn=None,data_format=data_format)
			if idx < repeat_num + 2:
				x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=channel_num,kernel_size=3,stride=2,
				activation_fn=None,data_format=data_format)
				#x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
				
		x=tensor_layer(x,layer_func=reshape,shape=[-1, np.prod([7,7,channel_num])],data_format=data_format)#figure this out
		z=x=tensor_layer(x,layer_func=slim.fully_connected,num_outputs=z,activation_fn=None)
		
		#decoder
		num_output=int(np.prod([7,7,channel_num]))
		x=tensor_layer(x,layer_func=slim.fully_connected,num_outputs=num_output,activation_fn=None)
		x=tensor_layer(x,layer_func=reshape,h=7,w=7,c=hidden_num,data_format=data_format)
		
		for idx in range(repeat_num):
			x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=hidden_num,kernel_size=3,stride=1,
			activation_fn=None,data_format=data_format)
			x=tensor_layer(x,layer_func=slim.conv2d,num_outputs=hidden_num,kernel_size=3,stride=1,
			activation_fn=None,data_format=data_format)
			#if idx < repeat_num - 1:
			x = tensor_layer(x,layer_func=upscale,scale=2,data_format=data_format)
		
		out=tensor_layer(x,layer_func=slim.conv2d,num_outputs=input_channel,kernel_size=3,stride=1,
		activation_fn=None,data_format=data_format)
	return net,out
	
def int_shape(tensor):
	shape = tensor.get_shape().as_list()
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
		
		
			
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
			