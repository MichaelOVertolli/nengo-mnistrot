import numpy as np
import tensorflow as tf
import nengo
import nengo_dl

from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config
from models import GeneratorCNN, GeneratorRCNN, DiscriminatorCNN

def main(config):
	prepare_dirs_and_logger(config)
        batch_size = config.batch_size
        z_num = config.z_num
        hidden_num = config.conv_hidden_num
        #do the rest of the importing... look at trainer.py in the __init__ and the code below for which vars

        z = tf.random_uniform(batch_size, z_num)

        neuron_type = nengo_dl.LIF(tau_rc=0.02, tau_ref=0.002)#, sigma=0.002)
        ens_params = dict(max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0]))

        with nengo.Network() as net:
                inpt = nengo.Node(nengo.processes.PresentInput(z, 0.1))
                G, G_out = GeneratorCNN(inpt, hidden_num, channel, repeat_num, reuse=False, data_format=data_format, neuron_type, ens_params)
                G_r, G_r_out = GeneratorRCNN(G_out, channel, z_num, repeat_num, hidden_num, data_format=data_format, neuron_type, ens_params)
                G2, G2_out = GeneratorCNN(G_r_out, hidden_num, channel, repeat_num, reuse=True, data_format=data_format, neuron_type, ens_params)
                D, D_out = DiscriminatorCNN(G_out, channel, z_num, repeat_num, hidden_num, data_format=data_format, neuron_type, ens_params)


        with net:
                out_p = nengo.Probe(out)

        sim = nengo_dl.Simulator(net, minimbatch_size=batch_size)
        #sim.load_params("path/to/zipfolderthatIsent")
        sim.close()

if __name__ == "__main__":
	config, unparsed = get_config()
	main(config)
