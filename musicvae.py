print 'Importing libraries and defining some helper functions...'
import magenta.music as mm
from magenta.music.sequences_lib import concatenate_sequences
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np
import os, time
import tensorflow as tf


class AutoEncoder:
    def __init__(self):
        self.trio_models = {}
        self.configs = {}
        self.load_model()
        self.input_data = None
        self.m1 = None
        self.m2 = None
        self.d = None

    def load_model(self, model='hierdec-trio_16bar', path='./content/checkpoints/trio_16bar_hierdec.ckpt'):
        print 'Loading models...'
        self.configs[model] = configs.CONFIG_MAP[model]
        self.trio_models[model] = TrainedModel(self.configs[model], batch_size=4, checkpoint_dir_or_path=path)

    def slerp(self, p0, p1, t):
        """Spherical linear interpolation."""
        omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1))))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

    def interpolate(self, model, start_seq, end_seq, num_steps, max_length=32,
                    assert_same_length=True, temperature=0.5,
                    individual_duration=4.0):
        """Interpolates between a start and end sequence."""
        print 'Interpolating...'
        _, mu, _ = model.encode([start_seq, end_seq], assert_same_length)
        z = np.array([self.slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, num_steps)])
        note_sequences = model.decode(
          length=max_length,
          z=z,
          temperature=temperature)
        print 'Start -> End Interpolation'
        interp_seq = concatenate_sequences(note_sequences, [individual_duration] * len(note_sequences))
        return interp_seq if num_steps > 3 else note_sequences[num_steps // 2]

    def download(self, note_sequence, filename):
        mm.sequence_proto_to_midi_file(note_sequence, filename)

    def default_interpolation(self, trio_interp_model, start_trio, end_trio, num_steps=3, max_length=256, individual_duration=32, temperature=0.5):
        trio_16bar_mean = self.interpolate(self.trio_models[trio_interp_model], start_trio, end_trio, num_steps=num_steps, max_length=max_length, individual_duration=individual_duration, temperature=temperature)
        return trio_16bar_mean

    def generate_samples(self, temperature=0.5, download=False):
        print 'Generating samples...'
        trio_sample_model = "hierdec_trio_16bar"
        trio_16_samples = self.trio_models[trio_sample_model].sample(n=2, length=256, temperature=temperature)

        if download:
            print 'Downloading samples...'
            for i, ns in enumerate(trio_16_samples):
                self.download(ns, './content/midi/%s_sample_%d.mid' % (trio_sample_model, i))
                print 'Finished downloading samples'

    def input_midi(self, input, single=False):
        """Define the input to be used by the auto-encoder
        Args:
            input: string indicating a midi file (when single=True) or a genre directory (when single=False)"""
        print 'Defining input...'
        if not single:
            input_path = './content/%s/*.mid' % input
            self.input_data = [
                tf.gfile.Open(fn).read()
                for fn in sorted(tf.gfile.Glob(input_path))]
        if single:
            input_path = './%s' % input
            self.input_data = [
                tf.gfile.Open(fn).read()
                for fn in sorted(tf.gfile.Glob(input_path))]

    def extract_trios(self, config, download=False, genre='undefined'):
        """Extract trios and convert to notesequence
        Args:
            input_trio_midi_data: midi files used for encoding
            config: pre-trained model configuration that was loaded
            download: should the trios be downloaded
            genre: genre string that will be used for the downloaded filename"""
        print 'Extracting trios...'
        seqs = []
        for m in self.input_data:
            try:
                mns = mm.midi_to_sequence_proto(m)
                seqs.append(mns)
            except:
                pass

        extracted_trios = []
        for ns in seqs:
          extracted_trios.extend(
              config.data_converter.to_notesequences(
                  config.data_converter.to_tensors(ns)[1]))

        if download:
            for i, ns in enumerate(extracted_trios):
                self.download(ns, './content/midi/%s_extrio_%d.mid' % (genre, i))
            print '[LG] Done downloading trios'

        return extracted_trios

    def get_mean(self, extracted_trios, store, im='hierdec-trio_16bar', temperature=0.5, genre='undefined'):
        """Get the mean latent vector for a genre
        Args:
            extracted_trios: ___
            im: interpolation model"""
        print 'Getting mean latent vector...'
        model = self.trio_models[im]
        ph = np.zeros((1, 512))
        counter = 0
        for i, trio in enumerate(extracted_trios):
            try:
                _, mu, _ = model.encode([trio], assert_same_length=True)
                ph += mu
                counter += 1
                print 'Encoding trio %d/%d' % (i, len(extracted_trios)-1)
            except:
                print 'Cannot encode trio %d' % i
        mean_mu = ph / counter
        if store == 1:
            self.m1 = mean_mu
        if store == 2:
            self.m2 = mean_mu

        output_sequence = model.decode(length=32, z=mean_mu, temperature=temperature)

        final_seq = concatenate_sequences(output_sequence, [32.0] * len(output_sequence))

        self.download(final_seq, './content/output/%s_truemean.mid' % genre)
        print 'Mean %s acquired' % genre

    def save_mean(self, mean, filename):
        data = np.array(mean)
        np.save(filename, data)
        # f = open('save/output_seqs/%s.txt' % (filename + time.strftime("%Y%m%d")), 'w+')
        # f.write(str(seq))
        # f.close()
        print "Saved output sequence to 'save/means/%s.npy'" % filename

    def load_mean(self, filename, mean):
        """Get a previously saved mean from a .npy file
        Args:
            filename: path to the .npy file of the mean
            mean: string that can be either 's' (source genre) or 't' (target genre)
            source and target are important since attribute vectors are always created from m1 to m2."""
        if mean == "s":
            self.m1 = np.load(filename)
        elif mean == "t":
            self.m2 = np.load(filename)
        # f = open(filename, 'r')
        # c = eval(f.read())
        # f.close()

    def compute_d(self):
        self.d = self.m2 - self.m1

    def save_d(self, filename):
        data = np.array(self.d)
        np.save(filename, data)
        # f = open('save/vectors/%s.txt' % filename, 'w+')
        # f.write(str(self.d))
        # f.close()
        print "Saved output sequence to 'save/vectors/%s.npy'" % filename

    def load_d(self, filename):
        data = np.load(filename)
        self.d = data.tolist()
        # f = open(filename, 'r')
        # self.d = eval(f.read())
        # f.close()

    def get_latent(self, extracted_trios, im='hierdec-trio_16bar'):
        print 'Getting latent vector...'
        model = self.trio_models[im]
        ph = np.zeros((1, 512))
        counter = 0
        for i, trio in enumerate(extracted_trios):
            try:
                _, mu, _ = model.encode([trio], assert_same_length=True)
                ph += mu
                counter += 1
                print 'Encoding trio %d/%d' % (i, len(extracted_trios)-1)
            except:
                print 'Cannot encode trio %d' % i
        mean_mu = ph / counter

        return mean_mu

    def convert(self, latent_vector, cf, filename, im='hierdec-trio_16bar', temperature=0.5, length=32):
        """Convert a song from genre X to genre Y
        Args:
            latent vector: latent vector of the song in X
            cf: conversion factor - to what degree should X be converted to Y"""

        new_lv = latent_vector + cf * self.d

        model = self.trio_models[im]
        output_sequence = model.decode(length=length, z=new_lv, temperature=temperature)

        final_seq = concatenate_sequences(output_sequence)#, [64.0] * len(output_sequence)) ### Change 64 back to 32.0

        self.download(final_seq, './content/output/converted_%s.mid' % filename)
        print 'Convertion successful. converted_%s.mid created.' % filename


genres = ['jazz', 'metal']
ae = AutoEncoder()

for genre in genres:
    ae.input_midi(genre)
    trios = ae.extract_trios(ae.configs['hierdec-trio_16bar'])
    ae.get_mean(trios, 1, genre=genre)

ae.input_midi(genres[0])
trios = ae.extract_trios(ae.configs['hierdec-trio_16bar'])
ae.get_mean(trios, 1, genre=genres[0])

ae.input_midi(genres[1])
trios = ae.extract_trios(ae.configs['hierdec-trio_16bar'])
ae.get_mean(trios, 2, genre=genres[1])

ae.compute_d()
ae.save_d("%s2%s" % (genres[0], genres[1]))
ae.load_d('%s2%s.npy' % (genres[0], genres[1]))
# ae.input_midi('content/test/orpheus.mid', single=True)
# trios = ae.extract_trios(ae.configs['hierdec-trio_16bar'])
# latent = ae.get_latent(trios)

# ae.convert(latent, 1, '003')
