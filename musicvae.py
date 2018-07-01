print 'Importing libraries and defining some helper functions...'
import magenta.music as mm
from magenta.music.sequences_lib import concatenate_sequences
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np
import os
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

    def input_midi(self, genre):
        print 'Defining input...'
        input_path = './content/%s/*.mid' % genre
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
        print 'type mean_mu:', type(mean_mu)
        output_sequence = model.decode(length=32, z=mean_mu, temperature=temperature)
        print 'type output_sequence:', type(output_sequence)

        final_seq = concatenate_sequences(output_sequence, [32.0] * len(output_sequence))
        print 'type final_sequence:', type(final_seq)

        self.download(final_seq, './content/output/%s_truemean.mid' % genre)
        print 'Mean %s acquired' % genre

    def get_d(self):
        self.d = self.m2 - self.m1

    def get_latent(self, extracted_trios, im='hierdec-trio_16bar', temperature=0.5, genre='undefined'):
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


genres = ['jazz', 'country']
ae = AutoEncoder()
for genre in genres:
    ae.input_midi(genre)
    trios = ae.extract_trios(ae.configs['hierdec-trio_16bar'])
# print 'default interpolation type:', type(ae.default_interpolation(ae.trio_models['hierdec-trio_16bar'],
#                                      trios[0], trios[1]), './content/output/test_defint.mid')
    ae.get_mean(trios, 1, genre=genre)
