print 'Importing libraries and defining some helper functions...'
import magenta.music as mm
from magenta.music.sequences_lib import concatenate_sequences
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np
import os
import tensorflow as tf


def play(note_sequence):
  mm.play_sequence(note_sequence, synth=mm.fluidsynth)


def slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def interpolate(model, start_seq, end_seq, num_steps, max_length=32,
                assert_same_length=True, temperature=0.5, 
                individual_duration=4.0):
  """Interpolates between a start and end sequence."""
  z_prime, mu, _ = model.encode([start_seq, end_seq], assert_same_length)
  z = np.array([slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, num_steps)])
  note_sequences = model.decode(
      length=max_length,
      z=z,
      temperature=temperature)

  # print 'Start Seq Reconstruction'
  # play(note_sequences[0])
  # print 'End Seq Reconstruction'
  # play(note_sequences[-1])
  # print 'Mean Sequence'
  # play(note_sequences[num_steps // 2])
  print 'Start -> End Interpolation'
  interp_seq = concatenate_sequences(note_sequences, [individual_duration] * len(note_sequences))
  # play(interp_seq)
  # print 'Plotting interpolation...' # doesn't work, only in colab notebook
  # mm.plot_sequence(interp_seq)
  # print 'Done plotting interpolation'
  return interp_seq if num_steps > 3 else note_sequences[num_steps // 2]


def get_genre_mean2(model, start_seq, end_seq, num_steps, max_length=32,
                assert_same_length=True, temperature=0.5,
                individual_duration=4.0):
  z_prime, mu, _ = model.encode([start_seq, end_seq], assert_same_length)
  z = np.array([slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, num_steps)])
  print 'z_prime: ', len(z_prime)
  print 'z: ', len(z)
  note_sequences = model.decode(
      length=max_length,
      z=z,
      temperature=temperature)


def get_genre_mean(model, sequences, assert_same_length=True):
    z_prime, mu, _ = model.encode(sequences, assert_same_length)
    print len(z_prime)


def download(note_sequence, filename):
  mm.sequence_proto_to_midi_file(note_sequence, filename)


print 'Done setting up environment'

print 'Loading pre-trained models...'
trio_models = {}
hierdec_trio_16bar_config = configs.CONFIG_MAP['hierdec-trio_16bar']
trio_models['hierdec_trio_16bar'] = TrainedModel(hierdec_trio_16bar_config, batch_size=4, checkpoint_dir_or_path='./content/checkpoints/trio_16bar_hierdec.ckpt')

# flat_trio_16bar_config = configs.CONFIG_MAP['flat-trio_16bar']
# trio_models['baseline_flat_trio_16bar'] = TrainedModel(flat_trio_16bar_config, batch_size=4, checkpoint_dir_or_path='./content/checkpoints/trio_16bar_flat.ckpt')
print 'Done loading models'

print 'Generating samples...'
trio_sample_model = "hierdec_trio_16bar"
temperature = 0.5

trio_16_samples = trio_models[trio_sample_model].sample(n=2, length=256, temperature=temperature)
# for ns in trio_16_samples:
#   play(ns)

print 'Done playing samples'

print 'Downloading samples...'
for i, ns in enumerate(trio_16_samples):
  download(ns, './content/midi/%s_sample_%d.mid' % (trio_sample_model, i))
print 'Finished downloading samples'

########################################## MusicVAE input
print 'Using example MIDI files'
genre = 'rock'
input_path = './content/%s/*.mid' % genre
input_trio_midi_data = [
    tf.gfile.Open(fn).read()
    for fn in sorted(tf.gfile.Glob(input_path))]
# input_trio_midi_data = [
#     tf.gfile.Open(fn).read()
#     for fn in sorted(tf.gfile.Glob('./content/midi/hierdec_trio_16bar*.mid'))]

print 'Extracting trios...'
seqs = []
for m in input_trio_midi_data:
    try:
        mns = mm.midi_to_sequence_proto(m)
        seqs.append(mns)
    except:
        pass
print '%d midi files sucessfully transformed into note sequences' % len(seqs)
trio_input_seqs = seqs
# trio_input_seqs = [mm.midi_to_sequence_proto(m) for m in input_trio_midi_data]

extracted_trios = []
for ns in trio_input_seqs:
  extracted_trios.extend(
      hierdec_trio_16bar_config.data_converter.to_notesequences(
          hierdec_trio_16bar_config.data_converter.to_tensors(ns)[1]))
# for i, ns in enumerate(extracted_trios):
#   print "Trio", i
#   play(ns)

print 'length extracted_trios: %d' % len(extracted_trios)

print '[LG] Downloading trios...'
for i, ns in enumerate(extracted_trios):
  download(ns, './content/midi/%s_extrio_%d.mid' % (genre, i))
print '[LG] Done downloading trios'

print 'Computing reconstructions and means...'
trio_interp_model = "hierdec_trio_16bar"

start_trio = 0
end_trio = 1
start_trio = extracted_trios[start_trio]
end_trio = extracted_trios[end_trio]

temperature = 0.5
# trio_16bar_mean = interpolate(trio_models[trio_interp_model], start_trio, end_trio, num_steps=3, max_length=256, individual_duration=32, temperature=temperature)
# trio_16bar_mean = get_genre_mean(trio_models[trio_interp_model], extracted_trios)
#TODO: Write a for-loop that tries to encode every trio in encoded trios. Then gather the z values from those trios in a list.
#TODO: Get the 'mean z' by averaging the values of the different z's in the list over the 512 dimensions
#TODO: Do this for 2 genres and figure out how to calculate an attribute vector

model = trio_models[trio_interp_model]
zlist = []
for i, trio in enumerate(extracted_trios):
    try:
        z_prime, mu, _ = model.encode([trio], assert_same_length=True)
        zlist.append(z_prime)
        print 'Encoding trio %d/%d' % (i, len(extracted_trios)-1)
    except:
        print 'Cannot encode trio %d' % i
print 'Length zlist: ', len(zlist)

mean_z = [sum(e)/len(e) for e in zip(*zlist)]
print 'lenth z_mean: ', len(mean_z[0])
print 'type z_mean:', type(mean_z)
print 'z_mean:', mean_z

# TODO: write decoding step to get the mean midi file

output_sequence = model.decode(length=32, z=mean_z[0], temperature=temperature)
# download(output_sequence, './content/output/%s_truemean.mid' % genre)

# print 'Downloading mean midi file...'
# download(trio_16bar_mean, './content/output/%s_mean.mid' % genre)
# print 'Done downloading mean midi file'
