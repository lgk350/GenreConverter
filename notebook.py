#@title Setup Environment
#@test {"output": "ignore"}

# import glob
#
# print 'Copying checkpoints and example MIDI from GCS. This will take a few minutes...'
# !gsutil -q -m cp -R gs://download.magenta.tensorflow.org/models/music_vae/colab2/* /content/
#
# print 'Installing dependencies...'
# !apt-get update -qq && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev
# !pip install -qU pyfluidsynth pretty_midi
#
# if glob.glob('/content/magenta*.whl'):
#   !pip install -q /content/magenta*.whl
# else:
#   !pip install -q magenta

# Hack to allow python to pick up the newly-installed fluidsynth lib.
# import ctypes.util
# def proxy_find_library(lib):
#   if lib == 'fluidsynth':
#     return 'libfluidsynth.so.1'
#   else:
#     return ctypes.util.find_library(lib)
#
# ctypes.util.find_library = proxy_find_library


print 'Importing libraries and defining some helper functions...'
# from google.colab import files
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
  _, mu, _ = model.encode([start_seq, end_seq], assert_same_length)
  z = np.array([slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, num_steps)])
  note_sequences = model.decode(
      length=max_length,
      z=z,
      temperature=temperature)

  print 'Start Seq Reconstruction'
  play(note_sequences[0])
  print 'End Seq Reconstruction'
  play(note_sequences[-1])
  print 'Mean Sequence'
  play(note_sequences[num_steps // 2])
  print 'Start -> End Interpolation'
  interp_seq = concatenate_sequences(note_sequences, [individual_duration] * len(note_sequences))
  play(interp_seq)
  mm.plot_sequence(interp_seq)
  return interp_seq if num_steps > 3 else note_sequences[num_steps // 2]

def download(note_sequence, filename):
  mm.sequence_proto_to_midi_file(note_sequence, filename)
  files.download(filename)

print 'Done setting up environment'

#@title Load the pre-trained models.
print 'Loading pre-trained models...'
trio_models = {}
hierdec_trio_16bar_config = configs.CONFIG_MAP['hierdec-trio_16bar']
trio_models['hierdec_trio_16bar'] = TrainedModel(hierdec_trio_16bar_config, batch_size=4, checkpoint_dir_or_path='./content/checkpoints/trio_16bar_hierdec.ckpt')

# flat_trio_16bar_config = configs.CONFIG_MAP['flat-trio_16bar']
# trio_models['baseline_flat_trio_16bar'] = TrainedModel(flat_trio_16bar_config, batch_size=4, checkpoint_dir_or_path='./checkpoints/trio_16bar_flat.ckpt')
print 'Done loading models'

#@title Generate 4 samples from the selected model prior.
print 'Generating samples...'
trio_sample_model = "hierdec_trio_16bar" #@param ["hierdec_trio_16bar", "baseline_flat_trio_16bar"]
temperature = 0.5 #@param {type:"slider", min:0.1, max:1.5, step:0.1}

trio_16_samples = trio_models[trio_sample_model].sample(n=4, length=256, temperature=temperature)
for ns in trio_16_samples:
  play(ns)

print 'Done playing samples'

#@title Option 1: Use example MIDI files for interpolation endpoints.
print 'Using example MIDI files'
input_trio_midi_data = [
    tf.gfile.Open(fn).read()
    for fn in sorted(tf.gfile.Glob('./content/midi/trio_16bar*.mid'))]

#@title Extract trios from MIDI files. This will extract all unique 16-bar trios using a sliding window with a stride of 1 bar.
print 'Extracting trios...'
trio_input_seqs = [mm.midi_to_sequence_proto(m) for m in input_trio_midi_data]
extracted_trios = []
for ns in trio_input_seqs:
  extracted_trios.extend(
      hierdec_trio_16bar_config.data_converter.to_notesequences(
          hierdec_trio_16bar_config.data_converter.to_tensors(ns)[1]))
for i, ns in enumerate(extracted_trios):
  print "Trio", i
  play(ns)

#@title Compute the reconstructions and mean of the two trios, selected from the previous cell.
print 'Computing reconstructions and means...'
trio_interp_model = "hierdec_trio_16bar" #@param ["hierdec_trio_16bar", "baseline_flat_trio_16bar"]

start_trio = 0 #@param {type:"integer"}
end_trio = 1 #@param {type:"integer"}
start_trio = extracted_trios[start_trio]
end_trio = extracted_trios[end_trio]

temperature = 0.5 #@param {type:"slider", min:0.1, max:1.5, step:0.1}
trio_16bar_mean = interpolate(trio_models[trio_interp_model], start_trio, end_trio, num_steps=3, max_length=256, individual_duration=32, temperature=temperature)
