import sox
import os

transformer = sox.Transformer()


input_midi_file_dir = '.'
output_mp3__file_dir = '.'

for file in os.listdir(input_midi_file_dir):
    if file.endswith('.midi'):
    
        transformer.build(file, file[-3] + "mp3")

