import music21,os,glob
import numpy as np
import pickle
from datetime import datetime
from music21 import *
from fractions import Fraction
from utils import encode
import config 

outdir = config.outdir
basedir =config.basedir
subs = config.subs
class_vector = []
melody_features= []
bass_features= []
chord_features = []
duration_features = []
duration_space = []
all_duration_array = []
reduced_chord_space = []
chord_tuple_reduced_space = []
all_song_reduced_chord_simplify =[]
all_song_chord_tuple = []
no = 1
chords_sequence =[]
weight_table = pickle.load(open("/Users/VietAnh/Documents/Courses/Machinelearning/Midterm/test/weight_table.p","rb" ))
named_chords_encode = pickle.load(open(config.named_chords_encode_path,"rb" ))
for sub in subs:
    for fname in glob.glob(os.path.join(basedir,sub,'*.mid')):
        print str(datetime.now())
        melody_array =[]
        bass_array =[]
        chord_array = []
        duration_array = []
        melody_tuple =[]
        chord_tuple =[]
        bass_tuple=[]        
        print fname
        song = converter.parse(fname)
        Key_Signature = song.analyze('key')
        
        # Use Music21 chordify to get chord
        sChords = song.chordify().recurse().getElementsByClass('Chord')
       
        #get duration
        for c in sChords: 
            d = c.duration.type
            duration_array.append(d)
            if d not in duration_space:
                duration_space.append(d)
        all_duration_array.append(duration_array)
        x = [[] for m in range(len(sChords))]
        i = 0
        
        # get notes relative to Key Signature
        for thisChord in sChords:
            for j in range(len(thisChord)):
                x[i].append((thisChord.pitches[j].midi-Key_Signature.tonic.midi+96)%12)
            i = i + 1
        # .tonic return tonic of Key Signature,e.g Key Signature f# minor then tonic is F#
        # .tonic.midi return the number related to this note e.g F# is 66,C#5 is 73, B4 is 71
        #  add 96 so that the distance don't have negative number, as an Octave has 12 notes
       
        # get melody, bass notes
        for k in range(len(x)):
            x[k] = np.array(x[k])
            x[k] = np.unique(x[k])
            #print x[k]
            melody_array.append(max(x[k]))
            bass_array.append(min(x[k]))
        # e.g melody array [7, 5, 3, 5, 7, 10, 7, 5, .....]
        
        # find the closest named chord
        for chord in x:
            chord_code = encode(chord)
            distance = []
            min_distance =[]
            potential_chord =[]
            chord_identify = []
            for c in weight_table:
                if c[1] == chord_code and c[0] != chord_code:
                    distance.append(c)
            distance = np.array(distance)
            
            for row in distance:
                if row[2] == min(distance[:,2]):
                    potential_chord.append(row[0])
           
            chord_identify = min(potential_chord)
            chord_array.append(chord_identify)
        # e.g. chord_array has format[ 145, 161, 73, 137, 649] 
           
        chord_simplify = []

        # convert chord_array to chord_simplify[0, 2, 4, 1, 6] 
        for i in range(len(chord_array)):
            for j in range(len(named_chords_encode)):
                if chord_array[i] == named_chords_encode[j]:
                    chord_simplify.append(j)
        for l in range(0,len(chord_simplify)-3):
            tup_temp = chord_simplify[l:l+4]
            tup_temp = sorted(tup_temp)
            chord_tuple.append(tup_temp)
            if tup_temp not in chord_tuple_reduced_space:
                chord_tuple_reduced_space.append(tup_temp)
        all_song_chord_tuple.append(chord_tuple)
        for l in range(0,len(melody_array)-3):
            melody_tuple.append(melody_array[l:l+4])
            bass_tuple.append(bass_array[l:l+4])
        #e.g melody_tuple is [[7, 5, 3, 5], [5, 3, 5, 7], [3, 5, 7, 10]      
        
        # note_space, melody_vector, bass_vector length is 12^4 = 20736
        melody_vector = np.zeros(20736)
        bass_vector = np.zeros(20736)
                
        for i in range(0,len(melody_tuple)):
            index = melody_tuple[i][0] * 1728 + melody_tuple[i][1] * 144 + melody_tuple[i][2] * 12 + melody_tuple[i][3]  
            melody_vector[index] = melody_vector[index] + 1
        # e.g melody_vector [ 2.  0.  0. ...,  0.  0.  0.]   
        
        for i in range(0,len(bass_tuple)):
            index = bass_tuple[i][0] * 1728 + bass_tuple[i][1] * 144 + bass_tuple[i][2] * 12 + bass_tuple[i][3]  
            bass_vector[index] = bass_vector[index] + 1
 
        melody_vector_norm = np.zeros(20736)
        bass_vector_norm = np.zeros(20736)        
     
        for i in range (0,20736):
            melody_vector_norm[i] = melody_vector[i]/len(melody_tuple)
            bass_vector_norm[i] = bass_vector[i]/len(bass_tuple)
        
        #Add vector of every single file to a list of files:
        melody_features.append(melody_vector_norm)
        bass_features.append(bass_vector_norm)
               
        #Add corresponding class
        class_vector.append(sub)
        no = no + 1
        print no

du_len = len(duration_space)

for m in range(len(all_duration_array)):
    duration_array = all_duration_array[m]
    duration_tuple = []
    duration_simplify = []
    duration_vector= np.zeros(du_len ** 4)
    duration_vector_norm = np.zeros(du_len ** 4)
    for i in range(len(duration_array)):
        for j in range(len(duration_space)):
            if duration_array[i] == duration_space[j]:
                duration_simplify.append(j)
    for l in range(0,len(duration_simplify)-3):
        duration_tuple.append(duration_simplify[l:l+4])
    for i in range(0,len(duration_tuple)):
        index = duration_tuple[i][0] * (du_len ** 3) + duration_tuple[i][1] * (du_len ** 2) + duration_tuple[i][2] * du_len + duration_tuple[i][3]  
        duration_vector[index] = duration_vector[index] + 1
    for i in range (0,len(duration_vector_norm)):
        duration_vector_norm[i] = duration_vector[i]/len(duration_tuple)
    duration_features.append(duration_vector_norm)
    
sct_len = len(chord_tuple_reduced_space) 

for m in range(len(all_song_chord_tuple)):
    chord_tuple_a_song = all_song_chord_tuple[m]
    chord_tuple_vector = np.zeros(sct_len)
    chord_tuple_vector_norm = np.zeros(sct_len)

    for i in range(len(chord_tuple_a_song)):
        for j in range(len(chord_tuple_reduced_space)):
            if chord_tuple_a_song[i] == chord_tuple_reduced_space[j]:
                chord_tuple_vector[j] = chord_tuple_vector[j] + 1
    for i in range (0,sct_len):
        chord_tuple_vector_norm[i] = chord_tuple_vector[i]/len(chord_tuple_a_song)
    chord_features.append(chord_tuple_vector_norm)
np.savez(os.path.join(outdir,'features_4_composers_final'), melody  = melody_features, bass = bass_features, chord = chord_features, duration = duration_features, genre = class_vector)       