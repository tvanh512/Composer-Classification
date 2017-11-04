import numpy as np
from utils import encode
import config 
import pickle

named_chords =[]
Maj = [0, 4, 7]
minor = [0, 3, 7]
sus = [0, 5, 7]
aug = [0, 4, 8]
dim = [0, 3, 6]
sixth = [0, 4, 7, 9]
min6 =[0, 3, 7, 9]
seventh = [0, 4, 7, 10]
M7 = [0, 4, 7, 11]
min7= [0, 3, 7, 10]
H7 = [0, 3, 6, 10]
o7 = [0, 3, 6, 9]
M9 = [0, 2, 4, 7, 11]
Ninth = [0, 2, 4, 7, 10]
DMN = [0, 1, 4, 7, 10]
min9 = [0, 2, 3, 7, 10]
named_chords.append(Maj)
named_chords.append(minor)
named_chords.append(sus)
named_chords.append(aug)
named_chords.append(dim)
named_chords.append(sixth)
named_chords.append(min6)
named_chords.append(seventh)
named_chords.append(M7)
named_chords.append(min7)
named_chords.append(H7)
named_chords.append(o7)
named_chords.append(M9)
named_chords.append(Ninth)
named_chords.append(DMN)
named_chords.append(min9)
for i in range (16,192):
    named_chords.append(sorted([ int((x + np.floor(i/16))% 12) for x in named_chords[i % 16]]))
named_chords_encode =[]
for chord in named_chords:
    named_chords_encode.append(encode(chord))
named_chords_encode = np.array(named_chords_encode)
named_chords_encode = np.unique(named_chords_encode)
pickle.dump(named_chords_encode,open(config.named_chords_encode_path, "wb" ) )