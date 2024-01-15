# A quick test for CLASSIX
#
# MIT License
#
# Copyright (c) 2022 Stefan Güttel, Xinye Chen
#

import numpy as np
from classix import CLASSIX, loadData

vdu_signals = loadData('vdu_signals')

for tol in np.arange(0.5,1.1,0.1):
    clx = CLASSIX(radius=tol, verbose=0)
    clx.fit_transform(vdu_signals)
    
    # version 0.2.0
    # np.save('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
    
    # test new version
    checkpoint = np.load('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy')
    comp = clx.labels_ == checkpoint
    assert(comp.all())

print("complete!")
