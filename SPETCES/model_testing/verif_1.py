"""

Verifying the ability of the model to recognize cosmics as TP, using candidates from ICRC2025

"""

import tensorflow as tf 
import numpy as np 
from glob import glob
import os


from SPETCES.model_training.model import model_1D_def, model_2D_def
from SPETCES.imported_fcts import plot_trace_and_PSD, master_path, path_weights_1D, path_weights_2D


# ----------- Import trained models ---------
model_1D = model_1D_def()
model_2D = model_2D_def()

model_1D.load_weights(path_weights_1D)
model_2D.load_weights(path_weights_2D)

# --------- Test models on CR events ----------

candidates_file = "pengxiong_202510"

repert_predict = sorted(glob(f'{master_path}/datasets/dataset_preliminary_work/{candidates_file}/*'))

pred_CRC_1D = np.array([])
pred_CRC_2D = np.array([])

for i in range(len(repert_predict)) :
    data_test = np.load(repert_predict[i])

    name = repert_predict[i].split('/')[-1]

    pred_1D = model_1D.predict(data_test)
    pred_2D = model_2D.predict(data_test)

    print(f'Prediction of {name} by model 1D: {pred_1D}')
    print(f'Prediction of {name} by model 2D: {pred_2D}')

    pred_CRC_1D = np.append(pred_CRC_1D, pred_1D)
    pred_CRC_2D = np.append(pred_CRC_2D, pred_2D)

    mask_antenna_1_1D = (pred_1D > 0.5)
    mask_antenna_1_2D = (pred_2D > 0.5)

    if np.sum(mask_antenna_1_1D) or np.sum(mask_antenna_1_2D) :
        out_path = f'{master_path}/plots/pred_{candidates_file}/{name}'
        os.makedirs(out_path, exist_ok=True)
        for j in range(len(pred_1D)):
            plot_trace_and_PSD(data_test[j],
                               save_path=f'{out_path}/example_trace{j}',
                               title=f'Antenna {j+1}; pred: 1D={pred_1D[j]} - 2D={pred_2D[j]}')

print(f'Accuracy on CRC events ({candidates_file}), for model 1D: ', np.sum((pred_CRC_1D>0.5))/len(pred_CRC_1D))
print(f'Accuracy on CRC events ({candidates_file}), for model 2D: ', np.sum((pred_CRC_2D>0.5))/len(pred_CRC_2D))