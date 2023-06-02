from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = output_file = "/home/shahina.kunhimon/PycharmProjects/unetr_plus_plus/DATASET/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/Task002_Synapse/unetr_pp_Plansv2.1_plans_3D.pkl"
    #output_file = "/home/shahina.kunhimon/PycharmProjects/unetr_plus_plus/DATASET/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung/Task006_Lung/unetr_pp_Plansv2.1_plans_3D.pkl"
    a = load_pickle(input_file)
    #a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    a['plans_per_stage'][0]['patch_size']=np.array([96,96,96])
    save_pickle(a, output_file)
