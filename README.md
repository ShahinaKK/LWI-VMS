# **Learnable Weight Initialization for Volumetric Medical Image Segmentation**

[Shahina Kunhimon](https://github.com/ShahinaKK)


> **Abstract:** *Hybrid volumetric medical image segmentation models, combining the advantages of local convolution and global attention, have recently received considerable attention. While mainly focusing on architectural modifications, most existing hybrid approaches still use conventional data-independent weight initialization schemes which restrict their performance due to ignoring the inherent volumetric nature of the medical data. To address this issue, we propose a learnable weight initialization approach that utilizes the available medical training data to effectively learn the contextual and structural cues via the proposed self-supervised objectives. Our approach is easy to integrate into any hybrid model and requires no external training data. Experiments on multi-organ and lung cancer segmentation tasks demonstrate the effectiveness of our approach, leading to state-of-the-art segmentation
performance.* 
<hr />



## Evaluation

To reproduce the results of UNETR++ (Ours): 
1 Download [ Synapse_UNETR++_Ours_weights]([https://drive.google.com/drive/folders/1yNWyPB_8o5_YaR5R_0b_h5rZJqmCs5Kv]) and paste ```model_final_checkpoint.model``` in the following path:
```shell
unetr_pp/evaluation/unetr_pp_ours_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/
```
Then, run 
```shell
bash evaluation_scripts/run_evaluation_synapse.sh






