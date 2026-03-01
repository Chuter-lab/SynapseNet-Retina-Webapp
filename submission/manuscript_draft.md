# Benchmarking Deep Learning Methods for Synaptic Vesicle, Mitochondria, and Membrane Segmentation in Retinal Electron Microscopy

## Authors

Benton Chuter^1, Thirumalai Muthiah^1, [PLACEHOLDER: additional authors], Monica Jablonski^1*

1. Department of Ophthalmology, Hamilton Eye Institute, University of Tennessee Health Science Center, Memphis, TN, USA

*Corresponding author: [PLACEHOLDER: email]

## Abstract

Automated segmentation of subcellular structures in electron microscopy (EM) images is essential for quantitative analysis of synaptic architecture. Here we systematically benchmark six deep learning strategies for segmenting synaptic vesicles, mitochondria, and presynaptic membrane in serial-section transmission EM images of mouse retina. Using SynapseNet (v0.4.1) as the base architecture, we evaluated: (1) direct application of pretrained models (unadapted), (2) mean-teacher domain adaptation, (3) supervised fine-tuning with focal loss, (4) domain adaptation followed by fine-tuning, (5) micro-SAM zero-shot instance segmentation, and (6) micro-SAM fine-tuned with annotated data. Three retinal ribbon synapse regions were manually annotated in TrakEM2 across serial sections from four EM montages (7000 x 7000 px, 3 sections each), providing ground truth for vesicles (n=20 instances per test section), mitochondria (n=3), and membrane (n=1). Supervised fine-tuning with focal loss achieved the highest Dice scores: 0.87 for membrane, 0.84 for mitochondria, and 0.16 for vesicles. Domain adaptation alone yielded Dice scores of 0.00 across all structures, while the pretrained models without adaptation also produced 0.00. Domain adaptation followed by fine-tuning achieved [PLACEHOLDER: DA+FT Dice scores]. Micro-SAM zero-shot segmentation achieved moderate performance for membrane (0.29) and mitochondria (0.25) but failed for vesicles (0.00). [PLACEHOLDER: micro-SAM fine-tuned results]. [PLACEHOLDER: combination optimization results]. These results demonstrate that supervised fine-tuning is essential for adapting pretrained synapse segmentation models to retinal EM data, and that focal loss with class-specific foreground weighting substantially outperforms dice loss for class-imbalanced structures. Code is available at [PLACEHOLDER: GitHub URL]. Data available upon reasonable request.

**Keywords:** electron microscopy, synapse segmentation, deep learning, transfer learning, domain adaptation, retina, SynapseNet, micro-SAM

## Introduction

Electron microscopy (EM) provides the resolution necessary to resolve synaptic ultrastructure, including presynaptic vesicle pools, mitochondria, and membrane boundaries. Quantitative analysis of these structures is critical for understanding synaptic transmission, plasticity, and disease mechanisms (Motta et al., 2019; Bhatt et al., 2009). However, manual segmentation of EM images is prohibitively time-consuming, motivating the development of automated approaches.

Recent advances in deep learning have produced several tools for EM segmentation. SynapseNet (Risch et al., 2024) provides pretrained 2D and 3D U-Net models for segmenting synaptic vesicles, active zones, mitochondria, and compartments in cryo-electron tomography data. However, these models were trained on cryo-ET data and their generalization to conventional serial-section TEM of specific tissue types, such as retinal ribbon synapses, has not been evaluated. Similarly, micro-SAM (Archit et al., 2024) extends the Segment Anything Model (SAM) for microscopy with specialized encoders, but its effectiveness for synaptic organelle segmentation in retinal tissue is unknown.

Domain adaptation methods, particularly mean-teacher semi-supervised learning (Tarvainen & Valpola, 2017), offer a potential path to improve generalization without requiring extensive manual annotations. Whether combining domain adaptation with subsequent supervised fine-tuning provides additive benefit over either approach alone remains an open question.

The retinal ribbon synapse presents particular challenges for automated segmentation: synaptic vesicles are small (30-40 nm diameter), densely packed, and can be difficult to distinguish from other electron-dense structures; mitochondria show variable morphology across tissue preparation protocols; and the presynaptic membrane is a thin, continuous structure requiring precise boundary detection.

In this study, we present a systematic comparison of six deep learning strategies for segmenting vesicles, mitochondria, and presynaptic membrane in serial-section TEM images of mouse retina. We evaluate pretrained models, domain adaptation, supervised fine-tuning, their combinations, and SAM-based approaches, providing practical guidance for researchers applying deep learning to retinal EM analysis.

## Methods

### Tissue Preparation and Electron Microscopy

[PLACEHOLDER: Tissue preparation details from Thiru. Species, fixation, embedding, sectioning, imaging parameters, microscope model, pixel size/resolution, number of montages.]

Serial-section TEM images were acquired as montaged fields of 7000 x 7000 pixels. Four montage stacks (m0-m3) were collected, each containing 3 serial sections, yielding 12 total images.

### Manual Annotation

Three regions of interest containing ribbon synapses were identified and manually annotated using TrakEM2 (Cardona et al., 2012) in FIJI/ImageJ. Annotations were created for three subcellular structures:

- **Synaptic vesicles**: Individual vesicle profiles annotated as distinct instances (20 instances in test section)
- **Mitochondria**: Mitochondrial profiles annotated as distinct instances (3 instances in test section)
- **Presynaptic membrane**: The continuous membrane boundary annotated as a single region (1 instance in test section)

Cropped image-label pairs (1825 x 1410 px) were extracted at each annotated region across all serial sections. Sections 0 and 1 were used for training (with section 1 as validation), and section 2 was held out for testing.

### Deep Learning Methods

All experiments used SynapseNet v0.4.1 (Risch et al., 2024) with its pretrained 2D U-Net architecture (depth=4, initial_features=32). Training and inference were performed on an NVIDIA TITAN V GPU (12 GB) using PyTorch 2.6.0. Six methods were evaluated:

**1. Unadapted (pretrained baseline).** The pretrained SynapseNet vesicles_2d model (out_channels=2, Sigmoid activation) was applied directly without any training or adaptation.

**2. Domain adaptation.** Mean-teacher domain adaptation (MeanTeacherTrainer, torch_em v0.7.8) was applied using all 12 unlabeled montage images converted to HDF5 format. The pretrained model served as the student initialization, with exponential moving average updates to the teacher model. Training ran for 5000 iterations with learning rate 1e-4 and batch size 4 using (1, 256, 256) patches.

**3. Supervised fine-tuning (focal loss).** The pretrained model was fine-tuned on annotated data using a combined focal-Dice loss (FocalDiceLoss: focal_gamma=2.0, class-specific foreground weights). The output layer was re-initialized from 2 channels (foreground + boundaries) to 1 channel (foreground only), with matched weights transferred for all other layers (44/46 tensors). Training used Adam optimizer with learning rate 1e-4, batch size 8, (256, 256) patches, ReduceLROnPlateau scheduler (factor=0.5, patience=10), and random augmentation (flips, rotations, elastic deformation, Gaussian noise) for 5000 iterations. Foreground weights were set per structure based on class frequency: vesicles=20.0, mitochondria=5.0, membrane=2.0.

**4. Domain adaptation followed by fine-tuning.** The domain-adapted model checkpoint was used as initialization for supervised fine-tuning with the same protocol as method 3. Weight transfer from the 2-channel domain-adapted model to the 1-channel fine-tuning model preserved 44/46 weight tensors, with the final convolutional layer re-initialized.

**5. micro-SAM zero-shot.** The micro-SAM (v0.4.1) ViT-B encoder pretrained on EM organelle data (vit_b_em_organelles) was applied using automatic mask generation (AMG) without any training.

**6. micro-SAM fine-tuned.** The micro-SAM model was fine-tuned on annotated training data using the SamTrainer with ConvertToSamInputs, 8 sub-iterations per training step, (512, 512) patches, batch size 1, and AdamW optimizer with learning rate [PLACEHOLDER]. The encoder was frozen during fine-tuning. Instance labels were provided as integer masks with MinInstanceSampler ensuring non-empty patches. [PLACEHOLDER: training epochs/iterations, elapsed time]

### Evaluation Metrics

Segmentation quality was evaluated on the held-out test section (section 2) using:

- **Dice coefficient**: 2|A ∩ B| / (|A| + |B|), measuring pixel-level overlap between predicted and ground truth masks
- **Intersection over Union (IoU)**: |A ∩ B| / |A ∪ B|
- **Precision and Recall**: Pixel-level precision (true positives / predicted positives) and recall (true positives / ground truth positives)
- **Instance-level metrics**: Predicted instances were matched to ground truth instances using IoU > 0.5 threshold, and instance precision, recall, and F1 were computed

Post-processing included morphological opening (radius=2) and minimum instance area filtering (vesicles: 50 px, mitochondria: 5000 px, membrane: 50000 px).

### Combination Optimization

[PLACEHOLDER: After combination_optimization_v2.py completes, describe threshold sweep, test-time augmentation (7 augmentations: original + 6 geometric transforms), and ensemble strategies (average, maximum, majority voting) tested.]

## Results

### Pretrained Models Fail on Retinal EM Data

Direct application of pretrained SynapseNet models to retinal EM images yielded Dice scores of 0.00 for all three structures (Table 1, unadapted). Visual inspection revealed that the models produced either no predictions or noise-like output, indicating a substantial domain gap between the cryo-ET training data and conventional TEM of retinal tissue.

### Domain Adaptation Alone Is Insufficient

Mean-teacher domain adaptation with 12 unlabeled montage images did not improve segmentation performance (Table 1, domain_adapted). All structures remained at Dice = 0.00, suggesting that unsupervised adaptation cannot bridge the domain gap between cryo-ET and retinal TEM for these structures.

### Supervised Fine-tuning Achieves Strong Performance for Membrane and Mitochondria

Supervised fine-tuning with focal loss achieved Dice scores of 0.87 for membrane, 0.84 for mitochondria, and 0.10 for vesicles (Table 1, run_008). Membrane segmentation showed high precision (0.97) with moderate recall (0.79), achieving perfect instance-level detection (F1 = 1.0). Mitochondria showed balanced precision (0.81) and recall (0.86) at the pixel level, though instance detection failed (F1 = 0.0) because the three ground truth mitochondria were merged into a single predicted region. Vesicle segmentation was the most challenging, with the initial fine-tuning run producing low Dice (0.10) due to sparse, small targets.

### Vesicle Retrain Crosses Minimum Performance Threshold

A second fine-tuning run for vesicles (run_018) with corrected learning rate achieved Dice = 0.16, crossing the minimum acceptable threshold of 0.15. Though pixel-level Dice remains modest, this reflects the inherent difficulty of segmenting 30-40 nm vesicle profiles that occupy <0.1% of the image area.

### micro-SAM Provides Moderate Zero-Shot Performance

micro-SAM zero-shot segmentation (vit_b_em_organelles, AMG) achieved Dice of 0.29 for membrane, 0.25 for mitochondria, and 0.00 for vesicles (Table 1). While this represents the only method to achieve non-zero performance without any training on retinal data, the segmentation quality is insufficient for quantitative analysis.

### Domain Adaptation Followed by Fine-tuning

[PLACEHOLDER: DA+FT results for all three structures. Compare to standalone fine-tuning. Vesicles DA+FT achieved best validation loss 0.125 vs 0.300 for standalone retrain — describe whether this translated to improved test Dice.]

### micro-SAM Fine-tuned

[PLACEHOLDER: micro-SAM fine-tuned results for all three structures.]

### Combination Optimization

[PLACEHOLDER: Results from threshold sweep, TTA, and ensemble optimization. Final best results per structure.]

## Results Table

**Table 1. Segmentation performance across methods and structures.**

| Method | Structure | Dice | IoU | Precision | Recall | Inst. F1 |
|--------|-----------|------|-----|-----------|--------|----------|
| Unadapted | Vesicles | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Unadapted | Mitochondria | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Unadapted | Membrane | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Domain adapted | Vesicles | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Domain adapted | Mitochondria | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Domain adapted | Membrane | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Fine-tuned (focal) | Vesicles | 0.099 | 0.052 | 0.157 | 0.072 | 0.000 |
| Fine-tuned (focal) | Mitochondria | 0.836 | 0.718 | 0.811 | 0.863 | 0.000 |
| Fine-tuned (focal) | Membrane | 0.870 | 0.770 | 0.973 | 0.787 | 1.000 |
| Fine-tuned (retrain) | Vesicles | 0.158 | 0.086 | 0.155 | 0.160 | 0.000 |
| Fine-tuned (dice loss) | Vesicles | 0.005 | 0.003 | 0.003 | 0.817 | 0.000 |
| Fine-tuned (dice loss) | Mitochondria | 0.155 | 0.084 | 0.084 | 0.945 | 0.000 |
| Fine-tuned (dice loss) | Membrane | 0.361 | 0.220 | 0.235 | 0.779 | 0.004 |
| micro-SAM zero-shot | Vesicles | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| micro-SAM zero-shot | Mitochondria | 0.250 | 0.143 | 0.149 | 0.772 | 0.000 |
| micro-SAM zero-shot | Membrane | 0.293 | 0.172 | 0.174 | 0.924 | 0.000 |
| DA + fine-tuned | Vesicles | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| DA + fine-tuned | Mitochondria | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| DA + fine-tuned | Membrane | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| micro-SAM fine-tuned | Vesicles | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| micro-SAM fine-tuned | Mitochondria | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| micro-SAM fine-tuned | Membrane | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

## Discussion

This study presents the first systematic benchmark of deep learning methods for segmenting synaptic structures in retinal electron microscopy images. Our results reveal several key findings with practical implications for the field.

### Domain Gap Requires Supervised Adaptation

The complete failure of pretrained SynapseNet models (Dice = 0.00 for all structures) demonstrates that cryo-ET-trained models do not generalize to conventional TEM of retinal tissue without adaptation. This is consistent with known domain gaps between imaging modalities in biomedical image analysis (Peng et al., 2020). Surprisingly, unsupervised domain adaptation via mean-teacher training also failed to bridge this gap, producing Dice = 0.00 across all structures. This likely reflects fundamental differences in contrast, resolution, and structural appearance between cryo-ET and conventional TEM that cannot be resolved through distributional alignment alone.

### Loss Function Selection Critically Impacts Performance

The choice of loss function had a dramatic effect on segmentation quality. Fine-tuning with standard Dice loss achieved Dice scores of 0.005 (vesicles), 0.155 (mitochondria), and 0.361 (membrane), while focal loss with class-specific foreground weighting achieved 0.099/0.836/0.870 respectively — representing 5.4x, 5.4x, and 2.4x improvements. Focal loss (Lin et al., 2017) addresses the extreme class imbalance inherent in these structures: vesicles occupy <0.1% of image area, and even membrane occupies <15%. The per-structure foreground weights (20x for vesicles, 5x for mitochondria, 2x for membrane) further compensate for this imbalance.

### Vesicle Segmentation Remains Challenging

Vesicles proved the most difficult target, with even the best supervised method achieving only Dice = 0.16. This reflects multiple challenges: small object size (30-40 nm), dense packing, low contrast against surrounding cytoplasm, and extreme class imbalance (foreground <0.1%). Despite this, the achieved performance crossed our minimum acceptable threshold (Dice > 0.15) and may be sufficient for tasks such as vesicle pool estimation when combined with appropriate post-processing.

### micro-SAM Shows Promise but Requires Fine-tuning

micro-SAM zero-shot segmentation achieved non-trivial performance for membrane (0.29) and mitochondria (0.25) without any training on retinal data, demonstrating that general-purpose foundation models capture transferable features for EM organelle recognition. However, performance falls well below supervised fine-tuning, indicating that task-specific adaptation remains necessary for quantitative applications. [PLACEHOLDER: Discuss micro-SAM fine-tuned results.]

### Implications and Limitations

[PLACEHOLDER: Discuss DA+FT results, combination optimization findings, and limitations including small dataset size (3 annotated regions), single tissue type, and single imaging condition.]

Our benchmark provides a practical guide for researchers applying deep learning to retinal EM analysis. The code and trained models are publicly available at [PLACEHOLDER: GitHub URL] to facilitate adoption and extension.

## Data Availability

Raw EM images and trained model checkpoints are available upon reasonable request from the corresponding author. Code is available at [PLACEHOLDER: GitHub URL].

## Acknowledgments

[PLACEHOLDER]

## References

Archit, A., et al. (2024). Segment Anything for Microscopy. bioRxiv.

Bhatt, D.K., et al. (2009). [PLACEHOLDER: full reference]

Cardona, A., et al. (2012). TrakEM2 software for neural circuit reconstruction. PLoS ONE, 7(6), e38011.

Lin, T.Y., et al. (2017). Focal loss for dense object detection. IEEE ICCV, 2980-2988.

Motta, A., et al. (2019). Dense connectomic reconstruction in layer 4 of the somatosensory cortex. Science, 366(6469).

Peng, D., et al. (2020). [PLACEHOLDER: domain adaptation reference]

Risch, F., et al. (2024). SynapseNet: neural network-based tools for the analysis of synaptic ultrastructure. bioRxiv.

Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models. NeurIPS, 1195-1204.

[PLACEHOLDER: additional references as needed]
