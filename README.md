# Advanced Comparative Analysis of Deep Learning and Transformer Models for Caries Detection on Dental Radiographs

![Dental Caries Detection](https://via.placeholder.com/800x200?text=Caries+Detection+with+AI)

This repository contains the implementation and analysis of deep learning (CNN) and transformer-based models for automated caries detection in dental radiographs, as presented in the paper **"Advanced Comparative Analysis of Deep Learning and Transformer Models for Caries Detection on Dental Radiographs"**.

---

## 📌 Key Contributions
- **Comparative study** of CNN (VGGNet16, ResNet50, EfficientNet) vs. Transformer (ViT, Swin Transformer) models.
- **Swin Transformer achieves SOTA performance**: 97.6% accuracy on panoramic X-rays (Dataset 1) and 81.5% F1-score on periapical images (Dataset 2).
- **Robust data augmentation**: Expanded datasets from 20,000 to 31,000 (panoramic) and 142 to 710 (periapical) images via rotation, flipping, and contrast adjustments.
- **Clinical relevance**: Demonstrates AI's potential to assist dentists in early caries detection, improving diagnostic workflows.

---

## 🚀 Models Evaluated
| Model           | Accuracy (Dataset 1) | F1-Score (Dataset 1) | Accuracy (Dataset 2) | F1-Score (Dataset 2) |
|-----------------|----------------------|----------------------|----------------------|----------------------|
| VGGNet16        | 94.5%                | 93.8%                | 54%                  | 59%                  |
| ResNet50        | 96.2%                | 96.4%                | 55%                  | 33%                  |
| EfficientNet-B2 | 96.7%                | 96.8%                | 58%                  | 52%                  |
| ViT             | 95.8%                | 95.8%                | 81%                  | 79.5%                |
| **Swin Transformer** | **97.6%**        | **97.5%**            | **77%**              | **81.5%**            |

---

## 📂 Dataset Overview
### **Dataset 1 (Panoramic X-rays)**
- **20,000 images** (15,000 caries / 5,000 non-caries).
- Augmented to **31,000 images**.
- Sourced from diverse demographics (ages, tooth positions).

### **Dataset 2 (Periapical X-rays)**
- **142 images** (79 caries / 63 non-caries).
- Augmented to **710 images**.
- Annotated by dental experts.

---

## 🛠️ Methodology
1. **Preprocessing**: 
   - Image resizing, normalization, and augmentation (rotation, flipping, contrast adjustment).
2. **Model Architectures**:
   - **CNNs**: VGGNet16, ResNet50, EfficientNet-B2 (using transfer learning).
   - **Transformers**: ViT (12-layer, 768 embeddings) and Swin Transformer (hierarchical window attention).
3. **Training**:
   - Batch size: 32, Learning rate: 0.001 (cosine annealing).
   - Regularization: Dropout, Batch Normalization.
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC.

---

## 📊 Results
- **Swin Transformer outperformed all models** with 97.6% accuracy (Dataset 1) and 81.5% F1-score (Dataset 2).
- **ViT excelled on small datasets** (81% accuracy on Dataset 2) due to global attention.
- **CNNs struggled with small datasets** (e.g., ResNet50 F1-score dropped to 33% on Dataset 2).

![ROC Curves](https://via.placeholder.com/600x300?text=ROC+Curves+Comparison)  
*Figure: ROC curves for all models (Swin Transformer achieves highest AUC).*

---

## 💡 Future Work
- **Explainability**: Integrate Grad-CAM for lesion localization.
- **Synthetic Data**: Use GANs to generate more training samples.
- **Multitask Learning**: Extend to other dental pathologies (e.g., periodontal disease).

---

## 📜 Citation
```bibtex
@article{rady2024caries,
  title={Advanced Comparative Analysis of Deep Learning and Transformer Models for Caries Detection on Dental Radiographs},
  author={Rady, Mohamed and Elaziz, Mohamed Abd and Galal, Osama and Mahmoud, Mohamed and EiBeshlawy, Dina Mohamed and Essam, Ahmad and Karim, Nada Abd el and Dahaba, Mushira},
  journal={Galala University},
  year={2024}
}
