# Curated Research Papers: Open-Set Audio & Out-of-Distribution Detection

Repository reference papers for Schematics NPC Data Science Competition 2026.

---

### 1. ROLE: Few-Shot Open-Set Audio Classification via Transductive Prototype Refinement and Class Logit Enhancement
- **File:** `ROLE_few_shot_open_set_audio_classification_2607.26607.pdf`
- **ArXiv:** [arXiv:2607.26607](https://arxiv.org/abs/2607.26607)
- **Authors:** Tianyan Deng, Yanxiong Li, Rui Gao, Jiahao Du (South China University of Technology, 2026)
- **Core Relevance:** Addresses the fatal flaw of transductive open-set audio classification where unknown soundscapes silently drag prototypes away from true class centers. Proposes:
  1. Latent inlierness-weighted prototype refinement.
  2. Decoupled scoring for unknown-class samples using a prior-adaptive free-energy rejection score.
  3. Transductive loss combining support cross-entropy, conditional entropy minimization, and marginal entropy maximization on frozen audio backbones.

---

### 2. Multi-Label Open-Set Audio Classification
- **File:** `Multi_label_open_set_audio_classification_2310.13759.pdf`
- **ArXiv:** [arXiv:2310.13759](https://arxiv.org/abs/2310.13759)
- **Venue:** DCASE Workshop 2023
- **Core Relevance:** Formalizes the mathematical taxonomy of open-set audio:
  - **KK (Known Known):** Target classes seen in training.
  - **KU (Known Unknown):** Out-of-vocabulary negative background soundscapes available during training.
  - **UU (Unknown Unknown):** Novel sounds appearing strictly during inference.
  Demonstrates why evaluating closed-set classification decoupled from unknown detection fails under multi-source overlap.

---

### 3. Energy-Based Out-of-Distribution Detection
- **File:** `Energy_based_OOD_detection_2010.03759.pdf`
- **ArXiv:** [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)
- **Venue:** NeurIPS 2020
- **Authors:** Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li (UC San Diego)
- **Core Relevance:** The foundational paper proving why Softmax posterior probabilities fail under OOD (the softmax confidence trap) due to arbitrary constant logit shifts. Introduces the Helmholtz free energy score $E(x; f) = -T \cdot \log \sum_{c} \exp(f_c(x)/T)$ as a theoretically grounded scoring function aligned with the input probability density.
