# HADACA3-datacamp2025
HADACA3: Multimodal Deconvolution Benchmark (2025/2026). Benchmarking reference-based deconvolution methods for pancreatic adenocarcinoma using integrated bulk RNA-seq, DNA methylation, and scRNA-seq profiles.
It sounds like you’re putting together a repository for a high-level bioinformatics challenge. Since this is the third iteration of the **HADACA** (HArnessing DAta for CAncer) series, the README needs to be professional, technical, and clear about the multimodal nature of the data.

Here is a structured proposal for your GitHub repository description and `README.md`.

---

Welcome to the **HADACA3-datacamp**, a crowdsourced benchmarking challenge designed to advance deconvolution methods in cancer genomics. This project focuses on deconstructing cellular heterogeneity in **Pancreatic Adenocarcinoma** by integrating transcriptomic and methylomic data.

## 🎯 The Challenge

Cellular heterogeneity is a major driver of tumor progression. However, when we sequence a tumor sample, we get a "bulk" profile—an average of millions of different cells.

**The Goal:** Given a bulk molecular profile (the "smoothie"), use reference data (the "individual fruits") to estimate the exact proportions of:

* **Immune cells**
* **Fibroblasts**
* **Endothelial cells**
* **Classical tumor cells**
* **Basal-like tumor cells**

## 🧪 Methodology & Hypotheses

While reference-based approaches are powerful, they are limited by the quality of the reference itself. HADACA3 tests the hypothesis that **integrating multimodal data** (RNA + DNA Methylation) improves reference quality and, consequently, deconvolution accuracy.

### Reference Datasets Provided:

1. **Bulk RNA-seq** of isolated cell populations.
2. **Bulk DNA Methylation** profiles of isolated populations.
3. **Single-cell RNA-seq** (Labeled data for tumoral and healthy cells).

## 📊 Competition Structure

* **Target:** Pancreatic adenocarcinoma mixtures (simulated, *in vivo*, and *in vitro*).
* **Current Phase Ends:** 16 February 2026.
* **Docker Environment:** `hombergn/hadaca3_final_light`

## 🛠 Getting Started

Detailed instructions on data formats, submission guidelines, and evaluation metrics can be found in the [Hadaca3_bootcamp](https://github.com/bcm-uga/hadaca3/tree/main/Hadaca3_bootcamp) directory.

> ### 💡 The Smoothie Metaphor
> 
> 
> If a tumor is a smoothie made of apples, bananas, and strawberries, deconvolution is the math used to determine exactly how many grams of each fruit went into the blender based only on the sweetness, fiber, and color of the final drink.

