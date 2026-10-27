- **Name & Branch**
  - **Name:** [Your full name]
  - **Branch:** [Your academic branch]

- **Model Architecture & Core Idea**
  - **Task:** Predict six independent toxicity probabilities for each Wikipedia comment: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
  - **Preprocessing:** Replace missing text with empty strings; lowercase through TF-IDF vectorizers; apply Unicode accent normalization to word features. Fit vocabularies and IDF weights on training rows only during validation.
  - **Features:** Concatenate 80,000 word unigram/bigram features and 60,000 within-word character 3–5-gram features (`char_wb`) into a sparse matrix. Both branches use sublinear term frequency and L2 normalization; feature matrices use `float32`.
  - **Architecture:** Six separate L2-regularized logistic regression classifiers, one per label, using scikit-learn's `liblinear` solver, `max_iter=2000`, and `class_weight="balanced"`. Each classifier outputs a positive-class probability; multiple labels can apply simultaneously.
  - **Validation:** Reproducible 90:10 train/validation split, stratified by whether a comment has any toxic label, with seed 42. Evaluate using the unweighted mean of six label-wise ROC AUC scores.
  - **Final training:** Refit both vectorizers and all six classifiers on the complete labelled training set; export test probabilities to `submission.csv` in the required label order.

- **Unique / Differentiating Ideas**
  - **Complementary text features:** Combine word features for vocabulary and short phrases with character features for partial spellings and spelling variations.
  - **Label-specific regularization:** Tune `C` separately over `[0.25, 0.5, 1, 2, 4]`; retune after enabling balanced class weights. Final values: `toxic=1`, `severe_toxic=0.25`, `obscene=1`, `threat=2`, `insult=1`, `identity_hate=0.5`.
  - **Measured improvement:** Validation mean ROC AUC improved from **0.97809** for the word TF-IDF logistic regression baseline to **0.987235** for the selected configuration; these are holdout scores used for model selection.
  - **Explicit size control:** Allocate 140,000 total features and audit **840,006 classifier parameters + 140,000 IDF values = 980,006**, below the 1,000,000-parameter limit even when including IDF weights.
  - **Submission checks:** Verify prediction shape, finite probabilities within `[0, 1]`, test ID order, and exact submission columns before saving.
