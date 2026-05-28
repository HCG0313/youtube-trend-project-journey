# Presentation Q&A Guide

This file is a short English FAQ based on the final presentation and the local `QnA_Report.pdf`.

It is not a word-for-word translation of the Korean sheet.  
Instead, it keeps the same questions while rewriting the answers so they match the actual final model, code, and reported metrics.

## 1. Why did we choose BiGRU instead of LSTM?

GRU is lighter than LSTM and uses fewer parameters, which made it a better fit for this project's data scale.  
We then extended it to a bidirectional version so the model could read the recent 12-week sequence with both forward and backward context.

## 2. How was the main rise target defined?

The project compares the current 4-week average reaction level with the next 4-week average reaction level.  
Instead of labeling every positive growth case as "rise," it marks only category-specific high-growth periods above a selected training quantile threshold.

## 3. Did Google Trends and calendar variables really help?

Yes, but the improvement was not identical on every metric.  
The final `calendar_search` setup was chosen because it gave more stable validation-time Top-5 ranking performance and better test-time balanced accuracy and ROC AUC than the `calendar_only` version.

## 4. Why were only 10 categories used in the final presentation?

The broader active set contained 16 categories, but several of them were too sparse or too unstable at the weekly level.  
The final presentation fixed the scope to 10 core categories with better continuity and more interpretable trends.

## 5. How were Google Trends weeks aligned with YouTube data?

Both data sources were converted to the same ISO week definition and then merged on a shared category-week time axis.

## 6. How was the multi-head loss designed?

The final model does not optimize only one binary rise target.  
It jointly learns:

- main rise classification
- rank-up classification
- 4-step weekly rise signals
- pairwise ranking signals

The final training loss combines these parts rather than using a simple 0.7/0.3 split.

## 7. Is the model biased toward categories that are always large?

That risk exists in general, so the project explicitly used relative-growth and momentum-oriented features rather than raw scale alone.  
The final Top-5 includes categories such as Pets and Education, which shows that the model is not simply repeating the biggest categories every time.

## 8. Are the final Top-5 categories reasonable?

Yes. The final list should be interpreted as "categories with strong signs of relative rise over the next 4 weeks," not simply "categories with the largest absolute reaction volume today."

## 9. Was there a risk of overfitting?

Yes, so the model was kept relatively compact and regularized.  
The true learning unit was not individual videos but category-week sequences, and the final pipeline also used dropout and constrained representation size to reduce overfitting risk.

## 10. What would be needed for finer-grained forecasting?

The next step would require text-heavy signals such as titles, tags, descriptions, and comments.  
At that point, topic modeling or embedding-based clustering would be needed to forecast not only broad categories but also finer content formats inside each category.

## One-line takeaway

This project is best understood as a **Top-5 rising-category selection model over the next 4 weeks**, not as a general-purpose predictor for all YouTube content.
