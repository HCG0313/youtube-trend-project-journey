# Project Journey

This project did not begin with the question, “Which model should we use?”

It began with a simpler question:

> Which YouTube categories are likely to rise over the next few weeks?

At first, we thought this could be handled as a standard video-level prediction task.  
Views, likes, comments, titles, and tags seemed like a reasonable starting point.

But the more we worked on the data, the more obvious it became that the real problem was not simply “predict a successful video.” What we really wanted was to **identify which categories were likely to gain momentum before they were already clearly popular**.

This document explains how that question evolved into the final project, what made the work difficult, how the problem was redefined, and why the final repository is about more than a single deep learning model.

## 1. We started from video-level signals

The first approach was straightforward:

- views
- likes
- comments
- titles
- tags
- upload time

These are common signals in virality prediction, so the starting point was reasonable.

However, once we began organizing the data, a limitation appeared quickly.  
Two videos with similar early reactions did not always mean the same thing. Their meaning changed depending on **which category trend they belonged to**.

That changed the central question.

- Instead of asking, “Which video will do well?”
- We moved toward: **“Which category is likely to rise first?”**

That was the first major turning point of the project.

## 2. The project shifted toward category context

From that point on, the project became less about isolated video performance and more about the relationship between:

1. long-term category behavior,
2. recent weekly reaction patterns,
3. and early signals from individual content.

The underlying assumptions became:

- categories have different baseline strength,
- video performance is influenced by the momentum of the category it belongs to,
- and early reaction signals become more useful when interpreted inside category context.

That is why the final project is framed around **category trend + recent time-series response + external variables**, rather than a purely video-level prediction pipeline.

## 3. The data was organized into three streams

By the time the final presentation structure settled, the data was best understood in three groups.

### Historical data

- category-level video performance
- long-term view trends
- baseline strength and seasonality

### Current metadata

- titles
- tags
- upload timing

### Time-series and external features

- the most recent 12 weeks of category response
- Google Trends search interest
- calendar variables such as holidays, long weekends, vacation periods, and exam periods

The challenge was not simply collecting more data.  
The real difficulty was combining data that lived on different time scales into one coherent forecasting task.

## 4. Data consistency was harder than the model itself

The most time-consuming part of the project was not the model architecture.  
It was making the data consistent enough to support the question we were asking.

### 4-1. Google Trends was not ready to use as-is

We wanted to use Google Trends as an external interest signal.  
In practice, the collection process was unstable, and the saved files were not always reliable enough to use directly.

That meant we had to:

- verify which files were actually usable,
- rebuild category-level weekly search features,
- and normalize them before using them as model inputs.

### 4-2. Weekly keys were inconsistent

Another major issue was the weekly axis itself.  
Different files did not always interpret `year_week` in the same way.

That sounds small, but it becomes critical when category trends and recent time-series features need to be aligned.  
We resolved this by rebuilding the weekly axis with ISO week logic and explicitly reconstructing the category-week timeline.

### 4-3. Not every category should be modeled equally

When we examined all categories together, some were:

- too sparse,
- too inactive in recent weeks,
- or too uneven in observation length.

That made the broader forecasting problem unstable.

One of the key lessons here was that **reducing the problem scope can itself be part of solving the problem**.

## 5. We redefined the task around the core 10 categories

The original goal was broad: predict future interest across YouTube categories in general.  
But once we looked at the data distribution, it became clear that treating all categories under the same assumptions would weaken reliability.

So the final setup became:

- use 16 sufficiently active categories for model construction,
- and fix the final presentation and evaluation around 10 core categories.

This was not simply a way to make the performance look better.  
It was a way to turn an unstable broad problem into a **more reliable deep learning forecasting task**.

## 6. The model explanation followed that redefinition

In the final presentation, the model is explained in four stages:

1. Data input
2. Category Trend Model
3. BiGRU-based time-series model
4. Final Prediction

What matters here is not only the model name.  
What matters is the logic:

- read category flow first,
- read the recent 12-week pattern next,
- combine external variables,
- then produce the final Top-N prediction.

The presentation therefore emphasizes **problem structure** more than algorithm branding.

## 7. Why BiGRU was chosen

The final model was based on BiGRU, a standard time-series deep learning model in the RNN family.

- RNN provides the basic sequential modeling framework,
- GRU improves long-range dependency handling,
- BiGRU extends GRU in both directions and makes it easier to read the context of a short sequence more completely.

Because our task depended heavily on interpreting the recent 12-week pattern, BiGRU was a natural fit for the final model.

Category embedding was also added so the model could distinguish between the structural characteristics of categories such as beauty, gaming, economy, education, or vlog content.

## 8. The derived features followed the same logic

The final presentation emphasized derived features such as:

- current response scale,
- engagement rate,
- recent 4-week moving averages,
- momentum ratio,
- competition score,
- opportunity score,
- rise label,
- and rank-up target.

These features were not included just to enrich the input table.  
They were meant to capture:

- how large current response is,
- whether recent trends are accelerating,
- whether a category is becoming relatively stronger than others,
- and whether the next four weeks should be labeled as rising.

In other words, the feature engineering strategy was closely tied to the way the forecasting problem itself was defined.

## 9. The final evaluation focused on Top-5 selection

The final project was ultimately framed as:

**Select the Top-5 categories among the core 10 that are most likely to rise over the next four weeks.**

The final metrics were:

- Accuracy: 0.767
- Balanced Accuracy: 0.798
- Precision: 0.944
- Recall: 0.739
- F1-score: 0.829
- ROC AUC: 0.845
- Precision@5: 0.900
- Recall@5: 0.801
- HitRate@5: 1.000
- NDCG@5: 0.881

The final predicted Top-5 categories were:

1. Pets
2. Mukbang
3. Economy
4. Vlog
5. Education

These results were not based on raw popularity alone.  
They reflected the combination of rise probability, rank-up probability, recent response trend, search interest, and relative position signals.

## 10. What this project means

The most meaningful outcome of this project is not a single performance number.

The more important points are:

1. we identified the limit of purely video-level prediction,
2. we reframed the task around category flow and recent time-series patterns,
3. we confirmed that data consistency issues matter as much as the model,
4. and we built a more stable forecasting problem around the core 10 categories.

So this repository is less about “we trained one deep learning model” and more about **how a prediction problem was redefined, stabilized, and then implemented in a defensible way**.

## One-sentence summary

This project documents how a broad video virality question gradually became a BiGRU-based forecasting task that predicts which YouTube core categories are most likely to rise over the next four weeks.
