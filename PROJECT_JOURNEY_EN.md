# Project Journey

This project did not begin with a model choice.

It began with a question:

> Which YouTube Shorts categories are likely to gain attention in the near future?

At first, that sounds like a standard prediction problem.  
But the more we worked on it, the more we realized that we were not simply trying to explain why an already successful video performed well. We were trying to build a structure that could help us notice **which categories might become more interesting before they are obviously popular**.

This document explains how that question evolved into the final project, what made the work difficult, how we handled those difficulties, and why the project matters beyond a single model score.

## 1. We started with video-level prediction

The first instinct was straightforward:

- views
- likes
- comments
- titles
- tags
- upload time

These are common signals in virality prediction tasks, so it was reasonable to start there.

However, once we organized the data, a problem became obvious.  
Two videos with similar early reactions did not always mean the same thing. Their meaning changed depending on **which category trend they were sitting on top of**.

That changed the central question.

- Instead of asking, “Which individual video will do well?”
- We moved toward: **“Which category is likely to gain attention first?”**

That was the first major turning point of the project.

## 2. The project shifted from videos to category context

From that point on, the project became less about isolated video signals and more about the relationship between:

1. long-term category behavior,
2. early video-level reactions, and
3. the alignment between the two.

The working assumption became:

- categories have different baseline strength,
- videos are influenced by the momentum of the category they belong to,
- and early reaction signals are useful, but more useful when interpreted inside category context.

That is why the final presentation is organized around **category trend + video metadata + early time-series response**, instead of a single video-only prediction pipeline.

## 3. The data naturally split into three streams

By the time the final presentation structure settled, the data was best understood in three groups.

### Historical data

- category-level video performance
- long-term view and engagement patterns
- baseline strength and seasonality

### Current metadata

- video titles
- tags
- upload timing

### Time-series signals

- hourly or early-stage reaction changes
- early growth velocity and momentum

The challenge was not only collecting data.  
The real challenge was combining data that lived on different time scales and had different meanings into one coherent prediction story.

## 4. Data consistency was harder than modeling

The most time-consuming part of the project was not the model itself.  
It was making the data trustworthy enough to support the story we wanted to tell.

### 4-1. Google Trends was not plug-and-play

We wanted to use Google Trends as an external interest signal.  
In practice, the collection process was unstable, and some saved files were not as cleanly weekly as we initially assumed.

That forced us to stop treating external signals as ready-made features.  
We had to verify which files were actually usable, normalize them again, and rebuild the weekly feature table carefully.

### 4-2. Week definitions were inconsistent

Another major issue was the `year_week` axis.  
Different files used slightly different week logic, which meant that two datasets could appear to share the same weekly key while actually pointing to different dates.

This sounds small, but it becomes critical when category trends and time-series features need to be aligned.  
We resolved it by rebuilding the weekly axis with ISO week logic and reconstructing the category-week timeline around explicit dates.

### 4-3. Not every category should be modeled the same way

When we looked at all categories together, some of them were:

- too sparse,
- recently inactive,
- or too uneven in observation length.

That made the broader prediction problem unstable.

One of the key lessons here was that **reducing the scope can be part of solving the problem**, not a weakness to hide.

## 5. We redefined the problem before refining the model

One of the most important parts of this project is that we did not treat model choice as the first answer.

Before worrying about architecture, we had to clarify:

- what exactly we were predicting,
- at what unit we were predicting it,
- and which signal should be treated as primary.

That is why the final presentation explains the model in four stages:

1. Data input
2. Category Trend Model
3. Video + Time-Series Model
4. Final Prediction

This structure matters because it explains the logic of the project:

- read category flow first,
- read video reaction second,
- combine them at the end.

The final presentation is less about naming a sophisticated deep learning block and more about showing how the prediction task was structured.

## 6. The derived features follow that logic

The four featured variables in the presentation were not decorative additions.  
They were the numeric trace of the project’s reasoning.

- **Category Future Interest Score**
- **Category Momentum**
- **Early Interest Velocity**
- **Trend Alignment Score**

Together, they answer four linked questions:

- How likely is a category to gain attention?
- Is that category rising or slowing down?
- Is this video spreading quickly in its early stage?
- Does the video behavior match the category trend?

The value of these variables is not only that they can improve a model.  
Their value is that they make the project’s logic easier to explain.

## 7. The repository also preserves implementation experiments

The final presentation focused on framing, structure, and feature strategy.  
The repository also keeps additional implementation work that pushed the idea into more concrete experiments.

Examples:

- [train_active_category_rank_bigru.py](./train_active_category_rank_bigru.py)
- [RESULTS.md](./RESULTS.md)

These files reflect a narrower and more quantitative experimental branch that used:

- recent weekly sequences,
- external variables,
- a BiGRU-based implementation,
- and Top-5 selection outputs.

Those experiments are useful, but the best way to read them is as **implementation traces behind the presentation narrative**, not as the only definition of the project.

## 8. What this project means

The point of this repository is not just that a model was trained.

The more meaningful outcomes are:

1. we identified the limit of purely video-level prediction,
2. we reframed the task around category trend and early reaction together,
3. we learned that data consistency problems can be as important as the model itself,
4. and we separated the final presentation story from deeper implementation experiments.

In that sense, this project is less about “we applied deep learning once” and more about **how we reshaped a prediction problem into something more coherent and defensible**.

## 9. How this repository should be read

The repository makes the most sense if you read it in two layers.

### Layer 1: the final presentation

This explains:

- what we wanted to predict,
- why that structure was needed,
- which feature strategy we used,
- and what kind of impact we expected.

### Layer 2: the experimental trail behind it

This includes:

- data issues,
- weekly alignment fixes,
- external signal cleanup,
- and more concrete implementation attempts.

If you read only one layer, the project can look flatter than it really was.  
The combination of both layers is what shows how the project actually evolved.

## One-sentence summary

This project documents how a simple video virality question gradually turned into a category-aware YouTube Shorts forecasting project that combines long-term category trends with early video response signals.
