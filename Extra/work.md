# 2-Minute Explanation For Teacher

## Simple 2-Minute Script

My project is about finding **why students make math mistakes**, not just checking whether the final answer is right or wrong.

I started from a strong Kaggle competition baseline that uses a trained Qwen model to predict the most likely misconception behind a student's explanation.

Then I extended it into a more research-oriented system.

The first improvement is a **misconception knowledge base**. I store structured information about common math misunderstandings, like related concepts, trigger patterns, and short explanations. From that, I build a small graph so the system can see which misconceptions are related.

The second improvement is **knowledge-guided scoring**. Instead of only looking at the student's text, the model also sees short retrieved context from the misconception graph. This helps it make more informed and more explainable predictions.

The third improvement is **counterfactual validation**. After the model predicts a misconception, I rewrite the student's answer as if that misconception were not present, and I score it again. If the score drops a lot, that means the predicted misconception was probably a real cause, not just a random match.

Finally, I combine the original baseline score, the knowledge-based score, and the counterfactual signal using a simple fusion step. This gives a stronger and more interpretable final ranking.

So in short, my work turns a strong competition solution into a system that is not only accurate, but also more explainable, more structured, and more useful for educational analysis.

## Very Short Version

My project predicts student math misconceptions from their written explanations. I improved a strong baseline by adding a misconception knowledge graph, context-aware scoring, and a counterfactual check to verify whether the predicted misconception is really causing the mistake.

## If Teacher Asks "What Is New Here?"

The new part is that I did not rely only on one model prediction. I added:

- a structured misconception knowledge base
- graph-based context retrieval
- a counterfactual validation step
- a final fusion/reranking step

This makes the system more explainable than a normal black-box classifier.

## If Teacher Asks "Why Is It Useful?"

It can help teachers quickly see what type of misunderstanding a student has, instead of only seeing that the answer is wrong. That means feedback can be more targeted and more useful.

## If Teacher Asks "Why Counterfactual Validation?"

Because sometimes a model gives a high score for the wrong reason. The counterfactual step checks whether removing that misconception from the student's explanation actually lowers the score. If it does, the prediction is more trustworthy.
