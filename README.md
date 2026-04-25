# AI-Powered Marketing Decision Engine

A machine learning web app that segments retail customers using RFM analysis + K-Means clustering, then prescribes personalized marketing actions per customer.

**Live demo:** https://kubra-mahrukh-marketing-engine.streamlit.app

---

## What it does

Most analytics dashboards *describe* what happened. This engine *prescribes* what to do next.

Given a customer's purchasing history, the engine answers three questions in one shot:

1. **Who is this customer?** → Segment (Champion, At-Risk, New, or Lost) via K-Means
2. **What offer should we send?** → Discount % and offer type from segment-specific business rules
3. **When should we send it?** → Best day-of-week and hour-of-day from their historical purchase pattern

## Results

Trained on the UCI Online Retail II dataset (~525,000 transactions, ~4,300 unique customers).

| Segment | Customers | % of base | Avg. Recency | Avg. Frequency | Avg. Monetary | Recommended action |
|---|---|---|---|---|---|---|
| Champions | 778 | 18% | 14 days | 13.6 orders | $7,382 | VIP early access, no discount |
| At-Risk | 1,211 | 28% | 80 days | 4.1 orders | $1,760 | 20% win-back |
| New / Promising | 945 | 22% | 24 days | 2.0 orders | $532 | 15% second-purchase |
| Lost / Hibernating | 1,378 | 32% | 191 days | 1.3 orders | $306 | 40% last-chance |

**Headline insight:** 60% of the customer base is slipping or gone. Only 18% are Champions. The engine helps a marketing team reallocate spend from broad blasts to segment-specific campaigns.

## How it works