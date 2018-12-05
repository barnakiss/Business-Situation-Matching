# Business-Situation-Matching

This project aims to find business situations similar to T-Mobile US' Uncarrier initiative in 2013. Aproach was based on the followings:

- We evaluate country markets
- We are after situations which are similar to T-Mobile US in '13 Q1 - since we know it worked!
- We are in search for roles:
    - The 'Incumbent': the operator who may lose most from the Challenger's disruption
    - The 'Dropped Follower(s)': follower operator(s) with too small market share (marginal players - dropped from Benchmark charts)
    - The 'Last Follower': a follower operator on the market who's not too small, i.e. she is basically known by the customers
    - The 'Challenger': the operator who has good chance to disrupt the market


Categorisation:

- The 'Incumbent': the operator with the highest market share
- The 'Last Follower': the operator with enough market share (market share is the lowest compared to the period average but >=5%)
- The 'Dropped Follower(s)': disregarded operator(s) (market share is <15% @ the end of the period)


Business logic:
- Revenue stream:
    - Connectedness towards customers: Subscriptions
    - Customer relationship stability (noise, loss, wastes): Churn
    - Customer monetisation: Market Share


Criteria:

- The 'Last Follower' has lower subscription growth rate than the market average (in the last 4 quarters)
    --> She can capitalise on churn if she solves problem(s)
- The 'Last Follower' has higher churn ratio than the period average
    --> She can capitalise on churn if she solves problem(s)
- The 'Last Follower' is losing market share during the period (negative slope in the last 4 quarters)
    --> She has problem(s) why customers turn away from her. If she solves this problem she can keep & attract more customers
- The 'Last Follower' has lower ARPU (average revenue per user) rate than the market average (in the last 4 quarters)
    --> She has more cost sensitive customers
- The 'Incumbent' high churn ratio (higher than the period average)
    --> The 'Last Follower' can take more easily from the 'Incumbent'

If at least two criteria is met: The 'Last Follower' --> The 'Challenger'
