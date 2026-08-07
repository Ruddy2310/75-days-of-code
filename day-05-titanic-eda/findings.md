
KEY FINDINGS - Titanic EDA (Day 5)
------------------------------------
1. Women had a significantly higher survival rate than men across all classes,
   especially in 1st and 2nd class (~90%+ survival for women in 1st class).
2. Younger passengers (children) had a noticeably higher survival rate,
   consistent with 'women and children first' evacuation priority.
3. Passengers who paid higher fares (likely 1st class) had better survival odds -
   fare and survival show a positive correlation.
4. 'Age' and 'fare' both contained outliers, handled via IQR detection.
   Missing 'age' values were imputed with median, 'embarked' with mode,
   and 'deck' was dropped due to excessive missingness.
