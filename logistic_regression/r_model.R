attach(train_data)

train_data$Winner <- factor(train_data$Winner)
eval_data$Winner <- factor(eval_data$Winner)

model <- glm(Winner ~ T0OrdinalRank + T1OrdinalRank + T02ptpct + T12ptpct + 
              T03ptpct + T13ptpct + T0APG + T1APG + T0BPG + T1BPG +
              T0DRPG + T1DRPG + T0FPG + T1FPG + T0FTpct + T1FTpct +
              T0ORPG + T1ORPG + T0PPG + T1PPG + T0SPG + T1SPG,
             family = "binomial")

train_predictions <- ifelse(predict(model, train_data) < 0.5, 0, 1)

detach(train_data)
attach(eval_data)

eval_predictions <- ifelse(predict(model, eval_data) < 0.5, 0, 1)

sum(train_predictions == train_data$Winner) / length(train_predictions)
sum(eval_predictions == eval_data$Winner) / length(eval_predictions)