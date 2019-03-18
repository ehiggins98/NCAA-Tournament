train_data$Winner <- factor(train_data$Winner)
eval_data$Winner <- factor(eval_data$Winner)

model <- glm(train_data$Winner ~ train_data$T0OrdinalRank + train_data$T1OrdinalRank + train_data$T02ptpct + train_data$T12ptpct + 
              train_data$T03ptpct + train_data$T13ptpct + train_data$T0APG + train_data$T1APG + train_data$T0BPG + train_data$T1BPG +
              train_data$T0DRPG + train_data$T1DRPG + train_data$T0FPG + train_data$T1FPG + train_data$T0FTpct + train_data$T1FTpct +
              train_data$T0ORPG + train_data$T1ORPG + train_data$T0PPG + train_data$T1PPG + train_data$T0SPG + train_data$T1SPG,
             family = "binomial")

train_predictions <- ifelse(predict(model, train_data) < 0.5, 0, 1)
eval_predictions <- ifelse(predict(model, eval_data) < 0.5, 0, 1)

sum(train_predictions == train_data$Winner) / length(train_predictions)
sum(eval_predictions == eval_data$Winner) / length(eval_predictions)
sum(eval_predictions[0:1048] == eval_data$Winner) / 1048

summary(model)