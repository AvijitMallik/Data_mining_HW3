```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
library(tidyverse)
library(lubridate)
library(randomForest)
library(gbm)
library(pdp)
library(modelr)
library(rsample)
library(rpart)
library(rpart.plot)
library(caret)
library(textir)
library(corrplot)
library(gridExtra)
library(GGally)
library(e1071)
library(ggthemes)
library(scales)
library(class) 
library(ggmap)
```


```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
# read in data
dengue <- read.csv("C:/Users/Avijit Mallik/Desktop/Data Mining/ECO395M-master/data/dengue.csv")
dengue = read.csv('../data/dengue.csv')
summary(dengue)
```



```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
set.seed(430)
dengue$season = factor(dengue$season)
dengue$city = factor(dengue$city)
dengue_split =  initial_split(dengue, prop=0.8)
dengue_train = training(dengue_split)
dengue_test  = testing(dengue_split)
```


First, we use CART model.


```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
dengue_tree_train = rpart(total_cases ~ city + season + specific_humidity +precipitation_amt, data=dengue_train,
                          control = rpart.control(cp = 0.000015))
# CV error is within 1 std err of the minimum
cp_1se = function(my_tree) {
  out = as.data.frame(my_tree$cptable)
  thresh = min(out$xerror + out$xstd)
  cp_opt = max(out$CP[out$xerror <= thresh])
  cp_opt
}
cp_1se(dengue_tree_train)
# this function actually prunes the tree at that level
prune_1se = function(my_tree) {
  out = as.data.frame(my_tree$cptable)
  thresh = min(out$xerror + out$xstd)
  cp_opt = max(out$CP[out$xerror <= thresh])
  prune(my_tree, cp=cp_opt)
}
# let's prune our tree at the 1se complexity level
dengue_tree_train_prune = prune_1se(dengue_tree_train)
rpart.plot(dengue_tree_train_prune, digits=-5, type=4, extra=1)
plotcp(dengue_tree_train_prune)
```



**Now we use random forest model.**
  
  
  
  
  ```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
forest1 = randomForest(total_cases ~ city + season + specific_humidity + precipitation_amt,
                       data=dengue_train, na.action = na.exclude)
# performance as a function of iteration number
plot(forest1)
yhat_test_dengue = predict(forest1, dengue_test)
plot(yhat_test_dengue, dengue_test$total_cases)
# a variable importance plot: how much SSE decreases from including each var
varImpPlot(forest1)
```





**Finally we model by using gradient Boosting model with Gaussian and Poisson distributions.**
  
  
  
  ```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
forest1 = randomForest(total_cases ~ city + season + specific_humidity + precipitation_amt,
                       data=dengue_train, na.action = na.exclude)
boost1 = gbm(total_cases ~ city + season + specific_humidity + precipitation_amt, 
             data = dengue_train,
             interaction.depth=4, n.trees=500, shrinkage=.01)
# Look at error curve -- stops decreasing much after ~300
gbm.perf(boost1)
yhat_test_gbm = predict(boost1, dengue_test, n.trees=350)
# RMSE
rmse(boost1, dengue_test)
# What if we assume a Poisson error model?
boost2 = gbm(total_cases ~ city + season + specific_humidity + precipitation_amt, 
             data = dengue_train, distribution='poisson',
             interaction.depth=4, n.trees=350, shrinkage=.01)
# Note: the predictions for a Poisson model are on the log scale by default
# use type='response' to get predictions on the original scale
# all this is in the documentation, ?gbm
yhat_test_gbm2 = predict(boost2, dengue_test, n.trees=350, type='response')
# but this subtly messes up the rmse function, which uses predict with default args
# so we need to roll our own calculate for RMSE
(yhat_test_gbm2 - dengue_test$total_cases)^2 %>% mean %>% sqrt
# relative importance measures: how much each variable reduces the MSE
summary(boost1)
```



```{r, echo=FALSE, message=FALSE, warning=FALSE}
rmse_dengue_1 = modelr::rmse(dengue_tree_train_prune, dengue_test)
rmse_dengue_2 = modelr::rmse(forest1, dengue_test)  # a lot lower!
rmse_dengue_3 = modelr::rmse(boost1, dengue_test)
rmse_dengue_4 = (yhat_test_gbm2 - dengue_test$total_cases)^2 %>% mean %>% sqrt
models_dengue_summary = data.frame(
  CART_RMSE = rmse_dengue_1,
  RForest_RMSE = rmse_dengue_2,
  Normal_Boost_RMSE = rmse_dengue_3,
  Poisson_Boost_RMSE = rmse_dengue_4)
models_dengue_summary
```

Based on the out of sample RMSE, the Gaussian Booster model seems to have the best prediction power. 


**Now we plot the partial dependence of 4 variables.**
  
  
  ```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
plot(boost1, 'specific_humidity')
plot(boost1, 'precipitation_amt')
plot(boost1, 'season')
plot(boost1, 'city')
```

The graphs above show the partial dependence (marginal effects) of the chosen variables on total cases of dengue based on the Gaussian boosting model. I have included all 4 variables since all of them seems interesting, especially with the high difference between the two cities, and the Fall season with the other seasons.
Q3
```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
# read in data
greenbuildings = read.csv('../data/greenbuildings.csv')
```



```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
summary(greenbuildings)
set.seed(488)
greenbuildings$renovated = factor(greenbuildings$renovated)
greenbuildings$class_a = factor(greenbuildings$class_a)
greenbuildings$class_b = factor(greenbuildings$class_b)
greenbuildings$LEED = factor(greenbuildings$LEED)
greenbuildings$Energystar = factor(greenbuildings$Energystar)
greenbuildings$green_rating = factor(greenbuildings$green_rating)
greenbuildings$net = factor(greenbuildings$net)
greenbuildings$amenities = factor(greenbuildings$amenities)
greenbuildings1 = greenbuildings %>%
  mutate(revenue = Rent*leasing_rate)
set.seed(488)
greenbuildings1_split =  initial_split(greenbuildings1, prop=0.8)
greenbuildings1_split_train = training(greenbuildings1_split)
greenbuildings1_split_test  = testing(greenbuildings1_split)
```


So I used three random forest models, and one gradient boosting model to measure the efficiency of the predictions.


```{r, echo=FALSE, message=FALSE, warning=FALSE}
set.seed(488)
forest_green = randomForest(revenue ~ . ,
                            data=greenbuildings1_split_train, na.action = na.exclude)
# a variable importance plot: how much SSE decreases from including each var
varImpPlot(forest_green)
rmse_green1 = modelr::rmse(forest_green, greenbuildings1_split_test)

set.seed(488)
forest_green2 = randomForest(revenue ~ Rent + City_Market_Rent + leasing_rate + Electricity_Costs + size + CS_PropertyID + stories + age + green_rating  ,
                             data=greenbuildings1_split_train, na.action = na.exclude)

rmse_green2 = modelr::rmse(forest_green2, greenbuildings1_split_test)  


set.seed(488)
forest_green3 = randomForest(revenue ~ Rent + City_Market_Rent + leasing_rate + Electricity_Costs + size + CS_PropertyID + stories + age + hd_total07  + total_dd_07 + total_dd_07 + green_rating,
                             data=greenbuildings1_split_train, na.action = na.exclude)
rmse_green3 = modelr::rmse(forest_green3, greenbuildings1_split_test)  

boost_green = gbm(revenue ~ Rent + City_Market_Rent + leasing_rate + Electricity_Costs + size + CS_PropertyID + stories +green_rating, 
                  data = greenbuildings1_split_train,
                  interaction.depth=4, n.trees=350, shrinkage=.02)
rmse_green4 = modelr::rmse(boost_green, greenbuildings1_split_test)  


models_green_summary = data.frame(
  RFM1_rmse = rmse_green1,
  RFM2_rmse = rmse_green2,
  RFM3_rmse = rmse_green3,
  Boost_rmse = rmse_green4)
models_green_summary
yhat_green_gbm = predict(boost_green, greenbuildings1_split_test, n.trees=350)
```

**Now we check for the partial dependence of green rating based on the boosting model (the optimal model).**
  
  ```{r, echo=FALSE, message=FALSE, warning=FALSE}
plot(boost_green, 'green_rating')

p4 = pdp::partial(boost_green, pred.var = 'green_rating', n.trees=350)
p4