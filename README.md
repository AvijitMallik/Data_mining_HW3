# Data_mining_HW3
## 2nd problem, Tree modeling: dengue cases

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
```




The goal of this exercise is predict the revenue per square foot per calender year of about 8,000 commercial rental properties across the US. In addition, some of those properties are green certified which means they got green certification from either LEED or Energystar. Another question we want to answer is whether being green certified will raise your revenue or not. Now let's move on the methodloly used to predict the revenue. 

First of all, I have mutated a new column to calculate the revenue per square foot per calender year based on the original data. In order to do that, I took the product of rent and leasing_rate. We need to do that to get unbiased prediction results since the occupancy or the rent_rate alone won't reflect the revenue.

Next, I needed to make sure that some of the variables are dummy variables, so I used the factor command on the 0/1 variables. Then, I start working on the model by splitting the data to training set (80%) and testing set (20%). I trained the data to predict revenue using random forest model. First model used is the base model, basically by regressing revenue on all variables, then check for the importance of each variable in order to try other models and compare them based on the results of their root mean squared errors.


Now we move to try other possible models based on the results of their importance. We can notice how green_rating is not an important variable in the model which indicates there will not be significant partial effect of the green certification on the revenue. However, I have to include it in order to observe the real partial effect using the partial dependence algorithm. 

So, after my base model, the 2nd model included 9 variables with different importance level for each one of them. The 3rd model had 12 variables with many more less important variables. I worried that it is going to overfit the model, so now we got to check the rmse for each one of them and compare it with what we got in the base model. So, the 2nd model with the 9 variables got slightly lower rmse than the first model which regressing revenue on all variables. However, should we stop now? since we are looking for the best predictive model, it is going to be worth it to try to model using gradient boosting model with the same variables of the best performing random forest model. 

After trying different shrinkage rates, I have succeeded in over-performing the 2nd model by having rmse = 134 compared to the best random forest model which was 167. So, I decided to select the boosting model to answer the question of the how much green certification is going to affect my revenue assuming all other variables are constant. So, I predicted the average value for both certified and certified, and as we can see, it has no partial effect at all. The values basically are the same and the plot gives us the same answer too.




## 4th problem, Predictive model building: California housing



```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
 CAhousing = read.csv('../data/CAhousing.csv')
```


```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
CAhousing1 = CAhousing %>%
  mutate(totalRooms_st = totalRooms/households) %>%
  mutate(totalBedrooms_st = totalBedrooms/households)
set.seed(1208)
CAhousing1_split =  initial_split(CAhousing1, prop=0.8)
CAhousing1_split_train = training(CAhousing1_split)
CAhousing1_split_test  = testing(CAhousing1_split)
```

We compare 4 different models to check which model is the optimal.

```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
forest_CA1 = randomForest(medianHouseValue ~ . ,
                       data=CAhousing1_split_train, na.action = na.exclude)
# a variable importance plot: how much SSE decreases from including each var
varImpPlot(forest_CA1)
  rmse_CA1 = modelr::rmse(forest_CA1, CAhousing1_split_test)
  
  
  ##Now we adjust the model based on the important plot
  
  forest_CA2 = randomForest(medianHouseValue ~ medianIncome + longitude + latitude + totalRooms_st  ,
                       data=CAhousing1_split_train, na.action = na.exclude)
  
  
    rmse_CA2 = modelr::rmse(forest_CA2, CAhousing1_split_test)
    
    ##a third adjusted model
    
    
      forest_CA3 = randomForest(medianHouseValue ~ medianIncome + longitude + latitude + totalRooms_st + population + housingMedianAge ,
                       data=CAhousing1_split_train, na.action = na.exclude) 
  
  
    rmse_CA3 = modelr::rmse(forest_CA3, CAhousing1_split_test)  # The lowest value
    
    
    
    ##Let's try gradient boosting model
    
boost_CA = gbm(medianHouseValue ~ medianIncome + longitude + latitude + totalRooms_st + population + housingMedianAge, 
             data = CAhousing1_split_train,
             interaction.depth=4, n.trees=350, shrinkage=.08)
  rmse_boost = modelr::rmse(boost_CA, CAhousing1_split_test)  # not better than CA3
  
  
  ##I select CA3 model to be the optimal one
  
  yhat_test_CA3 = predict(forest_CA3, CAhousing1_split_test)
  
  ##create new columns for the predicted values and residuals
CAhousing1_split_test1 = CAhousing1_split_test %>%
  mutate(yhat = yhat_test_CA3) 
CAhousing1_split_test1 = CAhousing1_split_test1 %>%  
mutate (resid =  medianHouseValue - yhat)
  
  
```
**Now we check which model out of the 4 has the lowest root mean squared errors.**


```{r, echo=FALSE, message=FALSE, warning=FALSE}
CA_models_summary = data.frame(
CA_RFM1_rmse = rmse_CA1,
CA_RFM2_rmse = rmse_CA2,
CA_RFM3_rmse = rmse_CA3,
CA_Boost_rmse = rmse_boost)
CA_models_summary
```



```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
CA_plot_org = ggplot(CAhousing1) +
 aes(x = longitude, y = latitude, colour = medianHouseValue) +
 geom_point(shape = "circle", 
 size = 1.5) + 
 labs(title = "California Median House Value", 
 subtitle = "Original Data Plot") +
 theme_minimal() + scale_color_continuous(labels = scales::comma)
CA_plot_yhat = ggplot(CAhousing1_split_test1) +
 aes(x = longitude, y = latitude, colour = yhat) +
 geom_point(shape = "circle", 
 size = 1.5)  +
 labs(title = "California Median House Value", 
 subtitle = "Predicted Plot") +
 theme_minimal() + scale_color_continuous(labels = scales::comma)
CA_plot_resid = ggplot(CAhousing1_split_test1) +
 aes(x = longitude, y = latitude, colour = resid) +
 geom_point(shape = "circle", 
 size = 1.5) +
 labs(title = "California Median House Value", subtitle = "Residuals Plot") +
 theme_minimal() + scale_color_continuous(labels = scales::comma)
CA_plot_org
CA_plot_yhat
CA_plot_resid
```


For this model, the aim was to predict the median house value in California State. In order to do that, I have used machine learning tools to provide me with reliable predictions. So, I have used the random forest model which utilize the interaction effects of the variables. First, I mutated to new columns to standardized the total rooms and total bedrooms by dividing each variable by households variable. Then, I split the data into 80% training set and 20% testing set and regress medianHousevalue on all the variables to test for the importance of each variables afterward. Next, I did two other specification models with different variables based on the results of the variables importance. The third model has the lowest root mean squared error which equals to 47,989. In order to check for room of improvements, I run a gradient boosting model with many different shrinkage rates, but I could not have a lower rmse value than the selected random forest model. 

So, I decided to continue with the results of the optimal random forest model and predict the median housing values based on the testing set. Then I plot the original observation which has the shape of California State, the predicted values based on the testing set, and the estimated residuals which is the difference between the two. 
