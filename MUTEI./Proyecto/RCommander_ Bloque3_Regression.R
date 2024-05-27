
Dataset <- readXL("C:/Users/BOUKA/OneDrive/Bureau/MUICE/AW/Chart-Studio/student+performance/student/mat.xlsx", 
  rownames=FALSE, header=TRUE, na="", sheet="student-mat", stringsAsFactors=TRUE)
summary(Dataset)
library(abind, pos=16)
library(e1071, pos=17)
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$activities, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$address, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$famsize, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$famsup, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$Fjob, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$guardian, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$higher, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$internet, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$Mjob, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$nursery, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$paid, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$Pstatus, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$reason, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$romantic, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$school, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$schoolsup, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"G3", drop=FALSE], groups=Dataset$sex, statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
sapply(Dataset, function(x)(sum(is.na(x)))) # NA counts
Tapply(studytime ~ reason, mean, na.action=na.omit, data=Dataset) # mean by groups
cor(Dataset[,c("absences","age","Dalc","failures","famrel","Fedu","freetime","G1","G2","G3","goout","health","Medu","studytime","traveltime","Walc")], use="complete")
normalityTest(~G3, test="shapiro.test", data=Dataset)
normalityTest(~G2, test="shapiro.test", data=Dataset)
normalityTest(~G1, test="shapiro.test", data=Dataset)
normalityTest(~absences, test="shapiro.test", data=Dataset)
normalityTest(~age, test="shapiro.test", data=Dataset)
normalityTest(~Dalc, test="shapiro.test", data=Dataset)
normalityTest(~failures, test="shapiro.test", data=Dataset)
normalityTest(~famrel, test="shapiro.test", data=Dataset)
normalityTest(~Fedu, test="shapiro.test", data=Dataset)
normalityTest(~freetime, test="shapiro.test", data=Dataset)
normalityTest(~goout, test="shapiro.test", data=Dataset)
normalityTest(~health, test="shapiro.test", data=Dataset)
normalityTest(~Medu, test="shapiro.test", data=Dataset)
normalityTest(~studytime, test="shapiro.test", data=Dataset)
normalityTest(~traveltime, test="shapiro.test", data=Dataset)
normalityTest(~Walc, test="shapiro.test", data=Dataset)
RegModel.1 <- lm(G3~absences+age+Dalc+failures+famrel+Fedu+freetime+G1+G2+goout+health+Medu+studytime+traveltime+Walc, data=Dataset)
summary(RegModel.1)
RegModel.2 <- lm(G3 ~ absences + age + famrel + G1 + G2, data=Dataset)
summary(RegModel.2)
RegModel.no_intercept <- lm(G3 ~ absences + age + famrel + G1 + G2 - 1, data=Dataset)
summary(RegModel.no_intercept)
plot(RegModel.no_intercept$fitted.values, rstandard(RegModel.no_intercept),
     main="Residuals vs Fitted", xlab="Fitted values", ylab="Standardized residuals")
abline(h=0, col="red")
if (!require(lmtest)) install.packages("lmtest")
library(lmtest)
bp_test <- bptest(RegModel.no_intercept)
print(bp_test)
Dataset$log_G3 <- log(Dataset$G3 + 1)  # Adding 1 to handle zero values if present
RegModel.log <- lm(log_G3 ~ absences + age + famrel + G1 + G2 - 1, data=Dataset)
summary(RegModel.log)

# Residuals vs Fitted plot
plot(RegModel.log$fitted.values, rstandard(RegModel.log),
     main="Residuals vs Fitted", xlab="Fitted values", ylab="Standardized residuals")
abline(h=0, col="red")

RegModel.log.no_age <- lm(log_G3 ~ absences + famrel + G1 + G2 - 1, data=Dataset)
summary(RegModel.log.no_age)
AIC(RegModel.log, RegModel.log.no_age)

# Residuals vs Fitted plot
plot(RegModel.log$fitted.values, rstandard(RegModel.log),
     main="Residuals vs Fitted", xlab="Fitted values", ylab="Standardized residuals")
abline(h=0, col="red")

# Normal Q-Q plot
qqnorm(rstandard(RegModel.log))
qqline(rstandard(RegModel.log), col = "red")

# Breusch-Pagan test for homoscedasticity
if (!require(lmtest)) install.packages("lmtest")
library(lmtest)
bptest(RegModel.log)
bptest(RegModel.log.no_age)

# Residuals vs Fitted plot
plot(RegModel.log.no_age$fitted.values, rstandard(RegModel.log.no_age),
     main="Residuals vs Fitted", xlab="Fitted values", ylab="Standardized residuals")
abline(h=0, col="red")

# Normal Q-Q plot
qqnorm(rstandard(RegModel.log.no_age))
qqline(rstandard(RegModel.log.no_age), col = "red")


# Square root transformation of the dependent variable
Dataset$sqrt_G3 <- sqrt(Dataset$G3)

# Fit the model with the square root transformed variable
RegModel.sqrt <- lm(sqrt_G3 ~ absences + age + famrel + G1 + G2 - 1, data=Dataset)
summary(RegModel.sqrt)

# Check diagnostics
par(mfrow=c(2,2))
plot(RegModel.sqrt)
bptest(RegModel.sqrt)

# Inverse transformation of the dependent variable
Dataset$inv_G3 <- 1 / (Dataset$G3+1)

# Fit the model with the inverse transformed variable
RegModel.inv <- lm(inv_G3 ~ absences + age + famrel + G1 + G2 - 1, data=Dataset)
summary(RegModel.inv)

# Check diagnostics
par(mfrow=c(2,2))
plot(RegModel.inv)
bptest(RegModel.inv)

# Doing BOX-COX Transformation
# Find the optimal lambda for the Box-Cox transformation
# Load the HH package
if (!require(HH)) install.packages("HH")
library(HH)
# Add a small positive value to the variable to make all values strictly positive
small_positive_value <- 0.001
Dataset$shifted_G3 <- Dataset$G3 + small_positive_value

# Find the optimal lambda for the Box-Cox transformation
power_transform <- powerTransform(Dataset[,"shifted_G3"])
optimal_lambda <- power_transform$lambda
optimal_lambda

# Apply the Box-Cox transformation with the optimal lambda
Dataset$boxcox_G3 <- bcPower(Dataset[,"shifted_G3"], optimal_lambda)

# Fit the model with the transformed response variable
RegModel.boxcox <- lm(boxcox_G3 ~ absences + age + famrel + G1 + G2 - 1, data=Dataset)
summary(RegModel.boxcox)

# Check diagnostics
par(mfrow=c(2,2))
plot(RegModel.boxcox)
bptest(RegModel.boxcox)

# Diagnostic plots for the model
par(mfrow=c(2,2))
plot(RegModel.boxcox)

# Create a boxplot for G3
boxplot(Dataset$G3, main="Boxplot of G3", ylab="G3")

# Identify outliers
Q1 <- quantile(Dataset$G3, 0.25)
Q3 <- quantile(Dataset$G3, 0.75)
IQR <- Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Identify the outliers
outliers <- which(Dataset$G3 < lower_bound | Dataset$G3 > upper_bound)
print(outliers)

# Fit the model without the G1 predictor
RegModel.boxcox_no_G1 <- lm(boxcox_G3 ~ absences + age + famrel + G2 - 1, data=Dataset)
summary(RegModel.boxcox_no_G1)

# Check diagnostics for the new model
par(mfrow=c(2,2))
plot(RegModel.boxcox_no_G1)

# Perform Breusch-Pagan test for heteroscedasticity
if (!require(lmtest)) install.packages("lmtest")
library(lmtest)
bptest(RegModel.boxcox_no_G1)

# Compute weights based on the residuals of the current model
residuals <- resid(RegModel.boxcox_no_G1)
weights <- 1 / (residuals^2)

# Fit the WLS model using the computed weights
RegModel.wls <- lm(boxcox_G3 ~ absences + age + famrel + G2 - 1, data=Dataset, weights=weights)
summary(RegModel.wls)

# Check diagnostics for the WLS model
par(mfrow=c(2,2))
plot(RegModel.wls)
bptest(RegModel.wls)


