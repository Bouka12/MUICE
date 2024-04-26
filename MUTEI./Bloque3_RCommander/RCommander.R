
Dataset <- read.table("C:/Users/BOUKA/Downloads/Datos_Radiacion_1.txt", header=TRUE, stringsAsFactors=TRUE, 
  sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
editDataset(Dataset)
Dataset <- read.table("C:/Users/BOUKA/Downloads/Datos_Radiacion_1.txt", header=TRUE, stringsAsFactors=TRUE, 
  sep="\t", na.strings="NA", dec=",", strip.white=TRUE)
RegModel.2 <- lm(Abateria~DJ+HR+NHL+RD+RDF+RG+Tm+Vv, data=Dataset)
summary(RegModel.2)
oldpar <- par(oma=c(0,0,3,0), mfrow=c(2,2))
plot(RegModel.2)
par(oldpar)
library(HH, pos=4)
load("C:/Users/BOUKA/Downloads/Datos_Radiacion_1.txt", header=TRUE, stringsAsFactors=TRUE, sep="\t", na.strings="NA", dec=",", strip.white=TRUE)
powerTransform(Dataset[,"Abateria"],start=NULL)
nuevodato <-bcPower(Dataset[,"Abateria"], 0.6513367)
nuevodato

with(Dataset, cor.test(DJ, RD, alternative="two.sided", method="pearson"))
library(abind, pos=16)
library(e1071, pos=17)
numSummary(Dataset[,"Abateria", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,
  .75,1))
normalityTest(~Abateria, test="shapiro.test", data=Dataset)
RegModel.3 <- lm(Abateria~NHL+Tm, data=Dataset)
summary(RegModel.3)
with(Dataset, cor.test(DJ, HR, alternative="two.sided", method="pearson"))

model000 <- lm(Abateria ~ Tm + NHL - 1, data = Dataset)
summary(model000)
Dataset <- read.table("C:/Users/BOUKA/Downloads/Datos_Radiacion_2.txt", header=TRUE, stringsAsFactors=TRUE, 
  sep="\t", na.strings="NA", dec=",", strip.white=TRUE)
with(Dataset, cor.test(Abateria, RDF, alternative="two.sided", method="pearson"))
RegModel.5 <- lm(Abateria~NHL+RD+Tm , data=Dataset)
summary(RegModel.5)

RegModelR2 <- lm(Abateria~ NHL + RD + Tm -1, data=Dataset)
summary(RegModelR2)
RegModel.6 <- lm(Abateria~DJ+HR+NHL+RD+RDF+RG+Tm+Vv, data=Dataset)
summary(RegModel.6)
numSummary(Dataset[,"Tm", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
normalityTest(~Abateria, test="shapiro.test", data=Dataset)
numSummary(Dataset[,"Abateria", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,
  .75,1))
numSummary(Dataset[,"RG", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
numSummary(Dataset[,"RDF", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,.75,1))
RegModel.7 <- lm(Abateria~NHL, data=Dataset)
summary(RegModel.7)
with(Dataset, cor.test(Abateria, HR, alternative="two.sided", method="pearson"))
with(Dataset, cor.test(HR, NHL, alternative="two.sided", method="pearson"))
with(Dataset, cor.test(DJ, RD, alternative="two.sided", method="pearson"))
with(Dataset, cor.test(RD, Tm, alternative="two.sided", method="pearson"))
with(Dataset, cor.test(RDF, RG, alternative="two.sided", method="pearson"))
library(HH, pos=4)
load("C:/Users/BOUKA/Downloads/Datos_Radiacion_2.txt", header=TRUE, stringsAsFactors=TRUE, sep="\t", na.strings="NA", dec=",", strip.white=TRUE)
powerTransform(Dataset[,"Abateria"],start=NULL)

