#SEEING THE CURRENT WORKING DIRECTORY
getwd()

#LOADING THE DATASET

data<-read.csv("all_currencies.csv")
head(data)
names(data)
tail(data)
nrow(data) #there are 632218 observations

str(data)
summary(data)
# from this we see that there are missing values in volume and market cap

data_len<-nrow(data)
data_len

#here we have symbols as strings so we are changing them into categorical values

unique(data$Symbol)
length(unique(data$Symbol))
sym_lem<-length(unique(data$Symbol))
uni_sym<-unique(data$Symbol)
#v<-data$Symbol[10000]==uni_sym[1:sym_lem]
#which(grepl(TRUE,v)

for(i in 1:data_len)
{
v<-data$Symbol[i]==uni_sym[1:sym_lem]
n<-which(grepl(TRUE,v))
data$Sym_value[i]<-n
}

#writing this into a csv file
#write.csv(data,"new_all_currencies.csv")
new_data<-read.csv("new_all_currencies.csv")
head(new_data)
sum(new_data$Sym_value==1)


for(i in 1:sym_lem)
{
value<-sum(new_data$Sym_value==i)
new_data$count_sym[new_data$Sym_value==i]<-value
}
tail(new_data)
sum(new_data$Sym_value==887)#checking

#Removing duplicate and unneccessary columns

names(new_data)
new_data<-new_data[,-c(1,2)]
head(new_data)
new_data<-new_data[,-2]
head(new_data)
new_data<-new_data[,-8]
head(new_data)

/*

#Predicting the missing values in Market Cap
#--------------------------------------------
#--------------------------------------------

#spliting the dataset into missing and non-missing values and dropping the volume column temporarily
mc<-is.na(new_data$Market.Cap)
sum(mc==TRUE)
b1<-!mc
mc_data<-new_data[,-6]
mc_train<-mc_data[b1,]
nrow(mc_train)
nrow(new_data)
nrow(new_data)-nrow(mc_train)  #checking
mc_test<-mc_data[!b1,]
nrow(mc_test)
#write.csv(mc_train,"mc_train.csv")
#write.csv(mc_test,"mc_test.csv")

#spliting input and output features for training and testing data
mc_train<-mc_train[,-1]
mc_train<-mc_train[,-1]

mc_train_x<-mc_train[,-5]
head(mc_train_x)
mc_train_y<-mc_train[,5]
head(mc_train_y)
#mc_test<-read.csv("mc_test.csv")
#head(mc_test)

*/
#new_data<-new_data[,-1]
#head(new_data)

#removing the observations that has missing values in Market Cap and Volume
mv1<-is.na(new_data$Market.Cap)
sum(mv1)
head(mv1)
bad1<-!mv1
new_data<-new_data[bad1,]
nrow(new_data)
mv2<-is.na(new_data$Volume)
sum(mv2)
bad2<-!mv2
new_data<-new_data[bad2,]
nrow(new_data)   #now we have a total of 562506 observations

#Removing outliers and exploratory data analysis
plot(new_data$Open)
sum(new_data$Open>500000) #24 observations
o1<-new_data$Open<=500000
head(o1)
new_data<-new_data[o1,]
nrow(new_data)

plot(new_data$High)
bad2<-new_data$High<500000
sum(bad2==F)
new_data<-new_data[bad2,]
nrow(new_data)

plot(new_data$Low)
bad3<-new_data$Low<=2.5e+05
sum(bad3==TRUE)
new_data<-new_data[bad3,]
nrow(new_data)

plot(new_data$Close)
bad4<-new_data$Close<250000
sum(bad4==FALSE)
new_data<-new_data[bad4,]
nrow(new_data)

plot(new_data$Market.Cap)
bad5<-new_data$Market.Cap<2.0e+11
sum(bad5==FALSE)
sum(bad5==T)
new_data<-new_data[bad5,]
nrow(new_data)

head(new_data)
hist(new_data$Volume)
plot(new_data$Volume)
bad6<-new_data$Volume<1.25e+10
sum(bad6==FALSE)
new_data<-new_data[bad6,]
nrow(new_data)
#after removing all the outliers we have a total of 562387 observations,that is we lost 11.04(nearly 69000) observations in the cleaning process

#correlation between variables

library(ggcorrplot)
#cor_mat<-cor(new_data[,-4])
cor_mat<-cor(new_data[,-1])
cor_mat
d<-dim(cor_mat)
l1<-d[1]
l2<-d[2]

ggcorrplot(cor_mat)  #heatmap of correlation matrix
#displaying the highly correlated variables

col_names<-colnames(cor_mat)
row_names<-colnames(cor_mat)
for(i in 1:l1)
{
for(j in 1:6)
{
if(cor_mat[i,j]>=0.85)
{
print(c(row_names[i],"-",col_names[j]))
}
}
}

new_data<-new_data[,-c(2,3,4)]
head(new_data)

#write.csv(new_data,"model_data.csv")  #this data can be used for model building

model_data<-read.csv("model_data.csv")
nrow(model_data)
summary(model_data)
str(model_data)
model_data<-model_data[,-1]

#feature selection

library(glmnet)
library(MASS)
library(FSelector)

information.gain(Market.Cap~.,data=model_data)
feat_imp_2<-cv.glmnet(as.matrix(model_data[,-4]),model_data[,4],type.measure="mse",alpha=1)
feat_imp_2
imp2<-coef(feat_imp_2,s="lambda.min",exact=TRUE)
inds<-which(imp2!=0)
var2<-row.names(imp2)[inds]
var2  #low is the most important value and market cap is the less important

#train and test split in the ratio of 70:30

set.seed(2)
id<-sample(2,nrow(model_data),prob=c(0.7,0.2),replace=TRUE)
head(id)
sum(id==1)
sum(id==2)
model_train<-model_data[id==1,]
model_test<-model_data[id==2,]
nrow(model_train)
nrow(model_test)

#write.csv(model_train,"model_train.csv")  #training data
#write.csv(model_test,"model_test.csv")    #testing data

#spliting input and output features for training and testing datset
head(model_train) 
model_train<-model_train[,-1]
train_inputs<-model_train[,-2]
train_output<-model_train[,2]
head(model_test)
model_test<-model_test[,-1]
test_inputs<-model_test[,-2]
test_output<-model_test[,2]

#Building models and the best fit model in choosen for deployement


