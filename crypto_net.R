data<-read.csv("model_train.csv")
head(data)
data<-data[,-c(1,2)]
head(data)

#use close,volume and count as input and market cap as output

#scaling the inputs using min-max scalar
unique(data$count_sym)
labels<-data$count_sym

mins<-apply(data,2,min)
maxs<-apply(data,2,max)
scaled<-as.data.frame(scale(data,center=mins,scale=maxs-mins))

set.seed(500)
id<-sample(2,nrow(scaled),prob=c(0.75,0.25),replace=TRUE)
train<-scaled[id==1,]
test<-scaled[id==2,]
nrow(train)
nrow(test)

#Splitting indp and dept variables

inputs<-train[,-3]
output<-train$Market.Cap
head(output)

len1<-nrow(inputs)
len1  #printing the number of rows in the inputs df

a0<-matrix(nrow=3,ncol=len1)
w1<-matrix(nrow=4,ncol=3)
w2<-matrix(nrow=4,ncol=4) 
w3<-matrix(nrow=4,ncol=4)
w4<-matrix(nrow=1,ncol=4)

b1<-matrix(nrow=4,ncol=len1)
b2<-matrix(nrow=4,ncol=len1)
b3<-matrix(nrow=4,ncol=len1)
b4<-matrix(nrow=1,ncol=len1)

z1<-matrix(nrow=4,ncol=len1)
z2<-matrix(nrow=4,ncol=len1)
z3<-matrix(nrow=4,ncol=len1)
z4<-matrix(nrow=1,ncol=len1)

a1<-matrix(nrow=4,ncol=len1)
a2<-matrix(nrow=4,ncol=len1)
a3<-matrix(nrow=4,ncol=len1)

#matrix for calculating gradients

dw1<-matrix(nrow=4,ncol=3)
dw2<-matrix(nrow=4,ncol=4)
dw3<-matrix(nrow=4,ncol=4)
dw4<-matrix(nrow=1,ncol=4)

db1<-matrix(nrow=4,ncol=len1)
db2<-matrix(nrow=4,ncol=len1)
db3<-matrix(nrow=4,ncol=len1)
db4<-matrix(nrow=1,ncol=len1)

dz1<-matrix(nrow=4,ncol=len1)
dz2<-matrix(nrow=4,ncol=len1)
dz3<-matrix(nrow=4,ncol=len1)
dz4<-matrix(nrow=1,ncol=len1)

da1<-matrix(nrow=4,ncol=len1)
da2<-matrix(nrow=4,ncol=len1)
da3<-matrix(nrow=4,ncol=len1)

alpha<-0.5
mse<-0

for(i in 1:len1)
{
for(j in 1:3)
{
a0[j,i]<-inputs[i,j]
}
}

#Random Initialisation of weights

rb1<-rnorm(4)
rdb1<-rnorm(4)
rb2<-rnorm(4)
rdb2<-rnorm(4)
rb3<-rnorm(4)
rdb3<-rnorm(4)
rb4<-rnorm(1)
rdb4<-rnorm(1)

for(i in 1:4)
{
for(j in 1:3)
{
w1[i,j]<-rnorm(1)
dw1[i,j]<-rnorm(1)
}
}
for(i in 1:4)
{
for(j in 1:4)
{
w2[i,j]<-rnorm(1)
dw2[i,j]<-rnorm(1)
}
}
for(i in 1:4)
{
for(j in 1:4)
{
w3[i,j]<-rnorm(1)
dw3[i,j]<-rnorm(1)
}
}
for(i in 1:1)
{
for(j in 1:4)
{
w4[i,j]<-rnorm(1)
dw4[i,j]<-rnorm(1)
}
}

for(i in 1:4)
{
for(j in 1:len1)
{
b1[i,j]<-rb1[i]
db1[i,j]<-rdb1[i]
}
}
for(i in 1:4)
{
for(j in 1:len1)
{
b2[i,j]<-rb2[i]
db2[i,j]<-rdb2[i]
}
}
for(i in 1:4)
{
for(j in 1:len1)
{
b3[i,j]<-rb3[i]
db3[i,j]<-rdb3[i]
}
}
for(i in 1:1)
{
for(j in 1:len1)
{
b4[i,j]<-rb4[i]
db4[i,j]<-rdb4[i]
}
}

#feed forward and back propagation using multivariate regression and gradient descent respectively
act1<-matrix(nrow=4,ncol=len1)
act2<-matrix(nrow=4,ncol=len1)
act3<-matrix(nrow=4,ncol=len1)

for(i in 1:100)
{
z1<-w1%*%a0+b1
a1<-tanh(z1)
z2<-w2%*%a1+b2
a2<-tanh(z2)
z3<-w3%*%a2+b3
a3<-tanh(z3)
z4<-w4%*%a3+b4
mse<-sum((output-z4)**2)/len1
dz4<--2*(output-z4)/len1
dw4<-(dz4%*%t(a3))/len1
db4<-sum(dz4)/len1
da3<-t(w4)%*%dz4
act3<-1/(cosh(z3)*cosh(z3))
dz3<-da3*act3
dw3<-(dz3%*%t(a2))/len1
db3<-sum(dz3)/len1
da2<-t(w3)%*%dz3
act2<-1/(cosh(z2)*cosh(z2))
dz2<-da2*act2
dw2<-(dz2%*%t(a1))/len1
db2<-sum(dz2)/len1
da1<-t(w2)%*%dz2
act1<-1/(cosh(z1)*cosh(z1))
dz1<-da1*act1
dw1<-(dz1%*%t(a0))/len1
db1<-sum(dz1)/len1
w4<-w4-alpha*dw4
w3<-w3-alpha*dw3
w2<-w2-alpha*dw2
w1<-w1-alpha*dw1
b4<-b4-alpha*db4
b3<-b3-alpha*db3
b2<-b2-alpha*db2
b1<-b1-alpha*db1
}
