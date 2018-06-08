#datasets to save: nba.df2 (contains all needed variables and years)

#Load Libraries
library(randomForest)
library(rpart)
library(MASS)
library(QuantPsyc)

# Loading Bagged Inference Function:
# The function below will generate confidence intervals (return lower and upper bounds; based on the full feature set) for all points in the test set, as well as perform a test for significance for the specified features.
New.Bagged.Inf <- function(train=df.demo,test=test.demo,testvars=c(6),verbose=TRUE,k=75,nx1=50,nmc=250,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0) {
  # train -- training set (data frame)
  # test -- test set (data frame)
  # testvars -- index of columns in the data frame corresponding to variables being tested (these are the variables EXCLUDED from the reduced set, so a high p-value indicates that the variables NOT LISTED are sufficient for predicting, and a low p value indicates that one of these variables adds siginficantly to prediction, so more should be added to the reduced set)
  # verbose -- if TRUE, outputs progress
  # k -- size of each subsample
  # nx1 -- variance estimation parameter; number of estimates of conditional expectation
  # nmc -- variance estimation parameter; number of monte carlo samples used to estimate conditional expectation
  # minsplit -- the minimum number of observations needed in a leaf to be eligible for splitting
  # maxcompete -- number of "back-up" splits to keep track of
  # maxsurrogate -- number of surrogate splits to consider
  # usesurrogate -- if nonzero, this helps deal with missing data
  
  # Defining rpart Control Parameters
  control.sim <- rpart.control(minsplit=minsplit,maxcompete=maxcompete,maxsurrogate=maxsurrogate,usesurrogate=usesurrogate)
  
  # Defining the size of the training set and ensemble
  n <- dim(train)[1]
  m <- nx1*nmc
  train.red <- train
  test.red <- test
  
  # Defining the permuted data:
  for (i in 1:length(testvars)) {
    ind1 <- testvars[i]
    train.red[,ind1] <- sample(train.red[,ind1])
  }
  
  # Build the trees and estimate the parameters:
  pred.all <- matrix(0,nrow=1,ncol=dim(test)[1])
  diff.all <- matrix(0,nrow=1,ncol=dim(test)[1])
  cond.exp.full <- matrix(0,nrow=nx1,ncol=dim(test)[1])
  cond.exp.diff <- matrix(0,nrow=nx1,ncol=dim(test)[1])
  for (i in 1:nx1) {
    ind.x1 <- sample(1:dim(train)[1],size=1,replace=FALSE)
    pred.full <- matrix(0,nrow=nmc,ncol=dim(test)[1])
    pred.red <- matrix(0,nrow=nmc,ncol=dim(test)[1])
    pred.diff <- matrix(0,nrow=nmc,ncol=dim(test)[1])
    for (j in 1:nmc) {
      ind <- c(ind.x1,sample((1:dim(train)[1])[-ind.x1],k-1,replace=FALSE))
      ss.full <- train[ind,]	
      ss.red <- train.red[ind,]
      tree.full <- rpart(y~.,data=ss.full,control=control.sim)
      tree.red <- rpart(y~.,data=ss.red,control=control.sim)
      pred.full[j,] <- predict(tree.full,test)
      pred.red[j,] <- predict(tree.red,test.red)
      pred.diff[j,] <- pred.full[j,] - pred.red[j,]
      if (verbose) cat("nx1:  ",i,"          nmc:  ",j,"\n")
    }
    pred.all <- rbind(pred.all,pred.full)
    diff.all <- rbind(diff.all,pred.diff)
    cond.exp.full[i,] <- apply(pred.full,2,mean)
    cond.exp.diff[i,] <- apply(pred.diff,2,mean)
  }
  pred.all <- pred.all[-1,]
  diff.all <- diff.all[-1,]
  
  mean.full <- apply(pred.all,2,mean)
  mean.diff <- apply(diff.all,2,mean)
  
  zeta1.full <- apply(cond.exp.full,2,var)
  zeta1.diff <- cov(cond.exp.diff)
  
  zetak.full <- apply(pred.all,2,var)
  zetak.diff <- cov(diff.all)
  
  sd.full <- sqrt((m/n)*((k^2)/m)*zeta1.full + (1/m)*zetak.full)
  lbounds <- qnorm(0.025,mean=mean.full,sd=sd.full)
  ubounds <- qnorm(0.975,mean=mean.full,sd=sd.full)
  
  tstat <- t(mean.diff) %*% ginv((m/n)*((k^2)/m)*zeta1.diff + (1/m)*zetak.diff) %*% mean.diff
  pval <- 1-pchisq(tstat,df=dim(test)[1])
  # lbounds -- lower bounds for the confidence intervals
  # ubounds -- upper bounds for the confidence intervals
  # pred -- predictions from the ensemble at the test points (center of the confidence intervals)
  # tstat -- test statistic from the test for significance
  # pval -- pvalue from the test for significance
  return(list("lbounds"=lbounds,"ubounds"=ubounds,"pred"=mean.full,"tstat"=tstat,"pval"=pval))
}

#read in dataset  (get rid of metrics not used)
nba.df2 <- read.csv('C:\\Users\\Owner\\Documents\\Spring 2018\\NBA\\Data\\nba.df2.csv')

#################################################
#################  Section 2  ###################
#################################################

lastyear <- nba.df2[nba.df2$Year == 2017,]
#Distributions: improve quality of plots later
hist(lastyear$PER)
hist(lastyear$VORP)
hist(lastyear$WS)

#################################################
################  Section 3  ####################
#################################################
m <- 3000
k <- 1000
mse.rf <- double(10)
mse.lm <- double(10)
mse.rf.ss <- double(10)
r2.rf <- double(10)
r2.lm <- double(10)
r2.rf.ss <- double(10)
control.sim <- rpart.control(minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)

#Using Only Years between 2001-2002 the 2015-2016 Season (Not sure about this choice)
nba.df2_15 <- nba.df2[nba.df2$Year < 2017 & nba.df2$Year > 2001,]
nba.df2_16 <- nba.df2[nba.df2$Year > 2001,]

#################### PER ########################

#Testing which models are best for PER
nba.df2_15_PER <- subset(nba.df2_15, select = c(Year, X3P, X3PA, X2P, X2PA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, TSp, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, MP, PER))

#Create 10 equally size folds
folds <- cut(seq(1,nrow(nba.df2_15_PER)),breaks=10,labels=FALSE)
#Perform 10 fold cross validation
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  df.test <- nba.df2_15_PER[testIndexes, ]
  df.train <- nba.df2_15_PER[-testIndexes, ]
  
  # Build the RF:
  rf <- randomForest(PER~.,data=df.train,importance=FALSE,xtest=df.test[,-24],ytest=df.test[,24], keep.forest=TRUE)
  mse.rf[i] <- mean((as.numeric(rf$test$predicted) - df.test[,24])^2)
  r2.rf[i] <- 1 - (sum((df.test$PER - predict(rf,df.test))^2))/(sum((mean(df.test$PER) - df.test$PER)^2))
  
  #Build the LM
  lm.nba <- lm(PER~.,data=df.train)
  mse.lm[i] <- mean((predict(lm.nba,df.test) - df.test[,24])^2 ) #change indeces of df.test to whatever your response variable is (24)
  r2.lm[i] <- 1 - (sum((df.test$PER - predict(lm.nba,df.test))^2))/(sum((mean(df.test$PER) - df.test$PER)^2)) 
  
  #Build Subsampled RF:
  #Select subsamples; use each to build tree and predict; save prediction results in pred.ss
  pred.bag.ss <- matrix(0,nrow=m,ncol=nrow(df.test))
  pred.rf.ss <- matrix(0,nrow=m,ncol=nrow(df.test))
  for (j in 1:m) {
    ind.ss <- sample(1:dim(df.train)[1],size=k,replace=FALSE)
    dat.ss <- df.train[ind.ss,]
    tree.ss <- rpart(PER~.,data=dat.ss,control=control.sim)
    pred.bag.ss[j,] <- predict(tree.ss,df.test[,-24])
    rf.ss <- randomForest(PER~.,data=dat.ss,ntree=1,replace=FALSE,nodesize=2,nperm=0,proximity=FALSE,oob.prox=FALSE,keep.forest=FALSE,corr.bias=FALSE,keep.inbag=FALSE,xtest=df.test[,-24])
    pred.rf.ss[j,] <- rf.ss$test$predicted
  }
  # save average prediction
  mse.rf.ss[i] <- mean((apply(pred.rf.ss,2,mean) - df.test[,24])^2) 
  r2.rf.ss[i] <- 1 - (sum((apply(pred.rf.ss,2,mean) - df.test[,24])^2))/(sum((mean(df.test$PER) - df.test[,24])^2))
}

## RESULTS  ##
mean(mse.rf)
mean(mse.lm)
mean(mse.rf.ss)
mean(r2.rf)
mean(r2.lm)
mean(r2.rf.ss)


#################### VORP #######################

#Testing which models are best for PER
nba.df2_15_VORP <- subset(nba.df2_15, select = c(Year, X3P, X3PA, X2P, X2PA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, TSp, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, MP, VORP))

#Create 10 equally size folds
folds <- cut(seq(1,nrow(nba.df2_15_VORP)),breaks=10,labels=FALSE)
#Perform 10 fold cross validation
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  df.test <- nba.df2_15_VORP[testIndexes, ]
  df.train <- nba.df2_15_VORP[-testIndexes, ]
  
  # Build the RF:
  rf <- randomForest(VORP~.,data=df.train,importance=FALSE,xtest=df.test[,-24],ytest=df.test[,24], keep.forest=TRUE)
  mse.rf[i] <- mean((as.numeric(rf$test$predicted) - df.test[,24])^2)
  r2.rf[i] <- 1 - (sum((df.test$VORP - predict(rf,df.test))^2))/(sum((mean(df.test$VORP) - df.test$VORP)^2))
  
    
  # Build the LM
  lm.nba <- lm(VORP~.,data=df.train)
  mse.lm[i] <- mean((predict(lm.nba,df.test[,-24]) - df.test[,24])^2 ) #change indeces of df.test to whatever your response variable is (24)
  r2.lm[i] <- 1 - (sum((df.test$VORP - predict(lm.nba,df.test))^2))/(sum((mean(df.test$VORP) - df.test$VORP)^2)) 
  
    
  #Build Subsampled RF:
  # Select subsamples; use each to build tree and predict; save prediction results in pred.ss
  pred.bag.ss <- matrix(0,nrow=m,ncol=nrow(df.test))
  pred.rf.ss <- matrix(0,nrow=m,ncol=nrow(df.test))
  for (j in 1:m) {
    ind.ss <- sample(1:dim(df.train)[1],size=k,replace=FALSE)
    dat.ss <- df.train[ind.ss,]
    tree.ss <- rpart(VORP~.,data=dat.ss,control=control.sim)
    pred.bag.ss[j,] <- predict(tree.ss,df.test[,-24])
    rf.ss <- randomForest(VORP~.,data=dat.ss,ntree=1,replace=FALSE,nodesize=2,nperm=0,proximity=FALSE,oob.prox=FALSE,keep.forest=FALSE,corr.bias=FALSE,keep.inbag=FALSE,xtest=df.test[,-24])
    pred.rf.ss[j,] <- rf.ss$test$predicted
  }
  # save average prediction
  mse.rf.ss[i] <- mean((apply(pred.rf.ss,2,mean) - df.test[,24])^2) 
  r2.rf.ss[i] <- 1 - (sum((apply(pred.rf.ss,2,mean) - df.test[,24])^2))/(sum((mean(df.test$VORP) - df.test[,24])^2))
}

## RESULTS  ##
mean(mse.rf)
mean(mse.lm)
mean(mse.rf.ss)
mean(r2.rf)
mean(r2.lm)
mean(r2.rf.ss)

###################### WS #######################

#Testing which models are best for WS
nba.df2_15_WS <- subset(nba.df2_15, select = c(Year, X3P, X3PA, X2P, X2PA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, TSp, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, MP, WS))

#Create 10 equally size folds
folds <- cut(seq(1,nrow(nba.df2_15_WS)),breaks=10,labels=FALSE)
#Perform 10 fold cross validation
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  df.test <- nba.df2_15_WS[testIndexes, ]
  df.train <- nba.df2_15_WS[-testIndexes, ]
  
  # Build the RF:
  rf <- randomForest(WS~.,data=df.train,importance=FALSE,xtest=df.test[,-24],ytest=df.test[,24], keep.forest=TRUE)
  mse.rf[i] <- mean((as.numeric(rf$test$predicted) - df.test[,24])^2)
  r2.rf[i] <- 1 - (sum((df.test$WS - predict(rf,df.test))^2))/(sum((mean(df.test$WS) - df.test$WS)^2))
  
  # Build the LM
  lm.nba <- lm(WS~.,data=df.train)
  mse.lm[i] <- mean((predict(lm.nba,df.test[,-24]) - df.test[,24])^2 ) #change indeces of df.test to whatever your response variable is (24)
  r2.lm[i] <- 1 - (sum((df.test$WS - predict(lm.nba,df.test))^2))/(sum((mean(df.test$WS) - df.test$WS)^2)) 
  
  
  #Build Subsampled RF:
  # Select subsamples; use each to build tree and predict; save prediction results in pred.ss
  pred.bag.ss <- matrix(0,nrow=m,ncol=nrow(df.test))
  pred.rf.ss <- matrix(0,nrow=m,ncol=nrow(df.test))
  for (j in 1:m) {
    ind.ss <- sample(1:dim(df.train)[1],size=k,replace=FALSE)
    dat.ss <- df.train[ind.ss,]
    tree.ss <- rpart(WS~.,data=dat.ss,control=control.sim)
    pred.bag.ss[j,] <- predict(tree.ss,df.test[,-24])
    rf.ss <- randomForest(WS~.,data=dat.ss,ntree=1,replace=FALSE,nodesize=2,nperm=0,proximity=FALSE,oob.prox=FALSE,keep.forest=FALSE,corr.bias=FALSE,keep.inbag=FALSE,xtest=df.test[,-24])
    pred.rf.ss[j,] <- rf.ss$test$predicted
  }
  # save average prediction
  mse.rf.ss[i] <- mean((apply(pred.rf.ss,2,mean) - df.test[,24])^2) 
  r2.rf.ss[i] <- 1 - (sum((apply(pred.rf.ss,2,mean) - df.test[,24])^2))/(sum((mean(df.test$WS) - df.test[,24])^2))
}

## RESULTS  ##
mean(mse.rf)
mean(mse.lm)
mean(mse.rf.ss)
mean(r2.rf)
mean(r2.lm)
mean(r2.rf.ss)


############## Variable Importance ############ (Possible use tables instead of plots to save space)

#Absolute Value of Standardized Coefficients plot: PER
lm.nba <- lm(PER~., data=nba.df2_15_PER)
coef_lmbeta <- sort(abs(lm.beta(lm.nba)))
plot(coef_lmbeta, xaxt="n", ylab = '', xlab = '', main='Absolute Value of Standardized Coefficients: PER')
axis(1,at=1:23,labels=FALSE)
text(1:23,-0.10, srt = 45, adj = 1, labels = names(coef_lmbeta),xpd = TRUE)

#Get OoB importance measure for the features and plot: VORP
nsim <- 25
imp.mat <- matrix(0,nrow=nsim,ncol=dim(nba.df2_15_VORP)[2]-1)
for (i in 1:nsim) {
  # Build the random Forest
  rf <- randomForest(VORP~.,data=nba.df2_15_VORP,ntree=500,importance=TRUE)
  imp.mat[i,] <- rf$importance[,1]
  # Verbose:
  cat("Random Forest ",i," of ",nsim,"\n")
}
#Make box plots of feature importance:
rank.feat <- matrix(0,nrow=nsim,ncol=dim(nba.df2_15_VORP)[2]-1)
for (i in 1:nsim) {
  rnk <- order(-imp.mat[i,])
  for (j in 1:dim(nba.df2_15v)[2]-1) {
    rank.feat[i,j] <- which(rnk==j)
  }
}
nba.names <- rownames(rf$importance)
par(mar = c(6, 4, 4, 2) + 0.1)
boxplot(imp.mat[,order(apply(-imp.mat,2,mean))],names=nba.names[order(apply(-imp.mat,2,mean))],xaxt="n",main="Importance of Individual Statistics: VORP",ylab="")
axis(1,at=1:22,labels=FALSE)
text(1:22,-0.15, srt = 45, adj = 1, labels = nba.names[order(apply(-imp.mat,2,mean))],xpd = TRUE)

#Get OoB importance measure for the features and plot: WS
nsim <- 25
imp.mat <- matrix(0,nrow=nsim,ncol=dim(nba.df2_15_WS)[2]-1)
for (i in 1:nsim) {
  # Build the random Forest
  rf <- randomForest(WS~.,data=nba.df2_15_WS,ntree=500,importance=TRUE)
  imp.mat[i,] <- rf$importance[,1]
  # Verbose:
  cat("Random Forest ",i," of ",nsim,"\n")
}
#Make box plots of feature importance:
rank.feat <- matrix(0,nrow=nsim,ncol=dim(nba.df2_15_WS)[2]-1)
for (i in 1:nsim) {
  rnk <- order(-imp.mat[i,])
  for (j in 1:dim(nba.df2_15v)[2]-1) {
    rank.feat[i,j] <- which(rnk==j)
  }
}
nba.names <- rownames(rf$importance)
par(mar = c(6, 4, 4, 2) + 0.1)
boxplot(imp.mat[,order(apply(-imp.mat,2,mean))],names=nba.names[order(apply(-imp.mat,2,mean))],xaxt="n",main="Importance of Individual Statistics: WS",ylab="")
axis(1,at=1:22,labels=FALSE)
text(1:22,-0.15, srt = 45, adj = 1, labels = nba.names[order(apply(-imp.mat,2,mean))],xpd = TRUE)

###############################################
################  Section 4  ##################
###############################################

################ Hypothesis Tests #################

#PER full vs reduced model
lm.1 <- lm(PER~.,data=nba.df2_15_PER)
lm.2 <- lm(PER~.-MP,data=nba.df2_15_PER)
anova(lm.1,lm.2,test="Chisq")

#VORP Full vs reduced
#First define train and test sets:
test.ind <- sample(1:dim(nba.df2_15_VORP)[1],30,replace=FALSE)
nba.train <- nba.df2_15_VORP[-test.ind,]
nba.test <- nba.df2_15_VORP[test.ind,]
nba.train <- nba.train[,c(24,seq(1,23,1))]
nba.test <- nba.test[,c(24,seq(1,23,1))]
names(nba.train)[1] <- "y"
names(nba.test)[1] <- "y"

test.VORP <- New.Bagged.Inf(train=nba.train,test=nba.test,testvars=which(names(nba.train) %in% c("ASTp", "X2P", "X2PA", "X3P", "X3PA", "FT", "USGp", "TOV", "FTA", "ORB", "AST", "BLKp", "PF", "STLp", "DRBp", "ORBp", "Year", "STL", "BLK", "DRB", "TOVp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)

#WS Full vs reduced
#First define train and test sets:
test.ind <- sample(1:dim(nba.df2_15_WS)[1],30,replace=FALSE)
nba.train <- nba.df2_15_WS[-test.ind,]
nba.test <- nba.df2_15_WS[test.ind,]
nba.train <- nba.train[,c(24,seq(1,23,1))]
nba.test <- nba.test[,c(24,seq(1,23,1))]
names(nba.train)[1] <- "y"
names(nba.test)[1] <- "y"

test.WS <- New.Bagged.Inf(train=nba.train,test=nba.test,testvars=which(names(nba.train) %in% c("ASTp" ,"X2P", "X2PA", "X3P", "X3PA", "FT", "USGp", "TOV", "FTA", "ORB", "AST", "BLKp", "PF", "STLp", "DRBp", "ORBp", "Year", "STL", "BLK", "DRB", "TOVp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)

####################Predicting 16/17############################
#Predicting 2016-2017 VORP 
nba.df2_16_VORP <- subset(nba.df2_16, select = c(Year, X3P, X3PA, X2P, X2PA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, TSp, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, MP, VORP))

#VORP Full vs reduced
#First define train and test sets:
nba.train.17 <- nba.df2_16_VORP[nba.df2_16_VORP$Year<=2016,]
nba.test.17 <- nba.df2_16_VORP[nba.df2_16_VORP$Year==2017,]
nba.train.17 <- nba.train.17[,c(24,seq(1,23,1))]
nba.test.17 <- nba.test.17[,c(24,seq(1,23,1))]
names(nba.train.17)[1] <- "y"
names(nba.test.17)[1] <- "y"

pred.VORP.17 <- New.Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("ASTp", "X2P", "X2PA", "X3P", "X3PA", "FT", "USGp", "TOV", "FTA", "ORB", "AST", "BLKp", "PF", "STLp", "DRBp", "ORBp", "Year", "STL", "BLK", "DRB", "TOVp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)

#Save information from bagged inf in separate objects (can change this code based on 1 of 6 objects above^)
preds.order <- pred.VORP.17$pred[order(pred.VORP.17$pred)]
lbounds.order <- pred.VORP.17$lbounds[order(pred.VORP.17$pred)]
ubounds.order <- pred.VORP.17$ubounds[order(pred.VORP.17$pred)]
true.preds <- nba.test.17$y[order(pred.VORP.17$pred)]
true.names <- nba.df2$Name[nba.df2$Year==2017][order(pred.VORP.17$pred)] 

###### Best Thirty Players ######
#Only change this code for window size and axis labels
par(mar = c(8, 4, 4, 2) + 0.1)
plot(preds.order[457:486],type="b",ylim=c(-1,15),pch=20,xaxt="n",ylab="VORP",xlab="",main="2016-2017 Predicted VORP")
lines(preds.order[457:486],col='black',type='l',lwd=2)
lines(lbounds.order[457:486],col='blue',type='l',lwd=2)
lines(ubounds.order[457:486],col='blue',type='l',lwd=2)
points(true.preds[457:486],col='red',pch=15,cex=1.05)

#Plotting true Metrics
pred.wild <- which((true.preds[457:486]<lbounds.order[457:486]) | (true.preds[457:486]>ubounds.order[457:486]))
points(pred.wild,true.preds[457:486][pred.wild],pch=8,cex=1.5,col='red')
col.red <- rep('black',30)
col.red[pred.wild] <- 'red'

#Adding names
axis(1,at=1:30,labels=FALSE)
text(1:30, -2.5, srt = 45, adj = 1, labels = true.names[457:486], col=col.red,xpd = TRUE)
box()

###### random thirty players ######
pred.17.order <- data.frame(preds.order,lbounds.order,ubounds.order,true.preds,true.names)
rs.preds <- pred.17.order[sample(nrow(pred.17.order),486),]
rs.preds <- pred.17.order[sample(nrow(pred.17.order),30),]

par(mar = c(8, 4, 4, 2) + 0.1)
plot(rs.preds$preds.order,type="b",ylim=c(-3,10),pch=20,xaxt="n",ylab="VORP",xlab="",main="2016-2017 Predicted VORP")
lines(rs.preds$preds.order,col='black',type='l',lwd=2)
lines(rs.preds$lbounds.order,col='blue',type='l',lwd=2)
lines(rs.preds$ubounds.order,col='blue',type='l',lwd=2)
points(rs.preds$true.preds,col='red',pch=15,cex=1.05)

axis(1,at=1:30,labels=FALSE)
text(1:30, -4.2, srt = 45, adj = 1, labels = rs.preds$true.names, col=col.red,xpd = TRUE)
box()


##Predicting 2016-2017 WS 
nba.df2_16_WS <- subset(nba.df2_16, select = c(Year, X3P, X3PA, X2P, X2PA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, TSp, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, MP, WS))

#First define train and test sets:
nba.train.17 <- nba.df2_16_WS[nba.df2_16_WS$Year<=2016,]
nba.test.17 <- nba.df2_16_WS[nba.df2_16_WS$Year==2017,]
nba.train.17 <- nba.train.17[,c(24,seq(1,23,1))]
nba.test.17 <- nba.test.17[,c(24,seq(1,23,1))]
names(nba.train.17)[1] <- "y"
names(nba.test.17)[1] <- "y"

pred.WS.17 <- New.Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("ASTp", "X2P", "X2PA", "X3P", "X3PA", "FT", "USGp", "TOV", "FTA", "ORB", "AST", "BLKp", "PF", "STLp", "DRBp", "ORBp", "Year", "STL", "BLK", "DRB", "TOVp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)

#Save information from bagged inf in separate objects (can change this code based on 1 of 6 objects above^)
preds.order <- pred.WS.17$pred[order(pred.WS.17$pred)]
lbounds.order <- pred.WS.17$lbounds[order(pred.WS.17$pred)]
ubounds.order <- pred.WS.17$ubounds[order(pred.WS.17$pred)]
true.preds <- nba.test.17$y[order(pred.WS.17$pred)]
true.names <- nba.df2$Name[nba.df2$Year==2017][order(pred.WS.17$pred)] 

###### Best Thirty Players ######
#Only change this code for window size and axis labels
par(mar = c(8, 4, 4, 2) + 0.1)
plot(preds.order[457:486],type="b",ylim=c(-1,15),pch=20,xaxt="n",ylab="WS",xlab="",main="2016-2017 Predicted VORP")
lines(preds.order[457:486],col='black',type='l',lwd=2)
lines(lbounds.order[457:486],col='blue',type='l',lwd=2)
lines(ubounds.order[457:486],col='blue',type='l',lwd=2)
points(true.preds[457:486],col='red',pch=15,cex=1.05)

#Plotting true Metrics
pred.wild <- which((true.preds[457:486]<lbounds.order[457:486]) | (true.preds[457:486]>ubounds.order[457:486]))
points(pred.wild,true.preds[457:486][pred.wild],pch=8,cex=1.5,col='red')
col.red <- rep('black',30)
col.red[pred.wild] <- 'red'

#Adding names
axis(1,at=1:30,labels=FALSE)
text(1:30, -2.5, srt = 45, adj = 1, labels = true.names[457:486], col=col.red,xpd = TRUE)
box()

###### random thirty players ######
pred.17.order <- data.frame(preds.order,lbounds.order,ubounds.order,true.preds,true.names)
rs.preds <- pred.17.order[sample(nrow(pred.17.order),486),]
rs.preds <- pred.17.order[sample(nrow(pred.17.order),30),]

par(mar = c(8, 4, 4, 2) + 0.1)
plot(rs.preds$preds.order,type="b",ylim=c(-3,10),pch=20,xaxt="n",ylab="WS",xlab="",main="2016-2017 Predicted VORP")
lines(rs.preds$preds.order,col='black',type='l',lwd=2)
lines(rs.preds$lbounds.order,col='blue',type='l',lwd=2)
lines(rs.preds$ubounds.order,col='blue',type='l',lwd=2)
points(rs.preds$true.preds,col='red',pch=15,cex=1.05)

axis(1,at=1:30,labels=FALSE)
text(1:30, -4.2, srt = 45, adj = 1, labels = rs.preds$true.names, col=col.red,xpd = TRUE)
box()

###############################################
################  Section 5  ##################
###############################################

#Discrimination
library(boot)
BVar <- function(data, indices){
  resample <- data[indices, ]
  PTS <- mean(resample$PTS)
  FGA <- mean(resample$FGA)
  FTA <- mean(resample$FTA)
  TSA <- FGA + (0.44*FTA)
  TSp <- PTS/(2*TSA)
  MP <- mean(resample$MIN)
  metric<- MP*TSp
  return(metric)
}
boxscore <- read.csv("C:/Users/Owner/Documents/Spring 2018/NBA/Data/NBA-2016-2017-Player-BoxScore-Dataset.csv")
players <- unique(boxscore$PLAYER.FULL.NAME)
players <- players[ players != 'Danuel House' ]
BVar.vec <- double(length(players))
player.season.avg <- double(length(players))
counter <- 1
for(player in players){
  playersubset <- boxscore[which(boxscore$PLAYER.FULL.NAME==player), ]
  boot.d <- boot(playersubset, statistic = BVar, R = 1000)
  BVar.vec[counter] <- var(boot.d$t)
  player.season.avg[counter] <- boot.d$t0 
  counter<-counter+1
}
BVar.vec <- BVar.vec[!is.na(BVar.vec)]
numerator <- mean(BVar.vec)
season.avg <- mean(player.season.avg)
player.var <- sapply(player.season.avg, function(x) (x-season.avg)^2)
denominator <- mean(player.var)
discrimination <- 1 - (numerator/denominator)
print(discrimination)









############################################
#Hypothesis Test
test.ind <- sample(1:dim(nba.df2_15v)[1],30,replace=FALSE)
nba.train <- nba.df2_15v[-test.ind,]
nba.test <- nba.df2_15v[test.ind,]
nba.train <- nba.train[,c(22,seq(1,21,1),23)]
nba.test <- nba.test[,c(22,seq(1,21,1),23)]
names(nba.train)[1] <- "y"
names(nba.test)[1] <- "y"

#All variables besides our metric insignificant: 
test <- New.Bagged.Inf(train=nba.train,test=nba.test,testvars=which(names(nba.train) %in% c("Year", "X3P", "X3PA", "X2P", "X2PA", "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF", "ORBp", "DRBp", "ASTp", "STLp", "BLKp", "TOVp", "USGp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)
print(test)

#Our metric significant by itself:
test1 <- New.Bagged.Inf(train=nba.train,test=nba.test,testvars=which(names(nba.train) %in% c("Metric")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)
print(test1)

############### Marginal Influence ################

#Defining the Test Set:
test.mp <- data.frame(matrix(as.numeric(apply(nba.df2_15_VORP,2,mean)),byrow=T,ncol=dim(nba.df2_15_VORP)[2],nrow=15))
names(test.mp) <- names(nba.df2_15_VORP)
test.mp$MP <- seq(1,15,1)
test.mp$Season <- rep(2012,15)

nba.df.hyptest <- nba.df2_15_VORP[,c(24,seq(1,23,1))]
test.mp <- test.mp[,c(24,seq(1,23,1))]
names(nba.df.hyptest)[1] <- "y"
names(test.mp)[1] <- "y"

mp.preds <- New.Bagged.Inf(train=nba.df.hyptest,test=test.mp,testvars=c(5),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)

plot(seq(1,15,1),mp.preds$pred,ylim=c(-2.55,0),pch=20,xlab="Minutes Played",ylab="VORP",main="Marginal Effect of Minutes Played")
#points(seq(2000,2014,1),as.numeric(aggregate(nba.df$RAPM,by=list(year=nba.df$Season),mean)$x),pch=18,col='red')
lines(1:15,mp.preds$lbounds,type="b",col='blue',pch=20,lwd=2)
lines(1:15,mp.preds$ubounds,type="b",col='blue',pch=20,lwd=2)



####################################################
############### Testing New Metric ##################
#####################################################
#Selecting variables for use in the models (Change if predicting VORP vs. WS) 
#nba.df2_15v <- subset(nba.df2, select = c(Year, X3P, X3PA, X2P, X2PA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, VORP, Metic))



#Creating New Metric
#nba.df2$Metric <- nba.df2$TSp*nba.df2$MP
#could show few other versions to compare
#Correlation between new metric and VORP
#LM <- lm(VORP~Metric, data = nba.df2_15v)
#summary(LM)
#plot((nba.df2_15v$Metric)^2,nba.df2_15v$VORP)




