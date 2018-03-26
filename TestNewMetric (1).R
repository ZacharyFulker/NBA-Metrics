#####################################################
############### Testing New Metric ##################
#####################################################

#datasets to save: nba.df, nba.df1 - delete X1 if R automatically inserts the variable upon import

#Load Libraries
library(randomForest)
library(rpart)
library(MASS)

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

#Read in CSV
nba.df1 <- read.csv(file = "C:\\Users\\Owner\\Documents\\Spring 2018\\NBA\\Data\\nba.df1.csv")

#Creating New Metric
nba.df1$Metric <- nba.df1$TSp*nba.df1$MP
nba.df1$Metric1 <- (nba.df1$Metric)^2
nba.df1$metric2 <- nba.df1$TSp*nba.df1$MP*nba.df1$ASTp 
nba.df1$metric3 <- nba.df1$TSp*nba.df1$MP+10*nba.df1$ASTp
nba.df1$Metric4 <- (nba.df1$Metric)^3
nba.df1$Metric5 <- nba.df1$TSp*nba.df1$MP*(nba.df1$X3PA + nba.df1$X2PA) 
nba.df1$Metric6 <- (nba.df1$Metric5)^2 
nba.df1$FGA <- (nba.df1$X3PA + nba.df1$X2PA)

#Using Only Years before the 2016-2017 Season
nba.df1_15 <- nba.df1[nba.df1$Year < 2017,]

#Selecting variables for use in the models (Change if predicting VORP vs. WS) 
nba.df1_15v <- subset(nba.df1, select = c(Year, FGA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, VORP))

#Correlation between new metric and VORP
LM <- lm(VORP~Metric2, data = nba.df1_15v)
summary(LM)
plot(nba.df1_15v$metric2,nba.df1_15v$VORP)

#############################################
#Get OoB importance measure for the features and plot:
nsim <- 1
imp.mat <- matrix(0,nrow=nsim,ncol=dim(nba.df1_15v)[2]-1)
for (i in 1:nsim) {
  
  # Build the random Forest
  rf <- randomForest(VORP~.,data=nba.df1_15v,ntree=500,importance=TRUE)
  imp.mat[i,] <- rf$importance[,1]
  
  # Verbose:
  cat("Random Forest ",i," of ",nsim,"\n")
  
}

#Make box plots of feature importance:
rank.feat <- matrix(0,nrow=nsim,ncol=dim(nba.df1_15v)[2]-1)
for (i in 1:nsim) {
  rnk <- order(-imp.mat[i,])
  for (j in 1:dim(nba.df1_15v)[2]-1) {
    rank.feat[i,j] <- which(rnk==j)
  }
}

nba.names <- rownames(rf$importance)

par(mar = c(6, 4, 4, 2) + 0.1)
boxplot(imp.mat[,order(apply(-imp.mat,2,mean))],names=nba.names[order(apply(-imp.mat,2,mean))],xaxt="n",main="Importance of Individual Statistics: VORP",ylab="")
axis(1,at=1:18,labels=FALSE)
text(1:18,-0.15, srt = 45, adj = 1, labels = nba.names[order(apply(-imp.mat,2,mean))],xpd = TRUE)

############################################
#Hypothesis Test
test.ind <- sample(1:dim(nba.df1_15v)[1],30,replace=FALSE)
nba.train <- nba.df1_15v[-test.ind,]
nba.test <- nba.df1_15v[test.ind,]
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

############################################
#Testing which models are best (change variables based on preference)
nba.df1_15v1 <- subset(nba.df1_15, select = c(Year, X3P, X3PA, X2P, X2PA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF, TSp, ORBp, DRBp, ASTp, STLp, BLKp, TOVp, USGp, MP, VORP))

#Use 10-fold validation for standard RF and LM:
#Create 10 equally size folds
folds <- cut(seq(1,nrow(nba.df1_15v1)),breaks=10,labels=FALSE)
#Perform 10 fold cross validation
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  df.test <- nba.df1_15v1[testIndexes, ]
  df.train <- nba.df1_15v1[-testIndexes, ]
  
  # Build the RF:
  rf <- randomForest(VORP~.,data=df.train,importance=FALSE,xtest=df.test[,-24],ytest=df.test[,24]) #change indeces of df.test to whatever your response variable is (24)
  mse.rf[i] <- mean( (as.numeric(rf$test$predicted) - df.test[,24])^2 ) #change indeces of df.test to whatever your response variable is (24)
  
  # Build the LM
  lm.nba <- lm(VORP~.,data=df.train)
  mse.lm[i] <- mean( (predict(lm.nba,df.test[,-24]) - df.test[,24])^2 ) #change indeces of df.test to whatever your response variable is (24)
  
  # Build Aug LM, No AIC for now
  nba.lm <- lm(VORP~.,data=nba.df1_15v1)
  lm.preds <- predict(nba.lm, nba.df1_15v1)
  lm.diff <- nba.df1_15v1$VORP - lm.preds
  nba.df.diff <- nba.df1_15v1
  nba.df.diff$VORP <- lm.diff
  df.test.aug <- nba.df.diff[testIndexes, ]
  df.train.aug <- nba.df.diff[-testIndexes, ]
  lm.preds.temp <- lm.preds[testIndexes]
  nba.true <- nba.df1_15v1[testIndexes,]
  # Build the RF:
  rf <- randomForest(VORP~.,data=df.train.aug,importance=FALSE,xtest=df.test[,-24],ytest=df.test[,24])
  mse.aug[i] <- mean( ((as.numeric(rf$test$predicted)+lm.preds.temp) - nba.true[,24])^2 ) 
  
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
  #mse.bag.ss[i] <- mean( (apply(pred.bag.ss,2,mean) - df.test[,15])^2 )  
  mse.rf.ss[i] <- mean( (apply(pred.rf.ss,2,mean) - df.test[,24])^2 )
}

plot(mse.rf.ss,type="l",ylim=c(0,1),lwd=2,main="Subsampled Random Forests vs. Linear Model vs. Augmented Model vs. Random Forest: MSE Predicting VORP",xlab="Simulation Index",ylab="MSE")
lines(mse.lm,col="blue",lwd=2)
lines(mse.rf,col="green",lwd=2)
lines(mse.aug,col="red",lwd=2)
legend("topright",c("Random Forest","Linear Model", "Subsampled Random Forest", "Augmented Model"),lty=c(1,1),col=c("Green","blue", "Black", "red"),lwd=c(2,2))


############################################
#Predicting 2016-2017 Win Shares / VORP 

#Training and Test sets split based on year
nba.train.17 <- nba.df1[nba.df1$Year<=2016,]
nba.test.17 <- nba.df1[nba.df1$Year==2017,]

#Only the Variables we want included in the model are retained and response variable name changed to y
nba.train.17 <- nba.train.17[,c(34,1,36)] #All traditional and advanced box score stats: 1,2:14,16,19:20,22:26,35
nba.test.17 <- nba.test.17[,c(34,1,36)] #All traditional and advanced box score stats: 1,2:14,16,19:20,22:26,35
names(nba.train.17)[1] <- "y"
names(nba.test.17)[1] <- "y"

#Remove year variable
nba.train.17 <- nba.train.17[,-which(names(nba.train.17)=="Year")]
nba.test.17 <- nba.test.17[,-which(names(nba.test.17)=="Year")]

#Create Objects that hold bagged inf and new bagged inf information for both WS and VORP (.v). newmet suggests use of only the new metric. newinf suggests use of the NEW bagged inference function. YOU CHOOSE VARIABLES IN THE PREVIOUS STEP... THIS IS SAVING RESULTS IN DIFFERENT OBJECTS.  YOU DON'T NEED TO RUN ALL OF THESE.
pred.17 <- Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("X3P", "X3PA", "X2P", "X2PA", "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF", "TSp", "ORBp", "DRBp", "ASTp", "STLp", "BLKp", "TOVp", "USGp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)
pred.17.v <- Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("X3P", "X3PA", "X2P", "X2PA", "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF", "TSp", "ORBp", "DRBp", "ASTp", "STLp", "BLKp", "TOVp", "USGp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)
pred.17.newinf <- New.Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("X3P", "X3PA", "X2P", "X2PA", "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF", "TSp", "ORBp", "DRBp", "ASTp", "STLp", "BLKp", "TOVp", "USGp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)
pred.17.newinf.v <- New.Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("X3P", "X3PA", "X2P", "X2PA", "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF", "TSp", "ORBp", "DRBp", "ASTp", "STLp", "BLKp", "TOVp", "USGp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)
pred.17.newinf.v.all <- New.Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("X3P", "X3PA", "X2P", "X2PA", "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF", "TSp", "ORBp", "DRBp", "ASTp", "STLp", "BLKp", "TOVp", "USGp")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)
pred.17.newinf.v.newmet <- New.Bagged.Inf(train=nba.train.17,test=nba.test.17,testvars=which(names(nba.train.17) %in% c("Metric")),verbose=TRUE,k=750,nx1=50,nmc=1000,minsplit=3,maxcompete=0,maxsurrogate=0,usesurrogate=0)

#Save information from bagged inf in separate objects (can change this code based on 1 of 6 objects above^)
#true.names was created from a different dataset that contained player names (nba.df).  This dataset does not contain MP, but you could merge them and change this code.
preds.order <- pred.17.newinf.v.newmet$pred[order(pred.17.newinf.v.newmet$pred)]
lbounds.order <- pred.17.newinf.v.newmet$lbounds[order(pred.17.newinf.v.newmet$pred)]
ubounds.order <- pred.17.newinf.v.newmet$ubounds[order(pred.17.newinf.v.newmet$pred)]
true.preds <- nba.test.17$y[order(pred.17.newinf.v.newmet$pred)]
true.names <- nba.df$Name[nba.df$Year==2017][order(pred.17.newinf.v.newmet$pred)] #nba.df needs to be saved because it has player names and nba.df1 doesn't


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