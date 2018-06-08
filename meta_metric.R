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

#Stability
boxscore16 <- read.csv("C:/Users/Owner/Documents/Spring 2018/NBA/Data/NBA-2015-2016-Player-BoxScore-Dataset.csv")
boxscore15 <- read.csv("C:/Users/Owner/Documents/Spring 2018/NBA/Data/NBA-2014-2015-Player-BoxScore-Dataset.csv")
boxscore14 <- read.csv("C:/Users/Owner/Documents/Spring 2018/NBA/Data/NBA-2013-2014-Player-BoxScore-Dataset.csv")
boxscore13 <- read.csv("C:/Users/Owner/Documents/Spring 2018/NBA/Data/NBA-2012-2013-Player-BoxScore-Dataset.csv")
season.totals <- read.csv("C:/Users/Owner/Documents/Spring 2018/NBA/Data/00-17 Advanced Metrics.csv")
all.boxscore <- c(boxscore13, boxscore14, boxscore15, boxscore16, boxscore)

season.totals <- season.totals[which(season.totals$.Season >=2013),]
season.totals$metric<-(season.totals$MP/season.totals$G)*season.totals$TS.
total.metric.mean<-mean(season.totals$metric)
fun <- function(x,y){x - y}
seasons <- unique(all.boxscore$.DATA.SET)##Imm heere
BVar.vec.s <- double(length(seasons))
player.season.metric <- double(length(seasons))
numerator.1 <- double(length(players)) 
denominator.1 <- double(length(players)) 
counter<-1
player.counter<-1
for(player in players){
  seasons <- unique(all.boxscore$.DATA.SET)
  for(season in seasons){
    player.season.subset <- boxscore[which(all.boxscore$PLAYER.FULL.NAME==player & all.boxscore$.DATA.SET==season), ]
    boot.s <- boot(player.season.subset, statistic = BVar, R = 1000)
    BVar.vec.s[counter] <- var(boot.s$t)
    player.season.metric[counter] <- boot.s$t0 
    counter<-counter+1
  }
  metric.avg <- mean(player.season.metric)
  season.var <- sapply(player.season.metric, function(x) (x-metric.avg)^2)
  temp.num <- sapply(season.var, function(x) mapply(fun,x,BVar.vec.s))
  numerator.1[player.counter] <- mean(temp.num)
  season.total.var <- sapply(player.season.metric, function(x) (x-total.metric.mean)^2)
  temp.den <- sapply(season.total.var, function(x) mapply(fun,x,BVar.vec.s))
  denominator.1[player.counter] <- mean(temp.den)
  player.counter<-player.counter+1
}
numerator.2<-mean(numerator.1)
denominator.2<-mean(denominator.1)
stability <- 1-(numerator.2/denominator.2)
print(stability)
