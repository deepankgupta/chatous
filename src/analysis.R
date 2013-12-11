Data=c()
for (i in 201:1000){
	D=read.table("second_pass_quality.txt",sep=";",stringsAsFactors =F,skip=i*1000,nrows=1000)
	Discon=(D$V8==D$V3)+1
	D=cbind(D[,c("V1","V4","V5","V11","V12")],Discon)
	D=D[(D$V11>0 & D$V12>0),]
	Data=rbind(Data,D)
}
head(Data)
write.table(Data,"temp.txt")
Data=c()
for (i in 321:330){
	D=read.table("pp_dataset.csv",sep=";",stringsAsFactors =F,skip=i*1000,nrows=1000,fill=T)
	RegDate=as.numeric(as.Date(substr(D$V6,1,10)))-as.numeric(as.Date("2013-11-01"))
	intro=nchar(D$V7)
	D=cbind(D[,c("V1","V4","V5")],RegDate,intro)
	Data=rbind(Data,D)
}
head(Data)
write.table(Data,"prof.txt")
============================

D=read.table("temp.txt",header=T,stringsAsFactors =F)
names(D)=c("chatid","id1","id2","line1","line2","Discon")
head(D)
D1=read.table("prof.txt",header=T,stringsAsFactors =F)
names(D1)=c("id","age","gender","regdate","intro")
head(D1)
Df=merge(D,D1,by.x="id1",by.y="id")
Df=merge(Df,D1,by.x="id2",by.y="id")
write.table(Df,"fdata.txt")
==============================

rm(list=ls(all=TRUE))
Df=read.table("fdata.txt",header=T,stringsAsFactors =F)
Df$age.x=as.numeric(Df$age.x)
Df$age.y=as.numeric(Df$age.y)
index=(Df$age.x>0 & Df$age.y>0)
Df=Df[index,]

attach(Df)

total.line=line1+line2
log.total.line=log(total.line)
age.x=as.numeric(age.x)
age.y=as.numeric(age.y)
log.age.diff=log(abs(age.y-age.x)+1)
hist(log(total.line),main="Histogram: log of number of total lines")

par(mfrow=c(1,2))
plot(log(age.x),log.total.line,xlim=c(2,5),main="log.total.line by log.age",
xlab="log.age")
plot(log.age.diff,log.total.line,xlab="log: difference of age between chatters",
main="log.total.line by log: \n difference of age between chatters")

par(mfrow=c(1,1))
plot(log.age.diff,log.total.line,xlab="log: difference of age between chatters",
main="log.total.line by log: \n difference of age between chatters")
log.age.diff2=(log.age.diff)^2
model=lm(log.total.line~(log.age.diff))
abline(model)
points(predict(model),type="l")

rm(list=ls(all=TRUE))
Df=read.table("fdata.txt",header=T,stringsAsFactors =F)
Df$age.x=as.numeric(Df$age.x)
Df$age.y=as.numeric(Df$age.y)
index=(Df$age.x>0 & Df$age.y>0)
index[is.na(index)]=FALSE
Df=Df[index,]
unique(Df$age.x)

attach(Df)

total.line=line1+line2
log.total.line=log(total.line)
log.line1=log(line1)
log.line2=log(line2)
age.x=as.numeric(age.x)
age.y=as.numeric(age.y)
gender.x=as.factor(gender.x)
gender.y=as.factor(gender.y)
log.age.diff=log(abs(age.y-age.x)+1)
hist(log(total.line),main="Histogram: log of number of total lines")

par(mfrow=c(1,2))
plot(log(age.x),log.total.line,xlim=c(2,5),main="log.total.line by log.age",
xlab="log.age")
plot(log.age.diff,log.total.line,xlab="log: difference of age between chatters",
main="log.total.line by log: \n difference of age between chatters")

par(mfrow=c(1,1))
plot(log.age.diff,log.total.line,xlab="log: difference of age between chatters",
main="log.total.line by log: \n difference of age between chatters",ylim=c(-2,10))
log.age.diff2=(log.age.diff)^2

model=lm(log.total.line~(log.age.diff)+log.age.diff2)
curve(model[[1]][1]+model[[1]][2]*x+model[[1]][3]*x^2,add=T,col="red",lty=2)

library(truncreg)
model2=truncreg(model,  point = 1, direction = "left")
curve(model2[[1]][1]+model2[[1]][2]*x+model2[[1]][3]*x^2,add=T,col="green",lty=1)

legend("bottomright", c("OLS", "Truncated"), col = c("red","green"),
       lty = c(2, 1))

model01=lm(log.line1~(log.age.diff)+log.age.diff2+gender.x+gender.y+regdate.x+regdate.y+intro.x+intro.y)
model02=lm(log.line2~(log.age.diff)+log.age.diff2+gender.x+gender.y+regdate.x+regdate.y+intro.x+intro.y)
score1=((fitted(model01)-fitted(model02))>0)*((log.line1-log.line2)>0)
score2=((fitted(model01)-fitted(model02))<0)*((log.line1-log.line2)<0)
output1=(length(which(score1==1))+length(which(score2==1)))/length(score1)

model001=truncreg(model01,  point = 0, direction = "left")
model002=truncreg(model02,  point = 0, direction = "left")
score1=((fitted(model001)-fitted(model002))>0)*((log.line1-log.line2)>0)
score2=((fitted(model001)-fitted(model002))<0)*((log.line1-log.line2)<0)
output2=(length(which(score1==1))+length(which(score2==1)))/length(score1)

line_dum=(log.line1-log.line2)>0
model01=glm(line_dum~(log.age.diff)+log.age.diff2+gender.x+gender.y+regdate.x+regdate.y+intro.x+intro.y, family = "binomial")
score1=(fitted(model01)>0.5)*((log.line1-log.line2)>0)
score2=(fitted(model01)<0.5)*((log.line1-log.line2)<0)
output3=(length(which(score1==1))+length(which(score2==1)))/length(score1)


score1=((log.line1-log.line2)>0)
score2=((log.line1-log.line2)<0)
output4=length(which(score1==1))/(length(which(score1==1))+length(which(score2==1)))

print(c(output1,output2,output3,output4))

========================================
model01=lm(Discon~(log.age.diff)+log.age.diff2+gender.x+gender.y+regdate.x+regdate.y+intro.x+intro.y)
score1=((fitted(model01))<1.5)*(Discon==1)
score2=((fitted(model01))>1.5)*(Discon==2)
output1=(length(which(score1==1))+length(which(score2==1)))/length(score1)

output1
=========================================
model01=lm(log.line1~1)
model02=lm(log.line2~1)
score1=((fitted(model01)-fitted(model02))>0)*(Discon==2)
score2=((fitted(model01)-fitted(model02))<0)*(Discon==1)
output1=(length(which(score1==1))+length(which(score2==1)))/length(score1)

output1
==================
length(which(Discon==2))/length(Discon)
==================
score1=((log.line1-log.line2)>0)*(Discon==2)
score2=((log.line1-log.line2)<0)*(Discon==1)
output1=(length(which(score1==1))+length(which(score2==1)))/length(score1)

output1