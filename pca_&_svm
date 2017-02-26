# Digit Recognizer

library(caret)
library(e1071)

####### Reading in data
train=read.csv("../input/train.csv")
test=read.csv("../input/test.csv")
test$label=1

comb_data=rbind(train,test)
comb_data$label=as.factor(comb_data$label)
pca.train=comb_data[1:nrow(train),-1]
pca.test=comb_data[42001:nrow(comb_data),-1]

####### Data Preprocessing
# all pixels having 0 variance are removed
store=c(1)
for(i in 2:785){
  if (var(comb_data[,i])==0){store=append(store,i)}
}
####### PCA
prin_comp=prcomp(pca.train)
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)

####### Visualizing PC's
plot(prop_varex, xlab = "Principal Component",ylab = "Proportion of Variance Explained")
plot(cumsum(prop_varex), xlab = "Principal Component",ylab = "Cumulative Proportion of Variance Explained",type = "b")

train.data=data.frame(label=train$label,predict(prin_comp,pca.train))
test.data=data.frame(predict(prin_comp,pca.test))

rownames(test.data)=NULL
head(test.data)
train.data$label=as.factor(train.data$label)

####### Optimization of no.of.components
# SVM = function(x){
#     obj=svm(label~.,data=train.data[,1:x])
#     return (predict(obj,newdata=train.data[1:100,2:x]))
# }
# no_of_components=c(10,25,50,100)
# store=data.frame(labels=train.data$label[1:100])
# for (i in no_of_components){
#     mylabels=SVM(i)
#    store=cbind(store,mylabels)
#    table(store$labels,store$mylabels)
# }

####### SVM
obj=svm(label~.,data=train.data[,1:51])       ## 50 principal components 
subm = data.frame(ImageId=1:nrow(test),Label=predict(obj,newdata=test.data[,1:50]))
write.csv(subm,'/kaggle/working/svmsub.csv',row.names=F)
