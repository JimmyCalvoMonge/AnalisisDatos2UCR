setwd("C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/AnalisisDatos2/Tarea8")

library(MASS)
library(class)
library(caret)
library(dummies)

datos<-read.csv("Ejemplo_AD.csv",sep = ";",dec='.',header=F,row.names = 1,stringsAsFactors = T)
datos<-datos[,c(1,2,3,4,5,6)]
colnames(datos)<-c("RT1","RT2","RT3","RT4","RT5","VC")
for(i in c(1:5)){
  datos[,i]=datos[,i]-mean(datos[,i])
}

X <- data.matrix(datos)
X1<- X[,-6]
tempo <- dummy.data.frame(datos, sep = ".")
X2 <- tempo[,c(6,7,8)]
D_G <- table(datos$VC,datos$VC)
I=diag(1,nrow=dim(D_G)[1])
inv.D_G <- solve(D_G,I)
G <- inv.D_G %*% t(X2) %*% X1
temp_mat_1=solve(t(X1)%*%X1,diag(1,nrow=dim(t(X1)%*%X1)[1]))
ACP_mat=temp_mat_1%*%t(G)%*%D_G%*%G
eigs=eigen(ACP_mat)
eigs$vectors


modelo <- lda(VC~., data=datos)
modelo


