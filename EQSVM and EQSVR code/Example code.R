###Three example for how to use EQSVM
###example 1: Use it directly without invoking cross validation
library(MASS)
library(ggplot2)
library("mvtnorm")
source("gg.R")
source("eq_svm.R")
source("Clip-DCD optimizer.R")
source("Pegasos optimizer.R")
source("Kernel Function.R")
source("Cross Validation function.R")
#Generate date
set.seed(111)
mean = c(0.8,0.4)
sigma = matrix(c(0.1,0,0,0.1),2,2)
x1 = rmvnorm(n = 80, mean, sigma)
mean = c(-0.8,-0.4)
x2 = rmvnorm(n = 80,mean,sigma)
x = rbind(x1,x2)
y = c(rep(-1,80),rep(1,80))
data <- data.frame(x, y)

#Bayesian decision boundary:
#fc = ln(p(y1))-ln(p(y2))+t((m1-m2))C(-1)*%*%x-1/2*t(m1)%*%solve(C)%*%m1+1/2*t(m2)%*%solve(C)%*%m2
slope = -2; intercept = 0; seed = 1234; k = 1
#Parameter setting
C = 2^(8); lambda = 2^(0); tau = 0; m = 0
cccp.steps = 10; eps.cccp = 1e-3
#Model solving
eq <- eq_svm(x, y, C = C, lambda = lambda, m = m, tau = tau, solver = "dual",kernel = "linear",
             max.steps = 200,cccp.steps = cccp.steps,eps.cccp = eps.cccp )
res <- predict(eq, x);table(res,y)
eq$coef = t(cbind(x,1)) %*% eq$coef

#分类超平面方程
p = ggplot(data = data,mapping = aes(x = x[,1],y = x[,2],shape = as.factor(y),color = y)) + geom_point(size = 5) + theme_bw() +
  geom_abline(slope = -eq$coef[1] / eq$coef[2],intercept = -eq$coef[3] / eq$coef[2],col = "red",lty = 1,lwd = 2.5) +
  geom_abline(intercept = intercept,slope = slope,col = "black",lty = 1,lwd = 2.5) +
  geom_abline(slope = -eq$coef[1] / eq$coef[2],intercept = (1 - eq$coef[3]) / eq$coef[2],col = "green",lty = 4,lwd = 2.5) +
  geom_abline(slope = -eq$coef[1] / eq$coef[2],intercept = (-1 - eq$coef[3]) / eq$coef[2],col = "green",lty = 4,lwd = 2.5)
(p = gg(p))

#Using the Pegasos solver
#Model solving
eq <- eq_svm(x, y, C = C, lambda = lambda, m = m, tau = tau, solver = "primal",kernel = "linear",
             max.steps = 5000 )
res <- predict(eq, x);table(res,y)

#Classification hyperplane
p = ggplot(data = data,mapping = aes(x = x[,1],y = x[,2],shape = as.factor(y),color = y)) + geom_point(size = 5) + theme_bw() +
  geom_abline(slope = -eq$coef[1] / eq$coef[2],intercept = -eq$coef[3] / eq$coef[2],col = "red",lty = 1,lwd = 2.5) +
  geom_abline(intercept = intercept,slope = slope,col = "black",lty = 1,lwd = 2.5) +
  geom_abline(slope = -eq$coef[1] / eq$coef[2],intercept = (1 - eq$coef[3]) / eq$coef[2],col = "green",lty = 4,lwd = 2.5) +
  geom_abline(slope = -eq$coef[1] / eq$coef[2],intercept = (-1 - eq$coef[3]) / eq$coef[2],col = "green",lty = 4,lwd = 2.5)
(p = gg(p))

###example 2: Call cross-validation for parameter tuning
source("Metric.R")
library("foreach")
source("Cross Validation function.R")
Affdu <- read.csv("Algerian_forest_fires_dataset_UPDATE.csv",header = T,sep = ",")
X <- Affdu[, -ncol(Affdu)]
X = scale(X,center = T,scale = T)
y <- Affdu[, ncol(Affdu)]

#parameter setting
C <- 2^(-8:8); gamma <- 1; lambda_eq <- 2^seq(-5,1,1); tau <- c(0, 0.1, 0.2, 0.5,1)
cccp.steps <- 10; cccp.eps <- 1e-3
metrics <- list("acc" = accuracy, "f1score" = f1score, "recall" = recall,"precision" = precision)
param_list <- list("C" = C, "lambda" = lambda_eq, "tau" = tau)
res <- grid_search_cv(eq_svm, X, y, 5, metrics = metrics, param_list = param_list,
                       cccp.eps = cccp.eps, seed = 666, kernel = "linear", max.steps = 80, cccp.steps = cccp.steps,
                       solver = "dual", sample_seed = 123)
print(res)

#Using the Pegasos solver
res <- grid_search_cv(eq_svm, X, y, 5, metrics = metrics, param_list = param_list,
                       seed = 666, kernel = "linear", max.steps = 500, 
                       solver = "primal", randx = 1, batch_size = floor(nrow(X)*0.8), sample_seed = 123)
print(res)

#Cross validation with noise
noisy_label_generator <- function(y, p, seed = NULL){
  if (is.null(seed) == FALSE) {
    set.seed(seed)
  }
  y <- as.matrix(y)
  class_set <- unique(y)
  class_num <- length(class_set)
  class_idx <- list()
  for (i in 1:class_num) {
    idx <- which(y == class_set[i])
    class_idx[[i]] <- idx
  }
  for (i in 1:class_num) {
    m <- length(class_idx[[i]])
    n <- round(m*p, 0)
    idx_temp <- sample(m, n)
    noisy_y <- sample(class_set[class_set != class_set[i]], n, replace = T)
    y[class_idx[[i]][idx_temp]] <- noisy_y
  }
  return(y)
}
y_noisy <- noisy_label_generator(y, 0.3, seed = 123)
res <- grid_search_cv_noisy(eq_svm, X, y, y_noisy, 5, metrics = metrics, param_list,
                             cccp.eps = cccp.eps, seed = 123, kernel = "linear", max.steps = 200, cccp.steps = cccp.steps,
                             solver = "dual", sample_seed = 123)
print(res)

#Using the Pegasos solver
res <- grid_search_cv_noisy(eq_svm, X, y, y_noisy, 5, metrics = metrics, param_list,
                            seed = 123, kernel = "linear", max.steps = 500,
                            solver = "primal", randx = 1, batch_size = floor(nrow(X)*0.8), sample_seed = 123)
print(res)


###example 3: Using Gaussian kernel functions
res <- grid_search_cv(eq_svm, X, y, 5, metrics = metrics, param_list = param_list,
                      cccp.eps = cccp.eps, seed = 123, kernel = "rbf", max.steps = 200, cccp.steps = cccp.steps,
                      solver = "dual", randx = 1, batch_size = floor(nrow(X)*0.8), sample_seed = 123)
print(res)


###An example for how to use EQSVR
###example 1: Use it directly without invoking cross validation
source("eq_svr.R")
source("Cross Validation function.R")
library(latex2exp)
library(reshape2)
set.seed(123); seed = 123; n = 160
x = c(-n:n)/(n/4); x = x[x != 0]
y = sin(3*x)/(3*x)
data <- data.frame(x, y)
#Optimal regression line y = sin(3*x)/(3*x)

#parameter setting
C = 1; gamma <- 1; lambda = 1; tau = 1; 
eq_dual <- eq_svr(x, y, C = C, lambda = lambda, tau = tau, gamma = gamma,solver = "dual",max.steps = 200,kernel = "rbf")
pre_eq_dual <- predict(eq_dual, x)

#Using the Pegasos solver
eq_pegasos <- eq_svr(x, y, C = C, lambda = lambda, tau = tau, gamma = gamma, solver = "primal",max.steps = 50000,kernel = "rbf")
pre_eq_pegasos <- predict(eq_pegasos, x)

#visualization
dataPred <- data.frame("x" = x,
                       "EQSVR_dual" = pre_eq_dual ,
                       "EQSVR_primal" = pre_eq_pegasos ,
                       "Real_Model" = sin(3*x)/(3*x))
dataXy <- data.frame("x" = x, "y" = y)
dataPred <- melt(dataPred, id.vars = "x")
mylegend <- c(TeX("EQSVR_dual"),
              TeX("EQSVR_primal"),
              TeX("${\\sin 3x}/{3x}$"))

p = ggplot(data = dataXy,mapping = aes(x = x,y = y)) + geom_point(size = 2) + theme_bw() +
  geom_line(data = dataPred, aes(x = x, y = value, color = variable,linetype = variable,),size = 2) +
  scale_color_discrete(labels = mylegend) + 
  scale_shape_discrete(labels = mylegend) + 
  scale_linetype_discrete(labels = mylegend) +
  theme(axis.line = element_line(linetype = "solid"),
        panel.grid.major = element_line(colour = "gray95",
        size = 1), panel.grid.minor = element_line(colour = "gray95",
        size = 1), axis.title = element_text(family = "serif",
        size = 20, face = "bold"), axis.text = element_text(family = "serif",
        size = 17, face = "bold"), axis.text.y = element_text(family = "serif"),
        legend.text = element_text(size = 25,hjust = 0,
                                   family = "sans"), legend.title = element_text(size = 28,
                                   face = "bold", family = "serif"),
        panel.background = element_rect(fill = "gray100",
                                        colour = "gray0", size = 0.5, linetype = "solid"),
        plot.background = element_rect(linetype = "solid"),
        legend.key = element_rect(fill = NA),
        legend.background = element_rect(fill = NA),
        legend.position = c(0.8, 0.8)) + labs(x = "x", y = NULL)
p












