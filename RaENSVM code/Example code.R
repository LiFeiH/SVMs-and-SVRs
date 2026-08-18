# Three examples of using RaENSVM
# Example 1: Direct model fitting without cross-validation
library(ggplot2)
library(mvtnorm)
library(foreach)

source("Kernel Function.R")
source("Cross Validation function.R")
source("Metric.R")
source("RaENSVM.R")
source("gg.R")
# Generate data
set.seed(123)
mean = c(0.8,0.8)
sigma = matrix(c(0.15,0,0,0.15),2,2)
x1 = rmvnorm(n = 80, mean, sigma)
mean = c(-0.8,-0.8)
x2 = rmvnorm(n = 80,mean,sigma)
x = rbind(x1,x2)
y = c(rep(-1,80),rep(1,80))

# Add label-noise points
outnoise =  matrix(c(1.3, 1.4, 1.5, -0.3, -0.1, -0.3), ncol = 2)
x = rbind(x,outnoise)
y = c(y,c(1,1,1))
data <- data.frame(x, y)

# Bayes decision boundary
# fc = ln(p(y1))-ln(p(y2))+t((m1-m2))C(-1)*%*%x-1/2*t(m1)%*%solve(C)%*%m1+1/2*t(m2)%*%solve(C)%*%m2
slope = -1; intercept = 0; seed = 1234; k = 1
# Parameter settings
C = 2^0; tau = 0.1; eta = 2; theta = 1
# Fit the model
raen <- raen_svm(x, y, C = C, eta = eta, tau = tau, theta = theta, solver = "primal",kernel = "linear", max.steps = 5)
res <- predict(raen, x);table(res,y)

p = ggplot(data = data,mapping = aes(x = x[,1],y = x[,2],shape = as.factor(y),color = y)) + geom_point(size = 5) + theme_bw() +
  geom_abline(slope = -raen$coef[1] / raen$coef[2],intercept = -raen$coef[3] / raen$coef[2],col = "red",lty = 1,lwd = 2.5) +
  geom_abline(intercept = intercept,slope = slope,col = "black",lty = 1,lwd = 2.5) +
  geom_rect(aes(xmin = 1.25, xmax = 1.55,ymin = -0.35, ymax = -0.01),color = "cadetblue3", lty = 1, lwd = 1.7, alpha = 0) +
  annotate("text", x = 1.28, y = -0.8, parse = T, label = "label~noise",size = 13) +
  annotate("segment", x = 1.08, xend = 1.2, y = -0.7, yend = -0.31, size = 2,color = "cadetblue3", arrow = arrow(length = unit(.2,"cm")))
(p = gg(p))


# Example 2: Hyperparameter selection by cross-validation
Affdu <- read.csv("Algerian_forest_fires_dataset_UPDATE.csv",header = T,sep = ",")
X <- Affdu[, -ncol(Affdu)]
X = scale(X,center = T,scale = T)
y <- Affdu[, ncol(Affdu)]

#parameter setting
C <- 2^(-8:8); gamma <- 1; theta = c(0, 0.1, 0.5, 1); eta <- c(0.2,0.5,1,2,3); tau <- c(0, 0.1, 0.2, 0.5, 1)
metrics <- list("acc" = accuracy, "f1score" = f1score, "recall" = recall,"precision" = precision)
param_list <- list("C" = C, "gamma" = gamma, "eta" = eta, "tau" = tau, "theta" = theta)
res <- grid_search_cv(raen_svm, X, y, 5, metrics = metrics, param_list = param_list,
                       seed = 123, kernel = "linear", max.steps = 200, randx = 0.1,
                       solver = "primal", sample_seed = 123)
print(res)

# Cross-validation under label noise
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
y_noisy <- noisy_label_generator(y, 0.15, seed = 123)
res <- grid_search_cv_noisy(raen_svm, X, y, y_noisy, 5, metrics = metrics, param_list = param_list,
                             seed = 123, kernel = "linear", max.steps = 200, randx = 0.1,
                             solver = "primal", sample_seed = 123)
print(res)


# Example 3: Gaussian-kernel RaENSVM
gamma <- 2^seq(-4, 4); 
param_list <- list("C" = C, "gamma" = gamma, "eta" = eta, "tau" = tau, "theta" = theta)
res <- grid_search_cv_noisy(raen_svm, X, y, y_noisy, 5, metrics = metrics, param_list = param_list,
                            seed = 123, kernel = "rbf", max.steps = 20, randx = 0.2,
                            solver = "primal", sample_seed = 123)
print(res)











