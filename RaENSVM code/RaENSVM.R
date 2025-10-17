#' Rescaled asymmetric elastic net Support Vector Machine
#'
#' \code{RaENSVM} is an R implementation of RaENSVM
#'
#' @param X,y dataset and label.
#' @param C plenty term.
#' @param kernel kernel function. The definitions of various kernel functions are as follows:
#' \describe{
#'     \item{linear:}{\eqn{u'v}{u'*v}}
#'     \item{poly:}{\eqn{(\gamma u'v + coef0)^{degree}}{(gamma*u'*v + coef0)^degree}}
#'     \item{rbf:}{\eqn{e^{(-\gamma |u-v|^2)}}{exp(-gamma*|u-v|^2)}}
#' }
#' @param tau parameter for RaEN loss (Control the degree of asymmetry).
#' @param eta parameter for RaEN loss (scaling constant).
#' @param theta parameter for RaEN loss (Control the weight of the primary and secondary terms).
#' @param gamma parameter for \code{'rbf'} and \code{'poly'} kernel. Default \code{gamma = 1/ncol(X)}.
#' @param degree parameter for polynomial kernel, default: \code{degree = 3}.
#' @param coef0 parameter for polynomial kernel,  default: \code{coef0 = 0}.
#' @param eps the precision of the optimization algorithm.
#' @param max.steps the number of iterations to solve the optimization problem.
#' @param solver \code{"primal"} are available.
#' @param fit_intercept if set \code{fit_intercept = TRUE},
#'                      the function will evaluates intercept.
#' @param randx parameter for reduce SVM, default \code{randx = 0.1}.
#' @param ... unused parameters.
#' @return return \code{SVMClassifier} object.
#' @export
#' 
raen_svm <- function(X, y, C = 1, kernel = c("linear", "rbf", "poly"),
                     gamma = 1 / ncol(X), degree = 3, coef0 = 0,
                     mu = 0.5, rm = 1.618,
                     eta = 1, tau = 0.5, theta = 0.5,
                     eps = 1e-5, max.steps = 200, 
                     solver = c("primal"),
                     fit_intercept = TRUE, randx = 0.1, ...) {
  raen_svm_primal_solver <- function(KernelX, y, C = 1, kernel,mu, rm,
                                     eta = 1, tau = 0.5, theta = 0.5,
                                     eps = 1e-5, max.steps = 200, ...) {
    laen <- function(z,theta,tau){
      value = matrix(0, nrow = nrow(z), ncol = 1)
      value[z >= 0] = theta/2 * (z[z >= 0])^2 + (1 - theta) * z[z >= 0]
      value[z < 0] = tau * (theta/2 * (z[z < 0])^2 - (1 - theta) * z[z < 0]) 
      return(value)
    }
    xn <- nrow(KernelX); xp <- ncol(KernelX)
    beta <- 1/(1 - exp(-eta))
    ###set initial value
    wk <- matrix(0.01, nrow = xp, ncol = 1)
    e <- matrix(1, nrow = xn, ncol = 1)
    zk <- e - y * (KernelX %*% wk)
    vk <- -exp(-eta*laen(zk,theta,tau))
    ak <- matrix(0, nrow = xn, ncol = 1)
    DA <- as.numeric(y) * KernelX
    if(kernel == "linear"){
      if (tau != 0 ) {
        if (xn >= xp) {
          inver1 <- -mu *  chol2inv(chol(diag(1,xp) + mu * t(DA) %*% DA)) %*% t(DA)
        }
        if (xn < xp) {
          inver2 <- -mu * t(DA) %*% chol2inv(chol(diag(1,xn) + mu * DA %*% t(DA)))
        }
      }
    }
    if(kernel == "rbf") {
      inver1 <- -mu * chol2inv(chol(diag(1,xp) + mu * t(KernelX)%*%KernelX)) %*% t(DA)
    }
    loss_ <- matrix(0, max.steps)
    for (i in 1:max.steps) {
      f <- 1 - DA %*% wk
      loss_pos <- C*beta*sum(1 - exp(-eta*(0.5*(theta)*f[f >= 0]^2 + (1 - theta)*f[f >= 0])))
      loss_neg <- C*beta*sum(1 - exp(-eta*tau*(0.5*(theta)*f[f < 0]^2 - (1 - theta)*f[f < 0])))
      loss_[i] <-  loss_pos + loss_neg + 0.5*norm(wk, type = "2")^2
      sk <- f - ak/mu 
      cbe = C * beta * eta; cbet = cbe * (1 - theta)
      #updata v
      v = -exp(-eta*laen(f,theta,tau))
      #update z
      index1 <- which(sk >= -(cbet/mu) * v)
      index2 <- which(sk < -(cbet/mu) * v & sk > (cbet * tau/mu) * v)
      index3 <- which(sk <= (cbet * tau/mu) * v)
      z = zk
      z[index1] <- (sk[index1] + (cbet/mu) * v[index1])/(1 - (cbe * theta/mu) * v[index1]) 
      z[index2] <- 0 
      z[index3] <- (sk[index3] - (cbet * tau/mu) * v[index3])/(1 - (cbe * theta * tau/mu) * v[index3]) 
      #update w
      if (tau == 0) {
        IK <- which(sk > 0)
      }
      if (tau != 0) {
        IK <- which(sk != 0)
      }
      if (length(IK) >= xp || kernel == "rbf") {
        if ( tau == 0 & kernel == "linear") {
          inver <- solve(diag(1,xp) + mu*t(DA[IK,]) %*% DA[IK,])
          w = -mu * inver %*% t(DA[IK,]) %*% (ak[IK]/mu + z[IK] - e[IK])
        }
        if ( tau != 0 || kernel == "rbf") {
          w = inver1 %*% (ak/mu + z - e)
        }
      }
      if (length(IK) < xp & kernel == "linear") {
        if ( tau == 0 & length(IK) >1) {
          inver <- solve(diag(1,length(IK)) + mu*DA[IK,] %*% t(DA[IK,]))
          w = -mu *  t(DA[IK,]) %*% inver %*% (ak[IK]/mu + z[IK] - e[IK])
        }
        if ( tau != 0) {
          w = inver2 %*% (ak/mu + z - e)
        }
      }
      #update alpha
      dert = z - e + DA %*% w##在这儿把wk换成了w
      a <- matrix(0, nrow = xn, ncol = 1)
      if (kernel == "linear") {
        if (length(IK) > 0) {
          a[IK] <- ak[IK] + rm * mu * dert[IK]
        }
      }
      if (kernel == "rbf") {
        a <- ak + rm * mu * dert
      }
      # ###Computed termination condition
      phi1 <- norm(w + t(DA) %*% a, type = "2")/(1 + norm(w, type = "2"))
      phi2 <- norm(dert, type = "2")/sqrt(xn)
      phi3 <- norm(v - vk, type = "2")/(1 + norm(vk, type = "2"))
      phi4 <- norm(z - zk, type = "2")/(1 + norm(zk, type = "2"))
      ma <- max(phi1,phi2,phi3,phi4)
      if (ma < eps) {
        break
      } else {
        wk <- w; ak = a; zk = z; vk = v
      }
    }
    BaseADMMRaENSVMClassifier <- list(coef = as.matrix(w[1:xp]),"loss" = loss_)
    class(BaseADMMRaENSVMClassifier) <- "BaseADMMRaENSVMClassifier"
    return(BaseADMMRaENSVMClassifier)
  }
  
  X <- as.matrix(X)
  y <- as.matrix(y)
  class_set <- unique(y)
  idx <- which(y == class_set[1])
  y[idx] <- 1
  y[-idx] <- -1
  y <- as.matrix(as.numeric(y))
  if (length(class_set) > 2) {
    stop("The number of class should less 2!")
  }
  kernel <- match.arg(kernel)
  solver <- match.arg(solver)
  if (fit_intercept == TRUE) {
    X <- cbind(X, 1)
  }
  kso <- manysvms:::kernel_select_option(X, kernel, solver, randx,
                                         gamma, degree, coef0)
  KernelX <- kso$KernelX
  X <- kso$X
  if (solver == "primal") {
    solver.res <- raen_svm_primal_solver(KernelX, y, C, kernel, mu, rm,
                                         eta, tau, theta,
                                         eps, max.steps, ...)
  }
  SVMClassifier <- list("X" = X, "y" = y, "class_set" = class_set,
                        "C" = C, "kernel" = kernel, "loss" = solver.res$loss,
                        "gamma" = gamma, "degree" = degree, "coef0" = coef0,
                        "solver" = solver, "coef" = solver.res$coef,
                        "fit_intercept" = fit_intercept)
  class(SVMClassifier) <- "SVMClassifier"
  return(SVMClassifier)
}
