#' Generalized ramp loss Support Vector Machine
#'
#' \code{gr_svm} is an R implementation of Generalized ramp loss Support Vector Machine
#'
#' @param X,y dataset and label.
#' @param C plenty term.
#' @param kernel kernel function. The definitions of various kernel functions are as follows:
#' \describe{
#'     \item{linear:}{\eqn{u'v}{u'*v}}
#'     \item{poly:}{\eqn{(\gamma u'v + coef0)^{degree}}{(gamma*u'*v + coef0)^degree}}
#'     \item{rbf:}{\eqn{e^{(-\gamma |u-v|^2)}}{exp(-gamma*|u-v|^2)}}
#' }
#' @param v parameter for Generalized ramp loss.
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
gr_svm <- function(X, y, C = 1, kernel = c("linear", "rbf", "poly"),
                     gamma = 1 / ncol(X), degree = 3, coef0 = 0,
                     vv = 0.1,mu = 0.5, rm = 1.618,
                     eps = 1e-5, max.steps = 200, 
                     solver = c("primal"),
                     fit_intercept = TRUE, randx = 0.1, ...) {
  gr_svm_primal_solver <- function(KernelX, y, C = 1,
                                  vv = 0.1, kernel,mu = 0.5, rm = 0.5,
                                  eps = 1e-5, max.steps = 200, ...) {
    xn <- nrow(KernelX); xp <- ncol(KernelX)
    ###set initial value
    wk <- matrix(0.01, nrow = xp, ncol = 1)
    e <- matrix(1, nrow = xn, ncol = 1)
    zk <- matrix(0, nrow = xn, ncol = 1)
    ak <- matrix(0, nrow = xn, ncol = 1)
    DA <- as.numeric(y) * KernelX
    if (kernel == "rbf") {
      inver2 = -mu * chol2inv(chol(diag(1,xp) + mu * t(KernelX)%*%KernelX)) %*% t(DA)
    }
    for (i in 1:max.steps) {
      f <- 1 - DA %*% wk
      sk <- f - ak/mu  
      index1 <- which(sk < C/(mu*vv) & sk > 0)
      index4 <- which(sk >= C/(mu*vv) & sk < (vv + C/(2*mu*vv)));index5 <- which(sk == (vv + C/(2*mu*vv)) & ak != 0)
      index2 <- c(index4,index5)
      index6 <- which(sk > 0 & sk < sqrt(2*C/mu)); index7 <- which(sk == sqrt(2*C/mu) & ak != 0)
      index3 <- c(index6,index7)
      ###Determine the working set IK and update z
      if (C/(mu*vv^2) < 2) {IK = union(index1,index2); z <- sk; z[index1] <- 0; z[index2] <- sk[index2] - C/(mu*vv) }
      if (C/(mu*vv^2) >= 2) {IK = index3; z <- sk; z[index3] <- 0 }
      #update w
      if (kernel == "linear") {
        w = wk
        if (length(IK) >= xp) {
          inver <- solve(diag(1,xp) + mu*t(DA[IK,]) %*% DA[IK,])
          w = -mu * inver %*% t(DA[IK,]) %*% (ak[IK]/mu + z[IK] - e[IK])
        }
        if (length(IK) < xp & length(IK) > 1 ) {
          inver <- solve(diag(1,length(IK)) + mu*DA[IK,] %*% t(DA[IK,]))
          w = -mu *  t(DA[IK,]) %*% inver %*% (ak[IK]/mu + z[IK] - e[IK])
        }
      }
      if (kernel == "rbf") {
        w = inver2 %*% (ak/mu + z - e)
      }
      #update alpha
      pai = z - e + DA %*% w
      a <- matrix(0, nrow = xn, ncol = 1)
      if (kernel == "linear") {
        if (length(IK) > 0) {
          a[IK] <- ak[IK] + rm * mu * pai[IK]
        }
      }
      if (kernel == "rbf") {
        a <- ak + rm * mu * pai
      }

      ###Computed termination condition
      phi1 <- norm(w + t(DA) %*% a, type = "2")/(1 + norm(w, type = "2"))
      phi2 <- norm(pai, type = "2")/sqrt(xn)
      phi3 <- norm(z - zk, type = "2")/(1+norm(zk, type = "2"))
      ma <- max(phi1,phi2,phi3)
      if (ma < eps) {
        break
      } else {
        wk <- w; zk <- z; ak = a;
      }
    }
    BaseADMMgrSVMClassifier <- list(coef = as.matrix(w[1:xp]))
    class(BaseADMMgrSVMClassifier) <- "BaseADMMgrSVMClassifier"
    return(BaseADMMgrSVMClassifier)
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
    solver.res <- gr_svm_primal_solver(KernelX, y, C, vv, kernel,mu, rm,
                                         eps, max.steps, ...)
  }
  SVMClassifier <- list("X" = X, "y" = y, "class_set" = class_set,
                        "C" = C, "kernel" = kernel,
                        "gamma" = gamma, "degree" = degree, "coef0" = coef0,
                        "solver" = solver, "coef" = solver.res$coef,
                        "fit_intercept" = fit_intercept)
  class(SVMClassifier) <- "SVMClassifier"
  return(SVMClassifier)
}
