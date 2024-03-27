#' Exponential Quantile Loss Support Vector Machine
#'
#' \code{eq_svm} is an R implementation of three-parameter version EQSVM
#' The two-parameter version only needs to set m to 10*tau
#' These include solvers using the ClipDCD-based CCCP algorithm and solvers using the Pegasos algorithm
#'
#' @param X,y dataset and label.
#' @param C plenty term.
#' @param kernel kernel function. The definitions of various kernel functions are as follows:
#' \describe{
#'     \item{linear:}{\eqn{u'v}{u'*v}}
#'     \item{poly:}{\eqn{(\gamma u'v + coef0)^{degree}}{(gamma*u'*v + coef0)^degree}}
#'     \item{rbf:}{\eqn{e^{(-\gamma |u-v|^2)}}{exp(-gamma*|u-v|^2)}}
#' }
#' @param tau parameter for exponential quantile loss (Control the degree of asymmetry).
#' @param lambda parameter for exponential quantile loss (loss increase speed).
#' @param m parameter for exponential quantile loss (Loss at zero of the growth rate).
#' @param gamma parameter for \code{'rbf'} and \code{'poly'} kernel. Default \code{gamma = 1/ncol(X)}.
#' @param degree parameter for polynomial kernel, default: \code{degree = 3}.
#' @param coef0 parameter for polynomial kernel,  default: \code{coef0 = 0}.
#' @param eps the precision of the optimization algorithm.
#' @param max.steps the number of iterations to solve the optimization problem.
#' @param batch_size mini-batch size for primal solver.
#' @param solver \code{"dual"} and \code{"primal"} are available.
#' @param rcpp speed up your code with Rcpp, default \code{rcpp = TRUE}.
#' @param fit_intercept if set \code{fit_intercept = TRUE},
#'                      the function will evaluates intercept.
#' @param optimizer default primal optimizer pegasos.
#' @param randx parameter for reduce SVM, default \code{randx = 0.1}.
#' @param ... unused parameters.
#' @return return \code{SVMClassifier} object.
#' @export
eq_svm <- function(X, y, C = 1, kernel = c("linear", "rbf", "poly"),
                   gamma = 1 / ncol(X), degree = 3, coef0 = 0,
                   lambda = 1, tau = 0.5, m = 0,
                   eps = 1e-5, eps.cccp = 1e-2, max.steps = 80, cccp.steps = 10,
                   batch_size = nrow(X) / 10, optimizer = pegasos, randx = 0.1,
                   solver = c("dual","primal"),
                   fit_intercept = TRUE, ...) {
  #m <- 10*tau #(The two-parameter version only needs to set m to 10*tau)
  eq_svm_dual_solver <- function(KernelX, y, C = 1, update_deltak,
                                 lambda = 1, tau = 0, m = 0, 
                                 eps = 1e-5, eps.cccp = 1e-2, max.steps = 80, cccp.steps = 10) {
    ###ClipDCD-based CCCP algorithm for EQSVM
    eta <- (1 + exp(m))/exp(m)
    n <- nrow(KernelX)
    H <-  (y%*%t(y))*KernelX
    H1 <- cbind(lambda^2*H,-lambda^2*tau*H)
    H1 <- rbind(H1,-H1*tau)
    I = diag(1,nrow = n)
    H2 = cbind(I,I)
    H2 = rbind(H2,H2)
    H3 = H1 + 5*H2/(eta*C)
    u0 <- matrix(0.1, 2*n, 1)
    e <- matrix(1, n, 1)
    delta_k = matrix(0, n, 1)
    for (i in 1:cccp.steps) {
      f <- 1 - H %*% (lambda*u0[1:n] - tau*lambda*u0[(n + 1):(2*n)])
      delta_k <- update_deltak(f, lambda, tau, m, eta )
      e2 = rbind(lambda * e, -tau*lambda * e) + rbind((e/(1 + exp(m)) - delta_k/lambda), (e/(1 + exp(m)) - delta_k/lambda))*5/eta
      u <- clip_dcd_optimizer(H3 , e2, rbind(-(C*delta_k)/lambda,matrix(0,n,1)), matrix(Inf, 2*n, 1), eps, max.steps, u0)$x
      if (norm(u - u0, type = "2") < eps.cccp) {
        break
      } else {
        u0 <- u
      }
    }
    coef <- y * (lambda*u[1:n] - tau*lambda*u[(n + 1):(2*n)])
    BaseDualEQSVMClassifier <- list(coef = as.matrix(coef))
    class(BaseDualEQSVMClassifier) <- "BaseDualEQSVMClassifier"
    return(BaseDualEQSVMClassifier)
  }
  eq_svm_primal_solver <- function(KernelX, y, C = 1,
                                   lambda = 1, tau = 0.5, m = 0, 
                                   eps = 1e-5, max.steps = 80,
                                   batch_size = nrow(KernelX) / 10,
                                   seed = NULL,
                                   optimizer = pegasos, ...) {
    ###Pegasos-based CCCP algorithm for EQSVM
    sgeq <- function(KernelX, y, w, pars, At, ...) { # sub-gradient of exponential quantile loss function
      lambda <- pars$lambda
      eta <- pars$eta
      m <- pars$m
      tau <- pars$tau
      xn <- pars$xn
      xmn <- nrow(KernelX)
      sg <- matrix(0, length(y))
      z <- 1 - y * (KernelX %*% w)
      idx <- which(z >= 0)
      termPos <- exp(-lambda*z[idx] + m)
      termNeg <- exp(tau*lambda*z[-idx] + m)
      sg[idx] <- lambda*eta*termPos/(1 + termPos)^2
      sg[-idx] <- -tau*lambda*eta*termNeg/(1 + termNeg)^2
      sg <- w - (C*xn/xmn)*t(KernelX) %*% (y*sg)
      return(sg)
    }
    
    eta <- (1 + exp(m))/exp(m)
    xn <- nrow(KernelX)
    xp <- ncol(KernelX)
    w0 <- matrix(0, xp, 1)
    pars <- list("C" = C, "xn" = xn,
                 "lambda" = lambda, "eta" = eta, tau = tau,"m" = m)
    wt <- optimizer(KernelX, y, w0, batch_size, max.steps, sgeq, pars, ...)
    BasePrimalEQSVMClassifier <- list(coef = as.matrix(wt[1:xp]))
    class(BasePrimalEQSVMClassifier) <- "BasePrimalEQSVMClassifier"
    return(BasePrimalEQSVMClassifier)
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
  kso <- kernel_select_option(X, kernel, solver, randx,
                              gamma, degree, coef0)
  KernelX <- kso$KernelX
  X <- kso$X
  update_deltak <- function(f, lambda, tau, m,eta) {
    delta_k <- matrix(0, nrow = length(f))
    idx <- which(f >= 0)
    ep <- exp(-lambda*f[idx] + m)
    ept <- exp(lambda*tau*f[-idx] + m )
    delta_k[idx] <-  (lambda^2)*eta*f[idx]/5 + lambda/(1 + exp(m)) - lambda*eta*ep / ((1 + ep)^2)
    delta_k[-idx] <- (lambda^2*tau^2)*eta*f[-idx]/5 - lambda*tau/(1 + exp(m)) + lambda*tau*eta*ept/((1 + ept)^2)
    return(delta_k)
  }
  if (solver == "dual") {
    solver.res <- eq_svm_dual_solver(KernelX, y, C, update_deltak, lambda, tau, m,
                                     eps, eps.cccp, max.steps, cccp.steps)
  } else {
    solver.res <- eq_svm_primal_solver(KernelX, y, C, lambda, tau, m,
                                     eps, max.steps, batch_size, optimizer,...)
  }
  SVMClassifier <- list("X" = X, "y" = y, "class_set" = class_set,
                        "C" = C, "kernel" = kernel,
                        "gamma" = gamma, "degree" = degree, "coef0" = coef0,
                        "solver" = solver, "coef" = solver.res$coef,
                        "fit_intercept" = fit_intercept)
  class(SVMClassifier) <- "SVMClassifier"
  return(SVMClassifier)
}
