#'  Primal Estimated sub-Gradient solver for SVM (Pegasos)
#'  
#' @param X,y dataset and label.
#' @param w initial point.
#' @param m mini-batch size for pegasos solver.
#' @param max.steps the number of iterations to solve the optimization problem.
#' @param fx sub-gradient of objective function.
#' @param pars parameters list for the sub-gradient.
#' @param projection projection option.
#' @param ... additional settings for the sub-gradient.
#' @return return optimal solution.
#' @references ${1:Pegasos: Primal Estimated sub-GrAdient SOlver for SVM}
#' @export
pegasos <- function(X, y, w, m, max.steps, fx, pars,projection = TRUE, ...) {
  C <- pars$C
  sample_seed <- list(...)$sample_seed
  if (is.null(sample_seed) == FALSE) {
    set.seed(sample_seed)
  }
  nx <- nrow(X)
  px <- ncol(X)
  for (t in 1:max.steps) {
    At <- sample(nx, m)
    xm <- X[At, ]
    xm <- X[At, ]
    dim(xm) <- c(m, px)
    ym <- as.matrix(y[At])
    # update parameter
    dF <- fx(xm, ym, w, pars, At = At)
    w <- w - (1/t)*dF
    if (projection == TRUE) {
      w <- min(1, sqrt(C)/norm(w, type = "2"))*w
    }
  }
  return(w)
}
