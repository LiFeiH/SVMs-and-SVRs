#' K-Fold Cross Validation
#'
#' @param model your model.
#' @param X,y dataset and label.
#' @param K number of folds.
#' @param metrics this parameter receive a metric function.
#' @param predict_func this parameter receive a function for predict.
#' @param ... additional parameters for your model.
#' @return return a metric matrix
#' @export
cross_validation <- function(model, X, y, K = 5, metrics, ...) {
  X <- as.matrix(X)
  y <- as.matrix(y)
  n <- nrow(X)
  if (is.list(metrics) == F) {
    metrics <- list(metrics)
    names(metrics) <- paste("metric", length(metrics), sep = "")
  }
  num_metric <- length(metrics)
  metric_mat <- matrix(0, num_metric, K)
  index <- sort(rep(1:K, length.out = n))
  for (i in 1:K) {
    idx <- which(index == i)
    X_test <- X[idx, ]
    y_test <- y[idx]
    if (K == 1) {
      X_train <- X_test
      y_train <- y_test
    }else{
      X_train <- X[-idx, ]
      y_train <- y[-idx]
    }
    model_res <- do.call("model", list("X" = X_train, "y" = y_train, ...))
    y_test_hat <- predict.SVMClassifier(model_res, X_test, ...)
    for (j in 1:num_metric) {
      metric_mat[j, i] <- metrics[[j]](y_test, y_test_hat)
    }
  }
  rownames(metric_mat) <- names(metrics)
  return(metric_mat)
}


#' Grid Search and Cross Validation
#'
#' @param model your model.
#' @param X,y dataset and label.
#' @param K number of folds.
#' @param metrics this parameter receive a metric function.
#' @param param_list parameter list.
#' @param predict_func this parameter receive a function for predict.
#' @param shuffle if set \code{shuffle==TRUE}, This function will shuffle the dataset.
#' @param seed random seed for \code{shuffle} option.
#' @param threads.num the number of threads used for parallel execution.
#' @param ... additional parameters for your model.
#' @return return a metric matrix
#' @import foreach
#' @import doParallel
#' @import doSNOW
#' @import stats
#' @export
grid_search_cv <- function(model, X, y, K = 5, metrics, param_list,
                           shuffle = TRUE, seed = NULL,
                           threads.num = parallel::detectCores() - 1, ...) {
  s <- Sys.time()
  X <- as.matrix(X)
  y <- as.matrix(y)
  if (is.list(metrics) == F) {
    metrics <- list(metrics)
    names(metrics) <- paste("metric", length(metrics), sep = "")
  }
  n <- nrow(X)
  if (is.null(seed) == FALSE) {
    set.seed(seed)
  }
  if (shuffle == TRUE) {
    idx <- sample(n)
    X <- X[idx, ]
    y <- y[idx]
  }
  param_grid <- expand.grid(param_list)
  n_param <- nrow(param_grid)
  param_names <- colnames(param_grid)
  cl <- parallel::makeCluster(threads.num)
  pb <- utils::txtProgressBar(max = n_param, style = 3)
  progress <- function(n){utils::setTxtProgressBar(pb, n)}
  opts <- list(progress = progress)
  doSNOW::registerDoSNOW(cl)
  i <- 1
  cv_res <- foreach::foreach(i = 1:n_param, .combine = rbind,.export = c("kernel_select_option", "cross_validation","kernel_function",
                                                               "r_linear_kernel", "r_rbf_kernel", "r_poly_kernel",
                                                               "clip_dcd_optimizer","predict.SVMClassifier","pegasos"),
                             .options.snow = opts) %dopar% {
                               temp <- data.frame(param_grid[i, ])
                               colnames(temp) <- param_names
                               params_cv <- append(list("model" = model,
                                                        "X" = X, "y" = y, "K" = K,
                                                        "metrics" = metrics,
                                                        ...),
                                                   temp)
                               cv_res <- do.call("cross_validation", params_cv)
                               cv_res <- rbind(c(apply(cv_res, 1, mean), apply(cv_res, 1, sd)))
                             }
  close(pb)
  parallel::stopCluster(cl)
  cat("\n")
  num_metrics <- length(metrics)
  name_matrics <- names(metrics)
  colnames(cv_res)[(num_metrics+1):(2*num_metrics)] <- paste(name_matrics, "- sd")
  e <- Sys.time()
  idx_max <- apply(as.matrix(cv_res[,1:num_metrics]), 2, which.max)
  idx_min <- apply(as.matrix(cv_res[,1:num_metrics]), 2, which.min)
  score_mat <- matrix(0, 2, 2*num_metrics)
  rownames(score_mat) <- c("max", "min")
  colnames(score_mat) <- c(name_matrics, paste(name_matrics, "- sd"))
  for (i in 1:num_metrics) {
    score_mat[1, i] <- cv_res[idx_max[i], i]
    score_mat[2, i] <- cv_res[idx_min[i], i]
    score_mat[1, num_metrics+i] <- cv_res[idx_max[i], num_metrics+i]
    score_mat[2, num_metrics+i] <- cv_res[idx_min[i], num_metrics+i]
  }
  cv_res <- cbind(cv_res, param_grid)
  cv_model <- list("results" = cv_res,
                   "idx_max" = idx_max,
                   "idx_min" = idx_min,
                   "num.parameters" = n_param,
                   "K" = K,
                   "time" = e - s,
                   "score_mat" = score_mat,
                   "param_grid" = param_grid
  )
  class(cv_model) <- "cv_model"
  return(cv_model)
}


#' Print Method for Grid-Search and Cross Validation Results
#'
#' @param x object of class \code{eps.svr}.
#' @param ... unsed argument.
#' @export
print.cv_model <- function(x, ...) {
  cat("Results of Grid Search and Cross Validation\n\n")
  cat("Number of Fold", x$K, "\n")
  cat("Total Parameters:", x$num.parameters, "\n\n")
  cat("Time Cost:\n")
  print(x$time)
  cat("Summary of Metrics\n\n")
  print(x$score_mat)
}


#' Grid Search and Cross Validation with Noisy (Simulation Only)
#'
#' @param model your model.
#' @param X,y dataset and label.
#' @param y_noisy label (contains label noise)
#' @param K number of folds.
#' @param metrics this parameter receive a metric function.
#' @param param_list parameter list.
#' @param predict_func this parameter receive a function for predict.
#' @param shuffle if set \code{shuffle==TRUE}, This function will shuffle the dataset.
#' @param seed random seed for \code{shuffle} option.
#' @param threads.num the number of threads used for parallel execution.
#' @param ... additional parameters for your model.
#' @return return a metric matrix
#' @import foreach
#' @import doParallel
#' @import doSNOW
#' @import stats
#' @export
grid_search_cv_noisy <- function(model, X, y, y_noisy, K = 5, metrics, param_list,
                                 shuffle = TRUE, seed = NULL,
                                 threads.num = parallel::detectCores() - 1,
                                 ...) {
  s <- Sys.time()
  X <- as.matrix(X)
  y <- as.matrix(y)
  if (is.list(metrics) == F) {
    metrics <- list(metrics)
    names(metrics) <- paste("metric", length(metrics), sep = "")
  }
  n <- nrow(X)
  if (is.null(seed) == FALSE) {
    set.seed(seed)
  }
  if (shuffle == TRUE) {
    idx <- sample(n)
    X <- X[idx, ]
    y <- y[idx]
    y_noisy <- y_noisy[idx]
  }
  param_grid <- expand.grid(param_list)
  n_param <- nrow(param_grid)
  param_names <- colnames(param_grid)
  cl <- parallel::makeCluster(threads.num)
  pb <- utils::txtProgressBar(max = n_param, style = 3)
  progress <- function(n){utils::setTxtProgressBar(pb, n)}
  opts <- list(progress = progress)
  doSNOW::registerDoSNOW(cl)
  i <- 1
  cv_res <- foreach::foreach(i = 1:n_param, .combine = rbind,.export = c("kernel_select_option", "cross_validation_noisy","kernel_function",
                                                                         "r_linear_kernel", "r_rbf_kernel", "r_poly_kernel",
                                                                         "clip_dcd_optimizer","predict.SVMClassifier","pegasos"),
                             .options.snow = opts) %dopar% {
                               temp <- data.frame(param_grid[i, ])
                               colnames(temp) <- param_names
                               params_cv <- append(list("model" = model,
                                                        "X" = X, "y" = y, "y_noisy" = y_noisy, "K" = K,
                                                        "metrics" = metrics,
                                                        ...),
                                                   temp)
                               cv_res <- do.call("cross_validation_noisy", params_cv)
                               cv_res <- rbind(c(apply(cv_res, 1, mean), apply(cv_res, 1, sd)))
                             }
  parallel::stopCluster(cl)
  cat("\n")
  num_metrics <- length(metrics)
  name_matrics <- names(metrics)
  colnames(cv_res)[(num_metrics+1):(2*num_metrics)] <- paste(name_matrics, "- sd")
  e <- Sys.time()
  idx_max <- apply(as.matrix(cv_res[,1:num_metrics]), 2, which.max)
  idx_min <- apply(as.matrix(cv_res[,1:num_metrics]), 2, which.min)
  score_mat <- matrix(0, 2, 2*num_metrics)
  rownames(score_mat) <- c("max", "min")
  colnames(score_mat) <- c(name_matrics, paste(name_matrics, "- sd"))
  for (i in 1:num_metrics) {
    score_mat[1, i] <- cv_res[idx_max[i], i]
    score_mat[2, i] <- cv_res[idx_min[i], i]
    score_mat[1, num_metrics+i] <- cv_res[idx_max[i], num_metrics+i]
    score_mat[2, num_metrics+i] <- cv_res[idx_min[i], num_metrics+i]
  }
  cv_res <- cbind(cv_res, param_grid)
  cv_model <- list("results" = cv_res,
                   "idx_max" = idx_max,
                   "idx_min" = idx_min,
                   "num.parameters" = n_param,
                   "K" = K,
                   "time" = e - s,
                   "score_mat" = score_mat,
                   "param_grid" = param_grid
  )
  class(cv_model) <- "cv_model"
  return(cv_model)
}

predict.SVMClassifier <- function(object, X, values = FALSE, ...) {
  X <- as.matrix(X)
  if (object$fit_intercept == TRUE) {
    X <- cbind(X, 1)
  }
  if (object$kernel == "linear" & object$solver == "primal") {
    KernelX <- X
  } else {
    KernelX <- kernel_function(X, object$X,
                               kernel.type = object$kernel,
                               gamma = object$gamma,
                               degree = object$degree,
                               coef0 = object$coef0)
  }
  fx <- KernelX %*% object$coef
  if (values == FALSE) {
    decf <- sign(fx)
    idx_pos <- which(decf > 0)
    idx_neg <- which(decf < 0)
    decf[idx_pos] <- object$class_set[1]
    decf[idx_neg] <- object$class_set[2]
  } else {
    decf <- fx
  }
  return(decf)
}
#' Predict Method for Support Vector Regression
#'
#' @param object a fitted object of class inheriting from \code{SVMRegressor}.
#' @param X new data for predicting.
#' @param ... unused parameter.
#' @importFrom stats predict
#' @export
predict.SVMRegressor <- function(object, X, ...) {
  X <- as.matrix(X)
  if (object$fit_intercept == TRUE) {
    X <- cbind(X, 1)
  }
  if (object$kernel == "linear" & object$solver == "primal") {
    KernelX <- X
  } else {
    KernelX <- kernel_function(X, object$X,
                               kernel.type = object$kernel,
                               gamma = object$gamma,
                               degree = object$degree,
                               coef0 = object$coef0)
  }
  pred <- KernelX %*% object$coef
  return(pred)
}

#' K-Fold Cross Validation with Noisy 
#'
#' \code{cross_validation_noisy} function use noisy data for training,
#' then calculates the average and standard deviation of your metric using clean samples
#'
#' @param model your model.
#' @param X,y dataset and label.
#' @param y_noisy label with label noise.
#' @param K number of folds.
#' @param metrics this parameter receive a metric function.
#' @param predict_func this parameter receive a function for predict.
#' @param ... additional parameters for your model.
#' @return return a metric matrix
#' @export
cross_validation_noisy <- function(model, X, y, y_noisy, K = 5, metrics, ...) {
  X <- as.matrix(X)
  y <- as.matrix(y)
  y_noisy <- as.matrix(y_noisy)
  if (is.list(metrics) == F) {
    metrics <- list(metrics)
    names(metrics) <- paste("metric", length(metrics), sep = "")
  }
  n <- nrow(X)
  num_metric <- length(metrics)
  metric_mat <- matrix(0, num_metric, K)
  index <- sort(rep(1:K, length.out = n))
  for (i in 1:K) {
    idx <- which(index == i)
    X_test <- X[idx, ]
    y_test <- y[idx]
    if(K == 1) {
      X_train <- X_test
      y_train <- y_test
    }else{
      X_train <- X[-idx, ]
      y_train <- y_noisy[-idx]
    }
    model_res <- do.call("model", list("X" = X_train, "y" = y_train, ...))
    y_test_hat <- predict.SVMClassifier(model_res, X_test, ...)
    for (j in 1:num_metric) {
      metric_mat[j, i] <- metrics[[j]](y_test, y_test_hat)
    }
  }
  return(metric_mat)
}

