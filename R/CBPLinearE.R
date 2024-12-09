#' Correlation-based Estimator for Linear Regression Models
#'
#' This function computes the correlation-based estimator for linear regression models.
#'
#' @param X A numeric matrix of predictors where rows represent observations and columns represent variables.
#' @param y A numeric vector of response variables.
#' @param lambda A regularization parameter.
#'
#' @details
#' The correlation-based penalized linear estimator is calculated as:
#' \deqn{
#' \hat{\beta} =  \text{argmin} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \sum_{i=1}^{p-1} \sum_{j>i} \left( \frac{(\beta_i - \beta_j)^2}{1 - \rho_{ij}} + \frac{(\beta_i + \beta_j)^2}{1 + \rho_{ij}} \right) \right\}
#' }
#' where \eqn{\rho_{ij}} denotes the (empirical) correlation between the \eqn{i}th and the \eqn{j}th predictor.
#'
#' @return A numeric vector of the estimated coefficients for the specified model.
#'
#' @references
#' Tutz, G., Ulbricht, J. (2009). Penalized regression with correlation-based penalty. Stat Comput 19, 239–253.
#'
#' @examples
#' set.seed(42)
#' n <- 100
#' p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(0.5, -1, 2, 5)
#' y <- X %*% beta_true + rnorm(n)
#' lambda <- 0.1
#'
#' result <- CBPLinearE(X, y, lambda = lambda)
#' print(result)
#'
#' @importFrom stats cor
#'
#' @export
CBPLinearE <- function(X, y, lambda) {
  p <- ncol(X)

  rho <- stats::cor(X)

  M <- matrix(0, nrow = p, ncol = p)
  for (i in 1:p) {
    M[i, i] <- 2 * sum(1 / (1 - rho[i, -i]^2))
    for (j in 1:p) {
      if (i != j) {
        M[i, j] <- -2 * rho[i, j] / (1 - rho[i, j]^2)
      }
    }
  }

  est <- solve(t(X) %*% X + lambda * M) %*% t(X) %*% y


  return(as.vector(est))
}

#for test (thanks)
