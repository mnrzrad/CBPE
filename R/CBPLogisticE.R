#' Correlation-based Estimator for Logistic Regression Models
#'
#' This function computes the correlation-based estimator for logistic regression models.
#'
#' @param X A numeric matrix of predictors where rows represent observations and columns represent variables.
#' @param y A numeric vector of binary outcomes (0 or 1).
#' @param lambda A regularization parameter.
#' @param max_iter An integer specifying the maximum number of iterations for the logistic regression algorithm. Default is 100.
#' @param tol A numeric value specifying the convergence tolerance for the logistic regression algorithm. Default is 1e-10.
#'
#' @details
#' The correlation-based penalized logistic estimator is calculated as:
#' \deqn{
#' \hat{\beta} = \text{argmin}\left\{ \sum_{i=1}^n \left( y_i \ln(\pi_i) + (1 - y_i) \ln(1 - \pi_i) \right) + \lambda \sum_{i=1}^{p-1} \sum_{j>i} \left( \frac{(\beta_i - \beta_j)^2}{1 - \rho_{ij}} + \frac{(\beta_i + \beta_j)^2}{1 + \rho_{ij}} \right) \right\}
#' }
#' where \eqn{\pi_i = \text{Pr}(y_i = 1|\mathbf{x}_i)} and \eqn{\rho_{ij}} denotes the (empirical) correlation between the \eqn{i}th and the \eqn{j}th predictor.
#'
#' @return A numeric vector of the estimated coefficients for the specified model.
#'
#' @references
#' Algamal, Z. Y., & Lee, M. H. (2015). Penalized logistic regression with the adaptive LASSO for gene selection in high-dimensional cancer classification. Expert Systems with Applications, 42(23), 9326-9332. \url{https://doi.org/10.1016/j.eswa.2015.08.016}
#'
#' @examples
#' set.seed(42)
#' n <- 100
#' p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(0.5, -1, 2, 5)
#' y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta_true)))
#' lambda <- 0.1
#'
#' result <- CBPLogisticE(X, y, lambda)
#' print(result)
#'
#' @importFrom stats cor
#'
#' @export
CBPLogisticE <- function(X, y, lambda, max_iter = 100, tol = 1e-6) {
  p <- ncol(X)

  # Calculate the correlation matrix
  rho <- stats::cor(X)

  # Define the weight matrix M
  M <- matrix(0, nrow = p, ncol = p)
  for (i in 1:p) {
    M[i, i] <- 2 * sum(1 / (1 - rho[i, -i]^2))
    for (j in 1:p) {
      if (i != j) {
        M[i, j] <- -2 * rho[i, j] / (1 - rho[i, j]^2)
      }
    }
  }

  # Fit logistic regression model using the custom function
  model_logistic <- logReg(X, y, max_iter, tol)
  W <- model_logistic$W
  z <- model_logistic$z

  # Logistic Regression Correlation-based Estimator
  est <- solve(t(X) %*% W %*% X + lambda * M) %*% t(X) %*% W %*% z

  return(as.vector(est))
}
