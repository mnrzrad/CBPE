logReg <- function(X, Y, max_iter = 100, tol = 1e-10) {
  # X <- scale(X)
  X <- cbind(1, X)
  X <- as.matrix(X)

  n <- nrow(X)
  p <- ncol(X)
  beta <- as.matrix(rep(0, p))
  converged <- FALSE

  for (iter in 1:max_iter) {
    eta <- X %*% beta
    pi <- 1 / (1 + exp(-eta))
    W <- diag(as.vector(pi * (1 - pi)))
    z <- eta + (Y - pi) / (pi * (1 - pi))

    beta_new <- solve(t(X) %*% W %*% X, t(X) %*% W %*% z)

    if (max(abs(beta_new - beta)) < tol) {
      converged <- TRUE
      break
    }

    # if (sum((beta_new - beta)^2) < tol^2) {
    #   converged <- TRUE
    #   break
    # }


    beta <- beta_new
  }

  if (iter == max_iter) {
    warning("Logistic regression did not converge.")
  }

  return(list(beta = beta, converged = converged, iterations = iter, W = W, z = z))
}


