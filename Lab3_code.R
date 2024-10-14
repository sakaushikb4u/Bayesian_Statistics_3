# ===================================================================
# Init Library
# ===================================================================
knitr::opts_chunk$set(echo = TRUE)
library(ggplot2)
library(gridExtra)
library(readxl)
library(LaplacesDemon)
library(MASS)
library(matrixStats)
library(mvtnorm)
library(rstan)
library(loo)
library(coda)
library(BayesLogit)
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')

# 1. Gibbs sampling for the logistic regression

#--------------------------------------------------------------------
# Init code for question 1
#--------------------------------------------------------------------
rm(list = ls())


## 1.a Implement a Gibbs sampler that simulates from the joint posterior

#--------------------------------------------------------------------
# load the data
#--------------------------------------------------------------------
WomenAtWork = read.table("WomenAtWork.dat", header = TRUE)
y = WomenAtWork$Work
X = WomenAtWork[,2:8]
X = as.matrix(X)
Xnames <- colnames(X)

tau <- 3

#get dimensions
n <- nrow(X)
p <- ncol(X)

set.seed(12345)

# Initialize parameters
beta <- rep(0, p)

# Number of iterations
n_iter <- 3000

burn_in = 200

beta_samples <- matrix(0, ncol = p, nrow = n_iter)
omega <- rep(1, n)

# B = tao^2 * I 
B <- diag(tau^2, p)

# Gibbs Sampling
for (iter in 1:n_iter) {
    # draw samples using rpg function
    omega <- rpg(n, 1, X %*% beta)
    
    # Update beta according to the formula mentioned previously
    V_beta <- solve(t(X) %*% diag(omega) %*% X + B)
    # b = 0 and k = (y - 0.5)
    m_beta <- V_beta %*% t(X) %*% (y - 0.5)
    beta <- mvrnorm(1, mu = m_beta, Sigma = V_beta)
    
    # Store samples
    beta_samples[iter, ] <- beta
  }

# Remove burn-in samples
beta_samples <- beta_samples[-(1:burn_in), ]

# Convert samples to mcmc object
mcmc_samples <- as.mcmc(beta_samples)
summary_stats <- summary(mcmc_samples)

# print out the summary and the time-series standard error
print(summary_stats)
print(summary_stats$statistics[,"Time-series SE"])

# Plot trajectories of the sampled Markov chains
par(mfrow = c(2, 2))
for (j in 1:ncol(beta_samples)) {
  plot(beta_samples[, j], type = "l", main = Xnames[j],
       xlab = "Iteration", ylab = "Sample value")
}

## 1.b compute a 90% equal tail credible interval

# Define the predictor vector x
# a husband with an income of 22
# 12 years of education
# 7 years of exp erience,
# a 38-year-old woman,
# one child (3 years old)
x_new <- c(1, 22, 12, 7, 38, 1, 0)

probabilities <- apply(beta_samples, 1, function(beta) {
  exp(sum(beta * x_new)) / (1 + exp(sum(beta * x_new)))
})

# Compute the 90% equal tail credible interval for the probabilities
credible_interval <- quantile(probabilities, probs = c(0.05, 0.95))
print(credible_interval)

# 2 Metropolis Random Walk for Poisson regression

#--------------------------------------------------------------------
# Init code for question 2
#--------------------------------------------------------------------
rm(list = ls())
ebay_data <- read.table("eBayNumberOfBidderData_2024.dat", header = T)

## 2.a Obtain the maximum likelihood estimator of $\beta$ in the Poisson regression model for the eBay data

#--------------------------------------------------------------------
# code for question 2.a
#--------------------------------------------------------------------
# remove covariate const (2nd column)
data_noconst <- ebay_data[,-2]
glm_model <- glm(nBids ~ ., family = poisson(link = "log"), data = data_noconst)
summary(glm_model)

## 2.b Bayesian analysis of the Poisson regression

#--------------------------------------------------------------------
# code for question 2.b
#--------------------------------------------------------------------
logPost <- function(beta,X=covariates,Y=response){
  linPred <- X%*%beta;
  logLik <- sum(Y * linPred - exp(linPred));  
  logPrior <- dmvnorm(t(beta), 
                      mean = matrix(0,nrow=ncol(X)), 
                      sigma = 100 * (solve(t(X) %*% X)), 
                      log=TRUE);
  return(logLik + logPrior)
}

# get response and covariates from original data
response <- as.matrix(ebay_data$nBids)
covariates <- as.matrix(ebay_data[,2:10])

# initial values
init_val <- matrix(1,nrow=9);

# optimize the log posterior
OptimRes <- optim(par = init_val,
                  fn = logPost,                                    
                  method=c("BFGS"),
                  control=list(fnscale=-1),
                  hessian=TRUE)
# set values to print out
posterior_mode  <- OptimRes$par
beta_jacobian <- -OptimRes$hessian
beta_inverse_jacobian <- solve(beta_jacobian) 

#--------------------------------------------------------------------
# print values
#--------------------------------------------------------------------
rownames(posterior_mode) <- colnames(covariates)
print('The posterior beta is:')
print(t(posterior_mode))
print('The glm_model coefficients is:')
print(glm_model$coefficients)
print('The beta_inverse_jacobian is:')
print(beta_inverse_jacobian)

#--------------------------------------------------------------------
# code for question 2.c
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#  metropolis function
#--------------------------------------------------------------------
metropolis_fn <- function(tgt_density, c, theta_i, sigma_proposal, steps, X, Y){
  # init seed
  set.seed(12345)

  result <- matrix(t(theta_i), ncol=9)
  accepted_count <- 0

  for(i in 1:steps){
    # generate sample from proposal
    theta_p <- rmvnorm(n = 1, mean = as.vector(theta_i), sigma = c * sigma_proposal)
    
    # calculate acceptance ratio
    acceptance_ratio <- tgt_density(as.vector(theta_p), X, Y) - 
                        tgt_density(as.vector(theta_i), X, Y)

    # apply exp to acceptance ratio(since original one is in log)
    acceptance_ratio <- exp(acceptance_ratio)

    # calculate alpha
    alpha <- min(1, acceptance_ratio)

    # draw from  uniform distribution
    u <- runif(1)

    # accept or reject
    if (u < alpha){
      theta_i <- theta_p
      accepted_count <- accepted_count + 1      
      result <- rbind(result, theta_i)      
    }else{
      result <- rbind(result, as.vector(theta_i))
    }    
  }
  acceptance <- accepted_count / steps
  return(list(result = result, acceptance = acceptance))    
}

#--------------------------------------------------------------------
# function call to run metropolis
#--------------------------------------------------------------------
sigma_proposal <- beta_inverse_jacobian
theta_init <- matrix(rep(0.5,9), ncol=9)

# run metropolis
metropolis_val <- metropolis_fn(tgt_density = logPost, 
                                c = 0.5,
                                theta_i = theta_init,
                                sigma_proposal = sigma_proposal,
                                steps = 1000,
                                X  = covariates,
                                Y = response)

metropolis_result <- metropolis_val$result
colnames(metropolis_result) <- rownames(posterior_mode)
metropolis_result_mean <- apply(metropolis_result, 2, mean)
names(metropolis_result_mean) <- rownames(posterior_mode)
metropolis_accept <- metropolis_val$acceptance


print('The metropolis_result_mean is:')
print(metropolis_result_mean)
print('The metropolis_accept is:')
print(metropolis_accept)
print('shape of metropolis_result is:')
print(dim(metropolis_result))


#--------------------------------------------------------------------
# plot
#--------------------------------------------------------------------
par(mfrow = c(3,3))
for(i in 1:9){
  plot((metropolis_result[,i]), type="l", ylab= colnames(metropolis_result)[i],xlab = "steps")
}


#--------------------------------------------------------------------
# code for question 2.d
#--------------------------------------------------------------------
new_data <- c(1, 1, 0, 1, 0, 1, 0, 1.2, 0.8)
result2 <- data.frame(matrix(0, nrow = nrow(metropolis_result), ncol = 9))

for(i in 1:nrow(metropolis_result)){
  lambda <- exp(new_data %*% as.numeric(metropolis_result[i,]))
  result2[i,] <- rpois(1, lambda)
}

#--------------------------------------------------------------------
# plot
#--------------------------------------------------------------------
print("Probability of no bidders is:")
print(mean(result2 == 0))
hist(result2[,1], main = "Predictive distribution", xlab = "Bidders",breaks = 30, freq = FALSE)
lines(density(result2[,1]),lwd=1,col="blue")

#--------------------------------------------------------------------
# Code for question 3
#--------------------------------------------------------------------
rm(list = ls())

#--------------------------------------------------------------------
# code for question 3.a
#--------------------------------------------------------------------
#----------------------------------------------
# Parameters for AR(1) process
#----------------------------------------------
phis <- seq(-0.99, 0.99, by = 0.20)
sigma2 <- 4
mu <- 9
T <- 250

#----------------------------------------------
# Function to simulate AR(1) process
#----------------------------------------------
simulate_ar1 <- function(mu=9, phi, sigma2=4, T=250) {
  # init variables
  n <- length(phis)
  values <- data.frame(0,ncol = n)
  
  for(i in 1:n){
    x = mu
    values[1,i] = x

    for (j in 2:T) {
        x <- mu + phi[i] * (x - mu) + rnorm(1, 0, sqrt(sigma2))
        values[j,i] = x
    }    
  }
  # set column names
  colnames(values) <- paste("phi_",phi)
  return(values)
}

# function call
ar1 <- simulate_ar1(mu = mu, phi = phis, sigma2 = sigma2,T = T)



#----------------------------------------------
# Plot AR(1) process simulated data
#----------------------------------------------
# set layout to 2 x 3
par(mfrow = c(2,3))
for(i in 1:length(phis)){
  plot(ar1[,i], type = 'l', ylab = paste("phi = ", phis[i]), xlab = "T")
}

#--------------------------------------------------------------------
# code for question 3.b
#--------------------------------------------------------------------

model_AR <- simulate_ar1(phi = c(0.3, 0.97))

StanModel = '
data {
  int<lower=0> N; // Number of observations
  vector[N] x;    // x_t
}
parameters {
  real mu;
  real<lower = 0> sigma2;
  real<lower = -1, upper = 1> phi;
}
model {
  mu ~ normal(0,50); 
  sigma2 ~ scaled_inv_chi_square(1,10); 
  phi ~ uniform(-1,1);
  
  for(i in 2:N){
    x[i] ~ normal(mu + phi * (x[i-1] - mu), sqrt(sigma2));
  }
}'


#--------------------------------------------------------------------
# code for question 3.b.i
#--------------------------------------------------------------------
#fit model for phi 0.3
fit_0.3 = stan(model_code = StanModel,
               data = list(x = model_AR$`phi_ 0.3`, N = 250),
               warmup = 1000,
               iter = 2000,
               chains = 4)

#fit model for phi 0.97
fit_0.97 = stan(model_code = StanModel,
               data = list(x = model_AR$`phi_ 0.97`, N = 250),
               warmup = 1000,
               iter = 2000,
               chains = 4)

post_samples_0.3 <- extract(fit_0.3)
post_mean_0.3 <- get_posterior_mean(fit_0.3)
post_samples_0.97 <- extract(fit_0.97)
post_mean_0.97 <- get_posterior_mean(fit_0.97)     

#--------------------------------------------------------------------
# code for question 3.b.ii
#--------------------------------------------------------------------
post_samples_0.3_df <- data.frame(mu = post_samples_0.3$mu, 
                                  sigma2 = post_samples_0.3$sigma2, 
                                  phi = post_samples_0.3$phi)

post_samples_0.97_df <- data.frame(mu = post_samples_0.97$mu, 
                                  sigma2 = post_samples_0.97$sigma2, 
                                  phi = post_samples_0.97$phi)   

CI_0.3 <- sapply(post_samples_0.3_df, function(x) quantile(x, probs = c(0.025, 0.975)))
CI_0.97 <- sapply(post_samples_0.97_df, function(x) quantile(x, probs = c(0.025, 0.975)))


#--------------------------------------------------------------------
# print
#--------------------------------------------------------------------
print("Posterior means when phi = 0.3 :")
print(post_mean_0.3)
print("Posterior means when phi = 0.97 :")
print(post_mean_0.97)

print("Posterior 95% CI when phi=0.3 :")
print(CI_0.3)
print("Posterior 95% CI when phi=0.97 :")
print(CI_0.97)

#--------------------------------------------------------------------
# plot
#--------------------------------------------------------------------
par(mfrow = c(2,1))
plot(y=model_AR[,1],x=c(1:250),type='l',ylab='phi = 0.3',xlab='T')
plot(y=model_AR[,2],x=c(1:250),type='l',ylab='phi = 0.97',xlab='T')

par(mfrow = c(2,2))
plot(post_samples_0.3$mu, type = 'l', ylab="posterior mu",main="phi = 0.3")
plot(post_samples_0.97$mu, type = 'l', ylab="posterior mu",main="phi = 0.97")

plot(post_samples_0.3$sigma2, type = 'l', ylab="posterior sigma2",main="phi = 0.3")
plot(post_samples_0.97$sigma2, type = 'l', ylab="posterior sigma2",main="phi = 0.97")

plot(post_samples_0.3$phi, type = 'l', ylab="posterior phi",main="phi = 0.3")
plot(post_samples_0.97$phi, type = 'l', ylab="posterior phi",main="phi = 0.97")

par(mfrow = c(1,2))

plot(post_samples_0.3$mu, post_samples_0.3$phi, type = 'p',
    xlab = "mu", ylab = "phi", main = "phi = 0.3")
plot(post_samples_0.97$mu, post_samples_0.97$phi, type = 'p', 
    xlab = "mu", ylab = "phi", main = "phi = 0.97")

