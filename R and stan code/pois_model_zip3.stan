data {
  int<lower=0> N;
  int<lower=0> n_year;
  int<lower=0> n_lake;
  int<lower=0> n_post;
  int<lower=0> n_trt;
  int<lower=0> n_stocked;
  // observation-level predictors
  int<lower=0, upper=1> post[N];
  int<lower=0, upper=1> trt[N];
  int<lower=0, upper=1> stocked[N];
  vector[N] survey_gdd;
  vector[N] survey_secchi;
  vector[N] offset;
  // lake and year indicator
  int<lower=0, upper=n_lake> lake[N];
  int<lower=0, upper=n_year> year[N];
  // response variable
  int<lower=0> y[N];
  // lake-level predictors
  vector[n_lake] lake_area;
  vector[n_lake] gdd;
  vector[n_lake] secchi;
}
parameters {
  real<lower=0> sigma_year;
  real<lower=0> sigma_lake;
  
  real<lower=0, upper=1> theta;
  
  real b_0;
  real b_post;
  real b_trt;
  real b_stocked;
  real b_post_trt;
  real b_survey_gdd;
  real b_survey_secchi;
  
  real b_lake_area;
  real b_gdd;
  real b_secchi;
  
  vector[n_year] b_year;
  
  vector[n_lake] b_hat;
}
model {
  vector[N] lambda;
  //vector[N] theta;                      // probability of zero 
  vector[n_lake] b_lake_hat;
  
  b_0 ~ normal(0, 100);
  b_post ~ normal(0, 100);
  b_trt ~ normal(0, 100);
  b_stocked ~ normal(0, 100);
  b_post_trt ~ normal(0, 100);
  b_survey_gdd ~ normal(0, 100);
  b_survey_secchi ~ normal(0, 100);
  
  
  b_year ~ normal(0, sigma_year);
  
  b_lake_area ~ normal(0, 100);
  b_gdd ~ normal(0, 100);
  b_secchi ~ normal(0, 100);
  
  for (j in 1:n_lake)
    b_lake_hat[j] = b_lake_area*lake_area[j] + b_gdd*gdd[j] + b_secchi*secchi[j];
  
  b_hat ~ normal(b_lake_hat, sigma_lake);
  
  
  for (i in 1:N){
    lambda[i] = offset[i] + b_hat[lake[i]] + b_0 + b_post*post[i]
    + b_trt*trt[i] + b_stocked*stocked[i] + b_post_trt*post[i]*trt[i] + b_year[year[i]]  
    + b_survey_gdd*survey_gdd[i] + b_survey_secchi*survey_secchi[i];
    
  }
  
  // likelihood
  for (n in 1:N) {
    if (y[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(1 | theta),
                            bernoulli_lpmf(0 | theta)
                            + poisson_lpmf(y[n] | exp(lambda[n])));
      else
        target+= bernoulli_lpmf(0 | theta)
        + poisson_lpmf(y[n] | exp(lambda[n]));
  }
}
