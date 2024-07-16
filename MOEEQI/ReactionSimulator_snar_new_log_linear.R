
library(deSolve)

ReactionSimulator_snar_new_log_linear <- function(x, noise_slope, noise_level) {
  
  
  tau <- x[1]
  equiv_pldn <- x[2]
  conc_dfnb <- x[3]
  temperature <- x[4]
  
  C_i <- numeric(5)
  C_i[1] <- conc_dfnb
  C_i[2] <- equiv_pldn * conc_dfnb
  temp <- temperature
  
  
  # Flowrate and residence time
  V <- 3  # mL
  q_tot <- V / tau
  
  
  fun <- function(t, C, parms) {
    
    R <- 8.314 / 1000  
    T_ref <- 140 + 273.71  
    Temp <- parms[[1]] + 273.71  
    
    k <- function(k_ref, E_a, Temp) {
      0.6 * k_ref * exp(-E_a / R * (1 / Temp - 1 / T_ref))
    }
    k_a <- k(57.9, 33.3, Temp)
    k_b <- k(2.70, 35.3, Temp)
    k_c <- k(0.865, 38.9, Temp)
    k_d <- k(1.63, 44.8, Temp)
    
    
    # Reaction Rates
    r <- numeric(5)
    for (i in c(1, 2)) {  
      if (C[i] < 1e-6 * C_i[i]) {
        C[i] <- 0
      }
    }
    r[1] <- -(k_a + k_b) * C[1] * C[2]
    r[2] <- -(k_a + k_b) * C[1] * C[2] - k_c * C[2] * C[3] - k_d * C[2] * C[4]
    r[3] <- k_a * C[1] * C[2] - k_c * C[2] * C[3]
    r[4] <- k_b * C[1] * C[2] - k_d * C[2] * C[4]
    r[5] <- k_c * C[2] * C[3] + k_d * C[2] * C[4]
    
    # Deltas
    dcdtau <- r
    return(list(dcdtau))
  }
  
  # Integrate
  res <- ode(y = C_i, times = c(0, tau), func = fun,  parms = list(temp))
  
  # Model predictions
  C_final <- res[2, 2:6]
  
  # # Add noises in product concentration, not use!
  # noise_term <- rnorm(1, mean = 0, sd = noise_level)
  # C_final[3] <- C_final[3] + C_final[3] * noise_term
  
  
  # Calculate STY and E-factor
  M <- c(159.09, 87.12, 226.21, 226.21, 293.32)  # molecular weights (g/mol)
  sty <- M[3] * C_final[3] / (tau/60)  # unit kg m^-3 h^-1
  mass_in <- C_i[1] * M[1] * (3/1000) + C_i[2] * M[2] * (3/1000)
  mass_prod <- C_final[3]  * M[3] * (3/1000)
  e_factor <- (mass_in - mass_prod) / mass_prod
  
  
  # Add noises in sty and e_factor (homogeneous)
  random_noise <- rnorm(2, mean = 0, sd = 1)
  sty <- sty + (sty ^ noise_slope * 10^ noise_level) * random_noise[1] 
  e_factor <- e_factor + (e_factor ^ noise_slope * 10^noise_level) * random_noise[2] 

  # sty <- pmax(0,sty)
  # e_factor <- pmax(0,e_factor)

  # # # Add noises in sty and e_factor (heterogeneous, linear assump)
  # sd_sty <- sty/13000
  # sd_e_factor <- e_factor/3.5
  # 
  # random_noise_sty <- rnorm(1, mean = 0, sd = sd_sty)
  # random_noise_e_factor <- rnorm(1, mean = 0, sd = sd_e_factor)
  # 
  # sty <- sty + sty * random_noise_sty * noise_level
  # e_factor <- e_factor + e_factor * random_noise_e_factor * noise_level

  
  # # # Add noises in sty and e_factor (heterogeneous, log assump)
  # sd_sty <- log(sty)
  # sd_e_factor <- log(e_factor)
  # 
  # random_noise_sty <- rnorm(1, mean = 0, sd = exp(sd_sty-log(13000)))
  # random_noise_e_factor <- rnorm(1, mean = 0, sd = exp(sd_e_factor-log(3.5)))
  # 
  # sty <- sty + sty * random_noise_sty * noise_level
  # e_factor <- e_factor + e_factor * random_noise_e_factor * noise_level
  

  # # # # Add noises in sty and e_factor (heterogeneous, func assump)
  # sty_range <- 13000
  # e_factor_range <- 3.5
  # 
  # if(sty < 1300 || sty > 12000){
  #   sty_sd <- 2}
  # else{
  #   sty_sd <- 0.1}
  # 
  # 
  # if(e_factor < 0.35 || e_factor > 3.15){
  #   e_factor_sd <- 2}
  #   else{
  #     e_factor_sd <- 0.1}
  # 
  # random_noise_sty <- rnorm(1, mean = 0, sd = sty_sd)
  # sty <- sty + sty * random_noise_sty * noise_level
  # random_noise_e_factor <- rnorm(1, mean = 0, sd = e_factor_sd)
  # e_factor <- e_factor + e_factor * random_noise_e_factor * noise_level



  #############
  
  results <- list(sty = sty, e_factor = e_factor)
  results <- lapply(results, function(x) unname(x))

  return(results)
  

}



# To test this function:
# ReactionSimulator_snar_new(c(0.8,5,0.3,105),0.05)
