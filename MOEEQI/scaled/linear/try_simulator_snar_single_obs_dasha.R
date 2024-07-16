# Please see https://github.com/StatsDasha/MO-E-EQI/ for instructions on installing the MOEEQI package
library(MOEEQI)
library(reticulate)
library(DiceKriging) # This is for stochastic Kriging model
library(prodlim)
library(tidyr)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("../../ReactionSimulator_snar_new.R")
source("../../InitialDesign.R")

coeff <- -1 / 1
MC_sample_size <- 1

initial_samples <- 20
#The only executed combinations are Nsteps=20 & prop_aggressive =1; Nsteps=20 & prop_aggressive =.5; Nsteps=40 & prop_aggressive =.5.
Nsteps <- 20
prop_aggressive <- 1#.5#

execution_time <- NULL
for (beta  in c(.6, .7, .8, .9)) {
  for (noise_level in c(0.01, 0.05, 0.1, 0.2)) {
    iter <- 20
    
    ranges <- matrix(c(0.5, 2, 1, 5, 0.1, 0.5, 30, 120), 4, 2, byrow = T)
    output_all <- input_all <- list()
    
    ####################################################################################################
    
    for (n in 1:iter) {
      # Generate initial design
      input_data <- InitialDesign(20,4,c(0.5,2.0),c(1.0,5.0),c(0.1,0.5),c(30,120))
      
      
      output_STY <- output_E.factor <- data.frame(matrix(ncol = MC_sample_size, nrow = initial_samples))
      
      for (m in 1:initial_samples) {
        res_sty <- res_e_factor <- NULL
        
        for (i in 1:MC_sample_size) {
          results <- ReactionSimulator_snar_new(as.vector(unlist(input_data[m, ])), noise_level)
          res_sty <- c(res_sty, results$sty)
          res_e_factor <- c(res_e_factor, results$e_factor)
        }
        
        output_STY[m, ] <- coeff * res_sty
        output_E.factor[m, ] <- res_e_factor
      }
      
      
      #Scale the inputs to [-1,1]
      input_data_scaled <-  Scale(
        input_data,
        a = ranges[, 1],
        b = ranges[, 2],
        u = 1,
        l = 0
      )
      
      colnames(input_data_scaled) <-
        c('res_time', 'equiv', 'conc', 'temp')
      #Scale the outputs
      
      mean_STY <- apply(output_STY, MARGIN = 1, FUN = mean)
      mean_E.factor <- apply(output_E.factor, MARGIN = 1, FUN = mean)
      
      #Scale the outputs
      sty_mean <- mean(mean_STY)
      sty_sd <-  sd(mean_STY)
      mean_STY_scaled <- (mean_STY - sty_mean) / sty_sd
      
      E.factor_mean <- mean(mean_E.factor)
      E.factor_sd <-  sd(mean_E.factor)
      mean_E.factor_scaled <- (mean_E.factor - E.factor_mean) / E.factor_sd
      
      noise_sd <-   noise_level * data.frame(tau1 =
                                               (mean_STY) / coeff / sty_sd,
                                             tau2 =
                                               mean_E.factor / E.factor_sd)
      noise_sd <- tau_new_func(MC_sample_size, noise_sd, 1)
      
      # noise.var is the only way to have a stochastic emulator
      noise.var <- list(tau1 = noise_sd$tau1 ^ 2, tau2 = noise_sd$tau2 ^ 2)
      
      ####################################################################################################
      # Fit emulators
      model_f1 <-
        km(
          formula =  ~ 1,
          design = input_data_scaled,
          response = mean_STY_scaled,
          covtype = "matern5_2",
          noise.var = noise.var$tau1
        )
      model_f2 <-
        km(
          formula =  ~ 1,
          design = input_data_scaled,
          response = mean_E.factor_scaled,
          covtype = "matern5_2",
          noise.var = noise.var$tau2
        )
      
      
      # select new points to calculate EQI at. Covers all the points in the ranges.
      newdata <- crossing(
        res_time = seq(0.50, 2.00, by = 0.1),
        equiv = seq(1.0, 5.0, by = 0.2),
        conc = seq(0.1, 0.5, by = 0.05),
        temp = seq(30, 120, by = 5)
      ) %>% as.matrix
      
      colnames(newdata) <-
        c('res_time', 'equiv', 'conc', 'temp')
      
      newdata_scaled <- Scale(
        newdata,
        a = ranges[, 1],
        b = ranges[, 2],
        u = 1,
        l = 0
      )
      
      n_sample <- length(newdata_scaled[, 1])
      
      # The next line checks which of the current design points exists in the newdata. This is nessessary for tau_new function
      des_rep <- design_repetitions(newdata_scaled, input_data_scaled)
      
      # The next line calculates the default tau_new if there were no repetitions
      tau_new <-   noise_level * data.frame(
        tau1 =
          (
            predict.km(model_f1, newdata_scaled, type = "UK")$mean + sty_mean / sty_sd
          ) / coeff,
        tau2 =
          predict.km(model_f2, newdata_scaled, type =
                       "UK")$mean + E.factor_mean / E.factor_sd
      )
      
      tau_new <- tau_new_func(MC_sample_size, tau_new, 1)
      
      # Update the design locations that were repeated
      if (sum(des_rep) != 0) {
        tau_new[des_rep[, 2], ] <-
          cbind(tau1 = sqrt(tau_eq_sqrd(
            noise.var$tau1[des_rep[, 1]], (tau_new[des_rep[, 2], ]$tau1 ^ 2)
          )),
          tau2 = sqrt(tau_eq_sqrd(
            noise.var$tau2[des_rep[, 1]], (tau_new[des_rep[, 2], ]$tau2 ^ 2)
          )))
      }
      
      
      # #Add constraint info for objectives
      ConstraintInfo <- NULL
      
      reps <- NULL
      improv <- NULL
      count <- 1 + dim(input_data_scaled)[1]
      output_all[[1]] <- cbind(mean_STY, mean_E.factor)
      input_all[[1]] <- input_data_scaled
      
      # This is EQI optimisation loop
      for (i in 1:Nsteps) {
        #calculate EQI metric. Note that other outputs are Pareto front, design and quantile sd
        
        if (i <= Nsteps * prop_aggressive) {
          EQI_newdata <-
            mult_EQI(
              newdata_scaled,
              input_data_scaled,
              model_f1,
              model_f2,
              beta,
              tau_new,
              ConstraintInfo = NULL,
              Option = 'NegLogEQI',
              aggressive = T
            )
        } else{
          EQI_newdata <-
            mult_EQI(
              newdata_scaled,
              input_data_scaled,
              model_f1,
              model_f2,
              beta,
              tau_new,
              ConstraintInfo = NULL,
              Option = 'NegLogEQI',
              aggressive = F
            )
        }
        
        best_X <- which.min(EQI_newdata$metric)
        current_improvement <- EQI_newdata$metric[best_X]
        #find the values of the best design points
        impr_x <- newdata_scaled[best_X, ]
        repetition <- row.match(impr_x, input_data_scaled)
        
        input_data_scaled <- rbind(input_data_scaled, impr_x)
        
        
        #########################################
        #### Run experiments ####
        
        
        new_STY <- new_E.factor <- NULL
        for (j in 1:MC_sample_size) {
          new_data <- ReactionSimulator_snar_new(Scale(
            matrix(impr_x, 1, 4),
            a = rep(0, 4),
            b = rep(1, 4),
            l = ranges[, 1],
            u = ranges[, 2]
          ),
          noise_level)
          new_STY <- c(new_STY, new_data$sty)
          new_E.factor <- c(new_E.factor, new_data$e_factor)
        }
        output_STY <- rbind(output_STY, coeff * new_STY)
        output_E.factor <- rbind(output_E.factor, new_E.factor)
        
        new_STY_mean <- mean(coeff * new_STY)
        new_E.factor_mean <- mean(new_E.factor)
        
        tau_at_best_X <- tau_new_func(
          MC_sample_size,
          noise_level * c(
            new_STY_mean / coeff / sty_sd,
            new_E.factor_mean / E.factor_sd
          ),
          1
        )
        # tau_at_best_X <- tau_new_func(MC_sample_size,
        #                               c(sd(new_STY)/sty_sd,
        #                                 sd(new_E.factor)/E.factor_sd),
        #                               1)
        
        if (is.na(repetition)) {
          mean_STY_scaled <- c(mean_STY_scaled, (new_STY_mean - sty_mean) / sty_sd)
          mean_E.factor_scaled <- c(mean_E.factor_scaled,
                                    (new_E.factor_mean - E.factor_mean) / E.factor_sd)
          
          # Update the observations noise
          noise.var <- data.frame(
            tau1 = c(noise.var$tau1, tau_at_best_X$tau1 ^ 2),
            tau2 = c(noise.var$tau2, tau_at_best_X$tau2 ^ 2)
          )
          
          
          
        } else{
          # Update observations
          input_data_scaled <- input_data_scaled[-dim(input_data_scaled)[1], ]
          mean_STY_scaled[repetition] <-
            mean_obs((new_STY_mean - sty_mean) / sty_sd,
                     mean_STY_scaled[repetition],
                     tau_at_best_X$tau1 ^ 2,
                     noise.var$tau1[repetition]
            )
          mean_E.factor_scaled[repetition] <-
            mean_obs((new_E.factor_mean - E.factor_mean) / E.factor_sd,
                     # eq 11
                     mean_E.factor_scaled[repetition],
                     tau_at_best_X$tau2 ^ 2,
                     noise.var$tau2[repetition]
            )
          
          # Update the observations noise
          noise.var$tau1[repetition] <-
            tau_eq_sqrd(noise.var$tau1[repetition], tau_at_best_X$tau1 ^ 2) # eq 13
          noise.var$tau2[repetition] <-
            tau_eq_sqrd(noise.var$tau2[repetition], tau_at_best_X$tau2 ^ 2)
          
          # record the repetitions
          reps <- rbind(reps, c(repetition, count))
        }
        
        model_f1 <-
          km(
            formula =  ~ 1,
            design = input_data_scaled,
            response = mean_STY_scaled,
            covtype = "matern5_2",
            noise.var = noise.var$tau1
          )
        model_f2 <-
          km(
            formula =  ~ 1,
            design = input_data_scaled,
            response = mean_E.factor_scaled,
            covtype = "matern5_2",
            noise.var = noise.var$tau2
          )
        des_rep <- design_repetitions(newdata_scaled, input_data_scaled)
        # The next line calculates the default tau_new if there were no repetitions
        tau_new <-   noise_level * data.frame(
          tau1 =
            (
              predict.km(model_f1, newdata_scaled, type = "UK")$mean + sty_mean / sty_sd
            ) / coeff,
          tau2 =
            predict.km(model_f2, newdata_scaled, type =
                         "UK")$mean + E.factor_mean / E.factor_sd
        )
        tau_new <- tau_new_func(MC_sample_size, tau_new, 1)
        
        # Update the design locations that were repeated using the current levels of observed noise and new expected funure noise
        if (sum(des_rep) != 0) {
          tau_new[des_rep[, 2], ] <-
            cbind(tau1 = sqrt(tau_eq_sqrd(
              noise.var$tau1[des_rep[, 1]], (tau_new[des_rep[, 2], ]$tau1 ^ 2)
            )),
            tau2 = sqrt(tau_eq_sqrd(
              noise.var$tau2[des_rep[, 1]], (tau_new[des_rep[, 2], ]$tau2 ^ 2)
            )))
        }
        
        ConstraintInfo$y <- cbind(mean_STY_scaled, mean_E.factor_scaled)  # output-already use harmonic mean
        count <- count + 1
        improv <- c(improv, current_improvement)
        
        output_all[[1 + i]] <- cbind(
          mean_STY_scaled * sty_sd + sty_mean,
          mean_E.factor_scaled * E.factor_sd + E.factor_mean
        )
        input_all[[1 + i]] <- input_data_scaled
        
      }
      
      input_data <- Scale(
        input_data_scaled,
        a = rep(0, 4),
        b = rep(1, 4),
        l = ranges[, 1],
        u = ranges[, 2]
      )
      newdata <- Scale(
        newdata_scaled,
        a = rep(0, 4),
        b = rep(1, 4),
        l = ranges[, 1],
        u = ranges[, 2]
      )
      
      noise.var <-  list(tau1 = noise.var$tau1 * sty_sd ^ 2,
                         tau2 = noise.var$tau2 * E.factor_sd ^ 2)
      
      if (Nsteps == 20 & prop_aggressive == 1) {
        output_dir <- "aggressive"
      } else if (Nsteps == 20 & prop_aggressive == .5) {
        output_dir <- "non_aggressive_10_10"
      } else if (Nsteps == 40 & prop_aggressive == .5) {
        output_dir <- "non_aggressive_20_20"
      } else{
        stop("Error: not compatible Nsteps and proportion of aggressive steps")
      }
      if (!dir.exists(output_dir)) {
        dir.create(output_dir)
      }
      # save results csv
      res_folder <- paste('results_single_noise_',
                          noise_level,
                          '_beta_',
                          beta,
                          sep = "")
      if (!dir.exists(paste(output_dir, '/', res_folder, sep = ""))) {
        dir.create(paste(output_dir, '/', res_folder, sep = ""))
      }
      # save results csv
      csv_name <- paste('repeat_', n, '.csv', sep = "")
      full_csv_path <- paste(
        output_dir,
        '/',
        res_folder,
        '/',
        csv_name,
        sep = ""
      )
      
      results <- cbind(
        input_data,-(mean_STY_scaled * sty_sd + sty_mean),
        mean_E.factor_scaled * E.factor_sd + E.factor_mean
      )
      # results <- cbind(input_data,-y1,y2)
      # colnames(results)[5:10] <- c('sty_1','sty_2','sty_3','e_factor_1','e_factor_2','e_factor_3')
      colnames(results)[5:6] <- c('sty', 'e_factor')
      # write.csv(results, file="results/test1.csv", row.names = FALSE)
      write.csv(results, file = full_csv_path, row.names = FALSE)
      
      # save r dataT
      data_name <- paste('data', n, '.RData', sep = "")
      full_data_path <- paste(
        output_dir,
        '/',
        res_folder,
        '/',
        data_name,
        sep = ""
      )
      
      save(
        input_data,
        output_STY,
        output_E.factor,
        noise.var,
        beta,
        MC_sample_size,
        model_f1,
        model_f2,
        current_improvement,
        reps,
        output_all,
        input_all,
        file = full_data_path
      )

    }
  }
}

