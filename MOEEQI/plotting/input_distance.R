library(reticulate)
library(tidyverse)
library(MOEEQI)
library(tidyverse)
library(pdist)

use_condaenv("r-reticulate")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("../ReactionSimulator_snar_new.R")  
source("../Scale.R")  
is_scaled <- ''#'_scale'#

ranges <- matrix(c(0.5, 2,1, 5, 0.1, 0.5, 30, 120), 4,2, byrow=T)

# # Find true pareto
# newdata <- crossing(
#   res_time = seq(0.50, 2.00, by = 0.1),
#   equiv = seq(1.0, 5.0, by = 0.2),
#   conc = seq(0.1, 0.5, by = 0.05),
#   temp = seq(30, 120, by = 5)
# ) %>% as.matrix
# 
# colnames(newdata) <-
#   c(
#     'res_time',
#     'equiv',
#     'conc',
#     'temp'
#   )
# 
# res_sty <- res_e_factor <- NULL
# for (i in 1:nrow(newdata)) {
# 
#   results <- ReactionSimulator_snar_new(as.vector(unlist(newdata[i,])),0)
#   res_sty <- c(res_sty, results$sty)
#   res_e_factor <- c(res_e_factor, results$e_factor)
# }
# save(res_sty,res_e_factor, newdata, file="true_pareto.RData")

load("true_pareto.RData")

pareto_true <-
  data.frame(pareto_front(-res_sty, res_e_factor, newdata))[, c(3:6, 1, 2)]
names(pareto_true) <-
  c("res_time", "equiv", "conc", "temp", "STY", "E.Factor")

pareto_true_inputs_scaled <-  Scale(pareto_true[,1:4],a=ranges[,1],b=ranges[,2],u=1,l=0)


MOEEQI_set_up <- 'single'#'mult'#
if(is_scaled == '_scale'){
  folder_path='../scaled/'
}else{
  folder_path='../not scaled/'
}

PDFPath = paste(folder_path,"MOEEQI_",MOEEQI_set_up,"_scaled_inputs_outputs_input_distance all betas.pdf",sep='')
pdf(file=PDFPath, width=23, height=13)


# noise_level <- 0.2

for(noise_level in c(0.01,0.05,0.1,0.2)){
  input_distance_MOEEQI <- array(NA, dim=c(20,21,4))
  input_distance_qNEHVI <- input_distance_TSEMO<- NULL
  for(n in 1:20){
    input_distance_MOEEQI_n <- NULL
    
    csv_name <- paste('../../qNEHVI_2024/TrainingSet1_noise_',noise_level,is_scaled,'/repeat_',n,'.csv', sep = "")
    results_qNEHVI <- read.csv(csv_name, header=T)
    
    csv_name <- paste('../../TSEMO_2024/TrainingSet1_noise_',noise_level,'_scale/repeat_',n,'.csv', sep = "")
    results_TSEMO <- read.csv(csv_name, header=T)
    
    MOEEQI_input_distance_all_betas <- NULL
    
    for(beta in c(.6,.7,.8,.9)){
      load(paste(folder_path,'results_2024_',MOEEQI_set_up,'_noise_',noise_level,'_beta_',beta,'/data',n,'.RData', sep = ""))
      if(is_scaled != '_scale'){
        input_all <- sapply(input_all, Scale, a=ranges[,1],b=ranges[,2],u=1,l=0)
      }
      assign(paste("output_all_beta_",beta,'_noise_level_',noise_level,sep=''),output_all)
      assign(paste("input_all_beta_",beta,'_noise_level_',noise_level,sep=''),input_all)

      pareto_MOEEQI_n <- pareto_front(output_all[[1]][,1], output_all[[1]][,2], input_all[[1]])$X
      
      MOEEQI_input_distance_all_betas <- c(MOEEQI_input_distance_all_betas,mean(apply(as.matrix(pdist::pdist(pareto_MOEEQI_n, pareto_true_inputs_scaled)),1,min)))
    }
    input_distance_MOEEQI_n <- c(input_distance_MOEEQI_n,MOEEQI_input_distance_all_betas)

    
    pareto_qNEHVI_n <- pareto_front(-results_qNEHVI[1:20,]$sty,results_qNEHVI[1:20,]$e_factor,results_qNEHVI[1:20,1:4])$X
    pareto_qNEHVI_n <- Scale(pareto_qNEHVI_n,a=ranges[,1],b=ranges[,2],u=1,l=0)
    input_distance_qNEHVI_n <- mean(apply(as.matrix(pdist::pdist(pareto_qNEHVI_n,pareto_true_inputs_scaled)),
                                         1,min)
    )

    pareto_TSEMO_n <- pareto_front(-results_TSEMO[1:20,]$sty,results_TSEMO[1:20,]$e_factor,results_TSEMO[1:20,1:4])$X
    pareto_TSEMO_n <- Scale(pareto_TSEMO_n,a=ranges[,1],b=ranges[,2],u=1,l=0)
    input_distance_TSEMO_n <- mean(apply(as.matrix(pdist::pdist(pareto_TSEMO_n,pareto_true_inputs_scaled)),
                                                                  1,min)
    )
    for(i in 21:40){
      MOEEQI_area_all_betas <- NULL
      MOEEQI_input_distance_all_betas <- NULL
      
      for(beta in c(.6,.7,.8,.9)){
        output_all <- get(paste("output_all_beta_",beta,'_noise_level_',noise_level,sep=''))
        input_all <- get(paste("input_all_beta_",beta,'_noise_level_',noise_level,sep=''))
        
        pareto_MOEEQI_n <- pareto_front(output_all[[i-19]][,1], output_all[[i-19]][,2], input_all[[i-19]])$X
        MOEEQI_input_distance_all_betas <- c(MOEEQI_input_distance_all_betas,mean(apply(as.matrix(pdist::pdist(pareto_MOEEQI_n, pareto_true_inputs_scaled)),1,min)))
        
      }
      input_distance_MOEEQI_n <- rbind(input_distance_MOEEQI_n,MOEEQI_input_distance_all_betas)
      
      pareto_qNEHVI_n <- pareto_front(-results_qNEHVI[1:i,]$sty,results_qNEHVI[1:i,]$e_factor,results_qNEHVI[1:i,1:4])$X
      pareto_qNEHVI_n <- Scale(pareto_qNEHVI_n,a=ranges[,1],b=ranges[,2],u=1,l=0)
      input_distance_qNEHVI_n <- c(input_distance_qNEHVI_n,mean(apply(as.matrix(pdist::pdist(pareto_qNEHVI_n,pareto_true_inputs_scaled)),
                                                                     1,min)
      ))

      pareto_TSEMO_n <- pareto_front(-results_TSEMO[1:i,]$sty,results_TSEMO[1:i,]$e_factor,results_TSEMO[1:i,1:4])$X
      pareto_TSEMO_n <- Scale(pareto_TSEMO_n,a=ranges[,1],b=ranges[,2],u=1,l=0)
      input_distance_TSEMO_n <- c(input_distance_TSEMO_n,mean(apply(as.matrix(pdist::pdist(pareto_TSEMO_n,pareto_true_inputs_scaled)),
                                                                   1,min)
      ))
     }

    input_distance_MOEEQI[n,,] <- input_distance_MOEEQI_n
    input_distance_qNEHVI <- rbind(input_distance_qNEHVI,input_distance_qNEHVI_n)
    input_distance_TSEMO <- rbind(input_distance_TSEMO,input_distance_TSEMO_n)
    
  }

  
  
  input_distance_MOEEQI_mean <- apply(input_distance_MOEEQI,c(2,3),mean)
  input_distance_qNEHVI_mean <- apply(input_distance_qNEHVI,2,mean)
  input_distance_TSEMO_mean <- apply(input_distance_TSEMO,2,mean)
  
  
  # over_HV_MOEEQI_min <- apply(over_HV_MOEEQI,2,min)
  # over_HV_qNEHVI_min <- apply(over_HV_qNEHVI,2,min)
  input_distance_MOEEQI_min <- apply(input_distance_MOEEQI,c(2,3),quantile, probs=0.05)
  input_distance_qNEHVI_min <- apply(input_distance_qNEHVI,2,quantile, probs=0.05)
  input_distance_TSEMO_min <- apply(input_distance_TSEMO,2,quantile, probs=0.05)
  
  # over_HV_MOEEQI_max <- apply(over_HV_MOEEQI,2,max)
  # input_distance_qNEHVI_max <- apply(input_distance_qNEHVI,2,max)
  input_distance_MOEEQI_max <- apply(input_distance_MOEEQI,c(2,3),quantile, probs=0.95)
  input_distance_qNEHVI_max <- apply(input_distance_qNEHVI,2,quantile, probs=0.95)
  input_distance_TSEMO_max <- apply(input_distance_TSEMO,2,quantile, probs=0.95)
  
  input_distance_MOEEQI_plot <- NULL
  for(i in 1:4){
    beta <-  c(.6,.7,.8,.9)[i]
    input_distance_MOEEQI_plot <- rbind(input_distance_MOEEQI_plot,data.frame(
      mean=input_distance_MOEEQI_mean[,i], min=input_distance_MOEEQI_min[,i], max=input_distance_MOEEQI_max[,i], method=paste('MOEEQI: beta = ',beta,sep=''), iter=1:21
    ))
  }
  input_distance_qNEHVI_plot <- data.frame(
    mean=input_distance_qNEHVI_mean, min=input_distance_qNEHVI_min, max=input_distance_qNEHVI_max, method="qNEHVI", iter=1:21
  )
  input_distance_TSEMO_plot <- data.frame(
    mean=input_distance_TSEMO_mean, min=input_distance_TSEMO_min, max=input_distance_TSEMO_max, method="TSEMO", iter=1:21
  )
  
  input_distance_plot <- rbind(input_distance_MOEEQI_plot,input_distance_qNEHVI_plot,input_distance_TSEMO_plot)
  
  g <- ggplot(input_distance_plot) +
    geom_line(aes(y = mean, x=iter, group = method, col=method ))+
    geom_ribbon(aes(ymin = min, ymax = max,x=iter, fill = method), alpha = 0.3)+
    ggtitle(paste('Noise level: ',noise_level,' MOEEQI: ',MOEEQI_set_up,' model run(s)',sep=''))
  # ggtitle(paste('Noise level: ',noise_level,' MOEEQI: mult model run: 3',sep=''))
  print(g)
  
}
dev.off()
