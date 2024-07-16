library(reticulate)
library(tidyverse)
library(MOEEQI)
library(PBSmapping)
library(tidyverse)
library(sf)
library(gridExtra)

use_condaenv("r-reticulate")
botorch <- import("botorch")
torch <- import("torch")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("../ReactionSimulator_snar_new_log_linear.R") 
is_scaled <- '_scale'#''#
is_linear <- 'log linear'#'linear'#


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


ref_point <- torch$tensor(c(800,-4))
data <- torch$tensor(cbind(-pareto_true$STY,-pareto_true[,6]))
pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
true_pareto_y = data[pareto_mask]
true_pareto_y <-  cbind(X=c(ref_point$numpy()[1],rep(true_pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]),
                        Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(true_pareto_y$numpy()[,2], each=2),ref_point$numpy()[2]))

polygon_true_pareto <- st_polygon(list(true_pareto_y))
sf_true_pareto <- st_sfc(polygon_true_pareto, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
sf_true_pareto_df <- st_coordinates(sf_true_pareto) %>% as.data.frame()





for(beta in c(.6,.7,.8,.9)){
  
  if(is_scaled == '_scale'){
    folder_path=paste('../scaled/',is_linear,'/',sep="")
  }else{
    folder_path==paste('../not scaled/',is_linear,'/',sep="")
  }
  # PDFPath = paste(folder_path,"MOEEQI_",MOEEQI_set_up,"_scaled_inputs_outputs_HV_distance_beta_",beta,".pdf",sep='')
  # pdf(file=PDFPath, width=15, height=13)
  
  

plots <- list()
ind <- 1
alg_names <- c("qNEHVI","qNParEGO","random","TSEMO")#"qEHVI",


for(MOEEQI_set_up in c('single', 'mult') ){
# noise_level <- 0.2

# for(noise_level in c(0.01,0.05,0.1,0.2)){
for(case in c(1,2)){
  if(case==1){
      noise_slope=0.8495
      noise_level=-1.699
  }else{
    noise_slope=1.2
    noise_level=-1.3
  }
  over_HV_MOEEQI <- over_HV_qEHVI <- over_HV_qNEHVI <- over_HV_qNParEGO  <- over_HV_random <- over_HV_TSEMO<- NULL
  for(n in 1:20){
    over_HV_MOEEQI_n <- NULL
    
    # if(noise_level %in% c(0.1,0.2)){
    #   noise_level_round <- paste(noise_level,'0',sep='')
    # }else{
    #   noise_level_round <- paste(noise_level)
    # }
    for(name in alg_names){
      csv_name <- paste('../../code_python/results/loglinear/',name,'/TrainingSet1_loglinear_',case,is_scaled,'/repeat_',n,'.csv', sep = "")
      assign(paste('results_',name,sep=""), read.csv(csv_name, header=T))
    }
    # 
    # csv_name <- paste('../../code_python/results/qEHVI/TrainingSet1_linear_',noise_level_round,is_scaled,'/repeat_',n,'.csv', sep = "")
    # results_qEHVI <- read.csv(csv_name, header=T)
    # 
    # csv_name <- paste('../../code_python/results/qNEHVI/TrainingSet1_linear_',noise_level_round,is_scaled,'/repeat_',n,'.csv', sep = "")
    # results_qNEHVI <- read.csv(csv_name, header=T)
    # 
    # csv_name <- paste('../../code_python/results/qNParEGO/TrainingSet1_linear_',noise_level_round,is_scaled,'/repeat_',n,'.csv', sep = "")
    # results_qNParEGO <- read.csv(csv_name, header=T)
    # 
    # csv_name <- paste('../../code_python/results/random/TrainingSet1_linear_',noise_level_round,'_scale/repeat_',n,'.csv', sep = "")
    # results_random <- read.csv(csv_name, header=T)
    # 
    # csv_name <- paste('../../code_python/results/TSEMO/TrainingSet1_linear_',noise_level_round,'_scale/repeat_',n,'.csv', sep = "")
    # results_TSEMO <- read.csv(csv_name, header=T)
  
    load(paste(folder_path,'results_2024_',MOEEQI_set_up,'_log_noise_',noise_slope,'_',noise_level,'_beta_',beta,'/data',n,'.RData', sep = ""))
    assign(paste("output_all_beta_",beta,'_noise_level_',noise_level,sep=''),output_all)
    
    data <- cbind(-output_all[[1]][,1],-output_all[[1]][,2])
    
    
    #Sort the data points
    data <- data[order(data[,1], decreasing = T),]
    data <- torch$tensor(data)
    pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
    pareto_y = data[pareto_mask]
    
    
    # Create step-function pareto front from pareto points
    p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
                Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
    )
    
    
    polygon1 <- st_polygon(list(p1))
    
    # Create simple features objects
    sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
    
    # Calculate the difference
    difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
    difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
    
    union_polygon <- st_union(difference_polygon1, difference_polygon2)
    
    valid <- st_is_valid(union_polygon)
    if (!valid) {
      warning("The difference polygon is not valid. Attempting to fix...")
      union_polygon <- st_make_valid(union_polygon)
    }
    
    over_HV_MOEEQI_n <- c(over_HV_MOEEQI_n,as.numeric(st_area(union_polygon)))
    
    
    for(name in alg_names){
      results_name <- get(paste("results_",name,sep=""))
      data <- cbind(results_name[1:20,5],-results_name[1:20,6])
      #Sort the data points
      data <- data[order(data[,1], decreasing = T),]
      data <- torch$tensor(data)
      pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
      pareto_y = data[pareto_mask]
      
      
      # Create step-function pareto front from pareto points
      p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
                  Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
      )
      
      
      polygon1 <- st_polygon(list(p1))
      
      # Create simple features objects
      sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
      
      # Calculate the difference
      difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
      difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
      
      union_polygon <- st_union(difference_polygon1, difference_polygon2)
      
      valid <- st_is_valid(union_polygon)
      if (!valid) {
        warning("The difference polygon is not valid. Attempting to fix...")
        union_polygon <- st_make_valid(union_polygon)
      }
      
      assign(paste("over_HV_",name,"_n",sep=""),as.numeric(st_area(union_polygon)))
    }
    # 
    #   
    # # over_HV_MOEEQI_n <- as.matrix(p3[,c(3,4)]) %>% areapl
    # # HV_MOEEQI_n <- (botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y)
    # # log_diff_HV_MOEEQI_n <- #log
    # #   (HV_true-(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y))
    # 
    # 
    # data <- cbind(results_qNEHVI[1:20,5],-results_qNEHVI[1:20,6])
    # #Sort the data points
    # data <- data[order(data[,1], decreasing = T),]
    # data <- torch$tensor(data)
    # pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
    # pareto_y = data[pareto_mask]
    # 
    # 
    # # Create step-function pareto front from pareto points
    # p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
    #             Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
    # )
    # 
    # 
    # polygon1 <- st_polygon(list(p1))
    # 
    # # Create simple features objects
    # sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
    # 
    # # Calculate the difference
    # difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
    # difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
    # 
    # union_polygon <- st_union(difference_polygon1, difference_polygon2)
    # 
    # valid <- st_is_valid(union_polygon)
    # if (!valid) {
    #   warning("The difference polygon is not valid. Attempting to fix...")
    #   union_polygon <- st_make_valid(union_polygon)
    # }
    # 
    # over_HV_qNEHVI_n <- st_area(union_polygon)
    # 
    # 
    # # over_HV_qNEHVI_n <- as.matrix(p3[,c(3,4)]) %>% areapl
    # # HV_qNEHVI_n <- (botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y)
    # # log_diff_HV_qNEHVI_n <- #log
    # # (HV_true-(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y))
    # 
    # 
    # data <- cbind(results_TSEMO[1:20,5],-results_TSEMO[1:20,6])
    # #Sort the data points
    # data <- data[order(data[,1], decreasing = T),]
    # data <- torch$tensor(data)
    # pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
    # pareto_y = data[pareto_mask]
    # 
    # 
    # # Create step-function pareto front from pareto points
    # p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
    #             Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
    # )
    # 
    # 
    # polygon1 <- st_polygon(list(p1))
    # 
    # # Create simple features objects
    # sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
    # 
    # # Calculate the difference
    # difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
    # difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
    # 
    # union_polygon <- st_union(difference_polygon1, difference_polygon2)
    # 
    # valid <- st_is_valid(union_polygon)
    # if (!valid) {
    #   warning("The difference polygon is not valid. Attempting to fix...")
    #   union_polygon <- st_make_valid(union_polygon)
    # }
    # 
    # over_HV_TSEMO_n <- st_area(union_polygon)
    # # over_HV_TSEMO_n <- as.matrix(p3[,c(3,4)]) %>% areapl
    # # HV_TSEMO_n <- (botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y)
    # # log_diff_HV_TSEMO_n <- #log
    # #   (HV_true-(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y))
    for(i in 21:40){
      
      output_all <- get(paste("output_all_beta_",beta,'_noise_level_',noise_level,sep=''))
      data <- cbind(-output_all[[i-19]][,1],-output_all[[i-19]][,2])
      # data <- cbind(results_MOEEQI[1:i,5],-results_MOEEQI[1:i,6])
      #Sort the data points
      data <- data[order(data[,1], decreasing = T),]
      data <- torch$tensor(data)
      pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
      pareto_y = data[pareto_mask]
      
      
      # Create step-function pareto front from pareto points
      p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
                  Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
      )
      
      
      polygon1 <- st_polygon(list(p1))
      
      # Create simple features objects
      sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
      
      valid <- st_is_valid(sf_polygon1)
      if (!valid) {
        warning("The difference polygon is not valid. Attempting to fix...")
        sf_polygon1 <- st_make_valid(sf_polygon1)
      }
      
      # Calculate the difference
      difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
      difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
      
      union_polygon <- st_union(difference_polygon1, difference_polygon2)
      
      valid <- st_is_valid(union_polygon)
      if (!valid) {
        warning("The difference polygon is not valid. Attempting to fix...")
        union_polygon <- st_make_valid(union_polygon)
      }
      
      over_HV_MOEEQI_n <- c(over_HV_MOEEQI_n, as.numeric(st_area(union_polygon)))
      
      # HV_MOEEQI_n <- c(HV_MOEEQI_n,(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y))
      # log_diff_HV_MOEEQI_n <- c(log_diff_HV_MOEEQI_n, #log
      #                           (HV_true-(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y)))
      
      
      for(name in alg_names){
        results_name <- get(paste("results_",name,sep=""))
        data <- cbind(results_name[1:i,5],-results_name[1:i,6])
        #Sort the data points
        data <- data[order(data[,1], decreasing = T),]
        data <- torch$tensor(data)
        pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
        pareto_y = data[pareto_mask]
        
        # Create step-function pareto front from pareto points
        p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
                    Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
        )
        
        
        polygon1 <- st_polygon(list(p1))
        
        # Create simple features objects
        sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
        
        # Calculate the difference
        difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
        difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
        
        union_polygon <- st_union(difference_polygon1, difference_polygon2)
        
        valid <- st_is_valid(union_polygon)
        if (!valid) {
          warning("The difference polygon is not valid. Attempting to fix...")
          union_polygon <- st_make_valid(union_polygon)
        }
        
        assign(paste("over_HV_",name,"_n",sep=""), c(get(paste("over_HV_",name,"_n",sep="")), as.numeric(st_area(union_polygon))))
      }
      # data <- cbind(results_qNEHVI[1:i,5],-results_qNEHVI[1:i,6])
      # #Sort the data points
      # data <- data[order(data[,1], decreasing = T),]
      # data <- torch$tensor(data)
      # pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
      # pareto_y = data[pareto_mask]
      # 
      # # Create step-function pareto front from pareto points
      # p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
      #             Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
      # )
      # 
      # 
      # polygon1 <- st_polygon(list(p1))
      # 
      # # Create simple features objects
      # sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
      # 
      # # Calculate the difference
      # difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
      # difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
      # 
      # union_polygon <- st_union(difference_polygon1, difference_polygon2)
      # 
      # valid <- st_is_valid(union_polygon)
      # if (!valid) {
      #   warning("The difference polygon is not valid. Attempting to fix...")
      #   union_polygon <- st_make_valid(union_polygon)
      # }
      # 
      # over_HV_qNEHVI_n <- c(over_HV_qNEHVI_n, st_area(union_polygon))
      # 
      # # HV_qNEHVI_n <- c(HV_qNEHVI_n,(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y))
      # # log_diff_HV_qNEHVI_n <- c(log_diff_HV_qNEHVI_n,#log
      # #                           (HV_true-(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y)))
      # 
      # 
      # data <- cbind(results_TSEMO[1:i,5],-results_TSEMO[1:i,6])
      # #Sort the data points
      # data <- data[order(data[,1], decreasing = T),]
      # data <- torch$tensor(data)
      # pareto_mask <- botorch$utils$multi_objective$pareto$is_non_dominated(data)
      # pareto_y = data[pareto_mask]
      # 
      # 
      # # Create step-function pareto front from pareto points
      # p1 <- cbind(X=c(ref_point$numpy()[1],rep(pareto_y$numpy()[,1], each=2),ref_point$numpy()[1],ref_point$numpy()[1]), 
      #             Y=c(ref_point$numpy()[2],ref_point$numpy()[2],rep(pareto_y$numpy()[,2], each=2),ref_point$numpy()[2])
      # )
      # 
      # 
      # polygon1 <- st_polygon(list(p1))
      # 
      # # Create simple features objects
      # sf_polygon1 <- st_sfc(polygon1, crs = st_crs("+proj=utm +zone=33 +datum=WGS84 +units=m"))
      # 
      # # Calculate the difference
      # difference_polygon1 <- st_difference(sf_polygon1, sf_true_pareto)
      # difference_polygon2 <- st_difference(sf_true_pareto, sf_polygon1)
      # 
      # union_polygon <- st_union(difference_polygon1, difference_polygon2)
      # 
      # valid <- st_is_valid(union_polygon)
      # if (!valid) {
      #   warning("The difference polygon is not valid. Attempting to fix...")
      #   union_polygon <- st_make_valid(union_polygon)
      # }
      # over_HV_TSEMO_n <- c(over_HV_TSEMO_n, st_area(union_polygon))
      
      # HV_TSEMO_n <- c(HV_TSEMO_n,(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y))
      # log_diff_HV_TSEMO_n <- c(log_diff_HV_TSEMO_n,#log
      #                           (HV_true-(botorch$utils$multi_objective$hypervolume$Hypervolume(ref_point=ref_point))$compute(pareto_y)))
    }
    
    
    # HV_MOEEQI <- rbind(HV_MOEEQI,HV_MOEEQI_n)
    # log_diff_HV_MOEEQI <- rbind(log_diff_HV_MOEEQI,log_diff_HV_MOEEQI_n)
    over_HV_MOEEQI <- rbind(over_HV_MOEEQI,over_HV_MOEEQI_n)
    
    for(name in alg_names){
      assign(paste("over_HV_",name,sep=""), rbind(get(paste("over_HV_",name,sep="")), get(paste("over_HV_",name,"_n",sep=""))))
    }
    # # HV_qNEHVI <- rbind(HV_qNEHVI,HV_qNEHVI_n)
    # # log_diff_HV_qNEHVI <- rbind(log_diff_HV_qNEHVI,log_diff_HV_qNEHVI_n)
    # over_HV_qNEHVI <- rbind(over_HV_qNEHVI,over_HV_qNEHVI_n)
    # 
    # # HV_TSEMO <- rbind(HV_TSEMO,HV_TSEMO_n)
    # # log_diff_HV_TSEMO <- rbind(log_diff_HV_TSEMO,log_diff_HV_TSEMO_n)
    # over_HV_TSEMO <- rbind(over_HV_TSEMO,over_HV_TSEMO_n)
  }
  
  # sum(is.na(log_diff_HV_MOEEQI))
  # sum(is.na(log_diff_HV_qNEHVI))
  
  # HV_MOEEQI_mean <- apply(HV_MOEEQI,2,mean)
  # HV_qNEHVI_mean <- apply(HV_qNEHVI,2,mean)
  # # HV_TSEMO_mean <- apply(HV_TSEMO,2,mean)
  # 
  # 
  # HV_MOEEQI_min <- apply(HV_MOEEQI,2,min)
  # HV_qNEHVI_min <- apply(HV_qNEHVI,2,min)
  # # HV_TSEMO_min <- apply(HV_TSEMO,2,min)
  # 
  # HV_MOEEQI_max <- apply(HV_MOEEQI,2,max)
  # HV_qNEHVI_max <- apply(HV_qNEHVI,2,max)
  # # HV_TSEMO_max <- apply(HV_TSEMO,2,max)
  # 
  # HV_MOEEQI_plot <- data.frame(
  #   mean=HV_MOEEQI_mean, min=HV_MOEEQI_min, max=HV_MOEEQI_max, Method="MOEEQI", iter=1:21
  # )
  # HV_qNEHVI_plot <- data.frame(
  #   mean=HV_qNEHVI_mean, min=HV_qNEHVI_min, max=HV_qNEHVI_max, Method="qNEHVI", iter=1:21
  # )
  # # HV_TSEMO_plot <- data.frame(
  # #   mean=HV_TSEMO_mean, min=HV_TSEMO_min, max=HV_TSEMO_max, Method="TSEMO", iter=1:21
  # # )
  # 
  # HV_plot <- rbind(HV_MOEEQI_plot,HV_qNEHVI_plot)#,HV_TSEMO_plot)
  # 
  # g <- ggplot(HV_plot) + 
  #   geom_line(aes(y = mean, x=iter, group = Method, col=Method ))+
  #   geom_ribbon(aes(ymin = min, ymax = max,x=iter, fill = Method), alpha = 0.3)+
  #   ggtitle(paste('Noise level: ',noise_level,sep=''))
  # 
  # print(g)
  # 
  # 
  # # plot log HV diff
  # 
  # log_diff_HV_MOEEQI_mean <- apply(log_diff_HV_MOEEQI,2,mean)
  # log_diff_HV_qNEHVI_mean <- apply(log_diff_HV_qNEHVI,2,mean)
  # # HV_TSEMO_mean <- apply(HV_TSEMO,2,mean)
  # 
  # 
  # log_diff_HV_MOEEQI_min <- apply(log_diff_HV_MOEEQI,2,min)
  # log_diff_HV_qNEHVI_min <- apply(log_diff_HV_qNEHVI,2,min)
  # # HV_TSEMO_min <- apply(HV_TSEMO,2,min)
  # 
  # log_diff_HV_MOEEQI_max <- apply(log_diff_HV_MOEEQI,2,max)
  # log_diff_HV_qNEHVI_max <- apply(log_diff_HV_qNEHVI,2,max)
  # # HV_TSEMO_max <- apply(HV_TSEMO,2,max)
  # 
  # log_diff_HV_MOEEQI_plot <- data.frame(
  #   mean=log_diff_HV_MOEEQI_mean, min=log_diff_HV_MOEEQI_min, max=log_diff_HV_MOEEQI_max, Method="MOEEQI", iter=1:21
  # )
  # log_diff_HV_qNEHVI_plot <- data.frame(
  #   mean=log_diff_HV_qNEHVI_mean, min=log_diff_HV_qNEHVI_min, max=log_diff_HV_qNEHVI_max, Method="qNEHVI", iter=1:21
  # )
  # # HV_TSEMO_plot <- data.frame(
  # #   mean=HV_TSEMO_mean, min=HV_TSEMO_min, max=HV_TSEMO_max, Method="TSEMO", iter=1:21
  # # )
  # 
  # HV_plot <- rbind(log_diff_HV_MOEEQI_plot,log_diff_HV_qNEHVI_plot)#,HV_TSEMO_plot)
  # 
  # g <- ggplot(HV_plot) +
  #   geom_line(aes(y = mean, x=iter, group = Method, col=Method ))+
  #   geom_ribbon(aes(ymin = min, ymax = max,x=iter, fill = Method), alpha = 0.3)+
  #   ggtitle(paste('Noise level: ',noise_level,' log difference HV.',sep=''))
  # 
  # print(g)
  
  
  
  # Write code for extracting different betas into one plot
  
  ###############################################################
  ###############################################################
  ###############################################################
  ###############################################################
  ###############################################################
  
  
  over_HV_MOEEQI_mean <- apply(over_HV_MOEEQI,2,mean)
  over_HV_MOEEQI_min <- apply(over_HV_MOEEQI,2,quantile, probs=0.05)
  over_HV_MOEEQI_max <- apply(over_HV_MOEEQI,2,quantile, probs=0.95)
  
  for(name in alg_names){
    assign(paste("over_HV_",name,"_mean",sep=""),apply(get(paste("over_HV_",name,sep="")),2,mean))
    assign(paste("over_HV_",name,"_min",sep=""),apply(get(paste("over_HV_",name,sep="")),2,quantile, probs=0.05))
    assign(paste("over_HV_",name,"_max",sep=""),apply(get(paste("over_HV_",name,sep="")),2,quantile, probs=0.95))
    
  }
  # # over_HV_MOEEQI_min <- apply(over_HV_MOEEQI,2,min)
  # # over_HV_qNEHVI_min <- apply(over_HV_qNEHVI,2,min)
  # over_HV_MOEEQI_min <- apply(over_HV_MOEEQI,2,quantile, probs=0.05)
  # over_HV_qNEHVI_min <- apply(over_HV_qNEHVI,2,quantile, probs=0.05)
  # over_HV_TSEMO_min <- apply(over_HV_TSEMO,2,quantile, probs=0.05)
  # 
  # # over_HV_MOEEQI_max <- apply(over_HV_MOEEQI,2,max)
  # # over_HV_qNEHVI_max <- apply(over_HV_qNEHVI,2,max)
  # over_HV_MOEEQI_max <- apply(over_HV_MOEEQI,2,quantile, probs=0.95)
  # over_HV_qNEHVI_max <- apply(over_HV_qNEHVI,2,quantile, probs=0.95)
  # over_HV_TSEMO_max <- apply(over_HV_TSEMO,2,quantile, probs=0.95)
  
  
  over_HV_MOEEQI_plot <- data.frame(
    mean=over_HV_MOEEQI_mean, min=over_HV_MOEEQI_min, max=over_HV_MOEEQI_max, Method=paste('MOEEQI: \u03B2 = ',beta,sep=''), iter=1:21
  )
  for(name in alg_names){
    assign(paste("over_HV_",name,"_plot",sep=""), data.frame(
      mean=get(paste("over_HV_",name,"_mean",sep="")), min=get(paste("over_HV_",name,"_min",sep="")), max=get(paste("over_HV_",name,"_max",sep="")), Method=name, iter=1:21
    ))
  }
  # 
  #   over_HV_qNEHVI_plot <- data.frame(
  #     mean=over_HV_qNEHVI_mean, min=over_HV_qNEHVI_min, max=over_HV_qNEHVI_max, Method="qNEHVI", iter=1:21
  #   )
  #   over_HV_TSEMO_plot <- data.frame(
  #     mean=over_HV_TSEMO_mean, min=over_HV_TSEMO_min, max=over_HV_TSEMO_max, Method="TSEMO", iter=1:21
  #   )
  
  HV_plot <- over_HV_MOEEQI_plot
  for(name in alg_names){
    HV_plot <- rbind(HV_plot,get(paste(paste("over_HV_",name,"_plot",sep=""))))
  }
  
  g <- ggplot(HV_plot) +
    geom_line(aes(y = mean, x=iter, group = Method, col=Method), size=1 )+
    # scale_y_discrete(breaks = c(10000,20000))+
    geom_ribbon(aes(ymin = min, ymax = max, x=iter, fill = Method), alpha = 0.2)+
    ggtitle(paste('Noise level: ',noise_level,', Noise slope: ',noise_slope,' MOEEQI: ',MOEEQI_set_up,' model run(s)',sep=''))+
    ylab("HV distance")+xlab("Iteration")+theme(legend.position = c(0.85, 0.8),axis.text.x=element_text(size=11),axis.text.y = element_text(size=8, angle = 90, hjust = 0.5),
                                                axis.title=element_text(size=15),plot.title = element_text(size=15),legend.title=element_text(size=15),
                                                panel.background = element_rect(fill='transparent'), #transparent panel bg
                                                plot.background = element_rect(fill='transparent', color=NA), #transparent plot bg
                                                panel.grid.major = element_blank(), #remove major gridlines
                                                panel.grid.minor = element_blank(), #remove minor gridlines
                                                legend.text=element_text(size=15), legend.background = element_rect(fill='transparent'),
                                                axis.line = element_line(colour = "black")
    )
  plots[[ind]] <- g
  ind <- ind+1
  
  # print(g)
}
}

# Path = paste(folder_path,"MOEEQI_",MOEEQI_set_up,"_scaled_inputs_outputs_HV_distance_beta_",beta,".",sep='')

# legend <- cowplot::get_legend(plots[[9]] + theme(legend.position = "bottom"))
# plots <- lapply(plots, function(x) x + theme(legend.position = "none"))
# 
# g_combined <- gridExtra::grid.arrange(arrangeGrob(plots[[1]],plots[[5]],
#                                                   plots[[2]],plots[[6]],
#                                                   plots[[3]],plots[[7]],
#                                                   plots[[4]],plots[[8]],
#                                                   ncol = 2),
#                         arrangeGrob(legend, ncol=1, nrow=1), heights=c(15,1))
# print(g_combined)

lemon::grid_arrange_shared_legend(plots[[1]],plots[[3]],
                                  plots[[2]],plots[[4]],
                                  ncol = 2, nrow = 2, position='bottom')

# ggsave(file=paste(Path,'png',sep=""), plot=image, width=11, height=13)
# ggsave(file=paste(Path,'svg',sep=""), plot=image, width=11, height=13)
# ggsave(file=paste(Path,'pdf',sep=""), plot=image, width=11, height=13, device = cairo_pdf)
# g_combined <- gridExtra::arrangeGrob(arrangeGrob(plots[[1]],plots[[2]],plots[[3]],plots[[4]],
#                                                  plots[[5]],plots[[6]],plots[[7]],plots[[8]],ncol = 2),
#                                      arrangeGrob(legend, ncol=1, nrow=1), heights=c(15,1))
# 
# ggplot2::ggsave(filename = paste(folder_path,"MOEEQI_",MOEEQI_set_up,"_scaled_inputs_outputs_HV_distance_beta_",beta,".pdf",sep=''),
# plot = g_combined,
# device = cairo_ps,
# dpi = 1200,
# width = 13,
# height = 19
# )

# dev.off()
}
