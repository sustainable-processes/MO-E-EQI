library(MaxPro)
library(tidyverse)

Scale <- function(x, a = 0, b = 1, l = -1, u = 1) {
  
  if( ( is.matrix( x ) | is.vector( x ) ) == FALSE ){ stop( "x must be a vector or a matrix." ) }
  if( length( a ) != 1 & length( a ) != NCOL( x ) ){ stop( "a must be a vector of length 1 or equal to number of columns of x" ) }
  if( length( b ) != 1 & length( b ) != NCOL( x ) ){ stop( "b must be a vector of length 1 or equal to number of columns of x" ) }
  if( length( l ) != 1 & length( l ) != NCOL( x ) ){ stop( "l must be a vector of length 1 or equal to number of columns of x" ) }
  if( length( u ) != 1 & length( u ) != NCOL( x ) ){ stop( "u must be a vector of length 1 or equal to number of columns of x" ) }
  
  t( (u - l) * ((t( x ) - a)/(b - a)) + l )
  
}

InitialDesign <- function(n, p, x1, x2, x3, x4) {
  # Latin hypercube design for continuous factors 
  obj<-MaxProLHD(n, p)
  FinalDesign <- obj$Design

  FinalDesign[,1:p] <- Scale(as.matrix(FinalDesign[,1:p]), 
                             l=c(x1[1],x2[1],x3[1],x4[1]),
                             u=c(x1[2],x2[2],x3[2],x4[2]) )
  
  colnames(FinalDesign) <- c("res_time","equiv","conc","temp")
  return(FinalDesign)
}
  
# To call this function, use:
# InitialDesign(20,4,c(0.5,2.0),c(1.0,5.0),c(0.1,0.5),c(30,120))