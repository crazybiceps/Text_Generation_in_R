# Loading Libraries                                                   ----
library(dplyr)
library(tidyr)
library(keras)
library(rebus)
library(tibble)
library(readr)
library(ggplot2)
library(stringr)

# Reweighting a probability distribution to a different temperature   ----
# reweight_distribution <- function(w , temp = 0.5){
#   distribution <- log(w) / temp
#   distribution <- exp(distribution)
#   
#   distribution / sum(distribution)
# }
# 
# par(mfrow=c(2,5))
# r = runif(100)
# r = r / sum(r)
# 
# sapply(1:10, function(x) r %>% 
#          reweight_distribution(temp = x/10) %>% 
#          hist(main = cat(x/10 , "hist")))

# Implementing character-level LSTM text generation                   ----
# path <- get_file(
#   "/home/kawal/D/nietzsche_textgen.txt" , 
#   origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
# )

w <- readChar( "/home/kawal/D/nietzsche_textgen.txt" , 
                 nchars = file.info("/home/kawal/D/nietzsche_textgen.txt")$size) %>% 
  tolower()

next_char <- function(w , temp = 1){
  distribution <- log(w) / temp
  distribution <- exp(distribution)
  
  rmultinom(1 , 1, (distribution / sum(distribution) )) %>% 
    t %>% 
    which.max()
}

# One-hot-encoding all the letters                                    ----
codes <- data.frame(tokens = w %>% 
                      casefold() %>% 
                      strsplit("") %>% 
                      unlist() %>% 
                      unique() , 
                    numbers = ( 1 : (length(w %>% 
                                            casefold() %>% 
                                            strsplit("") %>% 
                                            unlist() %>% 
                                            unique())) ) )

maxlen <- 60
step   <- 3

start_of_seq   <- seq(1 , nchar(w) / 4  , by = step)
sentence       <- str_sub(w , start = start_of_seq , end = start_of_seq + maxlen - 1)

next_character <- str_sub(w , start = start_of_seq + maxlen , end = start_of_seq + maxlen)

X <- array( 0L , dim = c(sentence %>% length() , maxlen  , nrow(codes)))
Y <- array( 0L , dim = c(sentence %>% length() , nrow(codes)))

for (i  in 1 : (sentence %>% length())) 
{
  word <- sentence[i] %>% strsplit("") %>% unlist()
  
  for (j in 1 : length(word)) 
  {
    X[i , j , codes[[which(codes$tokens == word[j]) , 2]]] <- 1
  } 
  Y[i , codes[[which(codes$tokens == next_character[i]) , 2]]] <- 1
}
codes$tokens <- as.character(codes$tokens)

# Model                                                               ----
model <- keras_model_sequential() %>% 
  layer_lstm(units = 128 , input_shape = c(maxlen , nrow(codes))) %>% 
  layer_dense(units = nrow(codes) , activation = "softmax")

model %>% 
  compile(
    loss = "categorical_crossentropy" , 
    optimizer = optimizer_rmsprop(lr = 0.01)
  )

# Train the model
model %>% 
  fit(
    X , 
    Y , 
    batch_size = 128 , 
    epochs = 50
  )

# Text generation loop                                                ----
text_gen <- function(sent , temperature){

cat(sent)

for (i  in 1:400) {
  x    <- array(0 , dim = c(1 , nchar(sent) , nrow(codes)))
  sent <- sent %>% strsplit("") %>% unlist()
  
  for (j in 1:length(sent)) {
    x[1 , j , codes[[which(codes$tokens == sent[j]) , 2]]] <- 1
  }
  
  l <- model %>% 
    predict(x , verbose = 0) %>% 
    next_char(temp = temperature)
    
  cat(codes[l , 1])
  
  sent <- sent %>% paste(collapse = "")
  sent <- paste0(sent , codes[l , 1] , collapse = "")
 
  sent <- substring(sent , 2)
  
 }
}

# model %>% 
#   save_model_hdf5("/home/kawal/D/text_generator.h5")

# Try this with other document such as whatsapp conversations
