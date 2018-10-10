# Load required libraries -------------------------------------------------

library(keras)
library(tm)
library(tidyverse)
library(caret)


# Set essential values for model ------------------------------------------

max_words <- 15000
maxlen <- 100
training_samples <- 2000
validation_samples <- 10000


# Read in dataset ---------------------------------------------------------

data <- read.csv("CleanMovie.csv")

names(data)

data$X <- NULL

df1 <- data[sample(nrow(data)),]

df1$text <- as.character(df1$text)

rownames(df1) <- NULL

table(df1$label)

nrow(df1)

data_part <- df1[1:(nrow(df1)/1), ] 
nrow(data_part)


# Required variables to fill for easier processing ----------------------------------------------

labels <- to_categorical(data_part$label)
texts <- data_part$text


# Tokenizing --------------------------------------------------------------

tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

system.time(data <- pad_sequences(sequences, maxlen = maxlen)) 
cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")


# Read in pre trained glove embeddings ------------------------------------

glove_dir = "glove"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))

embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

cat("Found", length(embeddings_index), "word vectors.\n")

embedding_dim <- 100

num_words <- min(max_words, length(word_index) + 1)

prepare_embedding_matrix <- function() {
  embedding_matrix <- matrix(0L, nrow = num_words, ncol = embedding_dim)
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index >= max_words)
      next
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector)) {
      # words not found in embedding index will be all-zeros.
      embedding_matrix[index,] <- embedding_vector
    }
  }
  embedding_matrix
}

embedding_matrix <- prepare_embedding_matrix()


# Data partitioning -------------------------------------------------------

indices <- sample(1:nrow(data_part))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                (training_samples + validation_samples)]

x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

cat(length(x_train), "train sequences\n")
cat(length(y_val), "test sequences")
cat("Pad sequences (samples x time)\n")


# Dense model -2l -------------------------------------------------------------

remove(model)
K <- backend()
K$clear_session()

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu", 
              kernel_regularizer = regularizer_l2(0.02)
  ) %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 16, activation = "relu"
  ) %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1, activation = "sigmoid")

# # Uncomment to use pre trained word embeddings
# get_layer(model, index = 1) %>%
#   set_weights(list(embedding_matrix)) %>%
#   freeze_weights()

summary(model)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history4 <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  verbose = 1,
  callbacks = callback_tensorboard("logs/run_b"),
  validation_data = list(x_val, y_val)
)


# Dense model -4l -------------------------------------------------------------

remove(model)
K <- backend()
K$clear_session()

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu"
              ) %>%
  layer_dropout(rate = 0.0) %>% 
  layer_dense(units = 64, activation = "relu"
              ) %>%
  layer_dropout(rate = 0.0) %>% 
  layer_dense(units = 32, activation = "relu"
              ) %>%
  layer_dropout(rate = 0.0) %>% 
  layer_dense(units = 16, activation = "relu"
              ) %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1, activation = "sigmoid")

# # Uncomment to use pre trained word embeddings
# get_layer(model, index = 1) %>%
#   set_weights(list(embedding_matrix)) %>%
#   freeze_weights()

summary(model)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history4 <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  verbose = 1,
  callbacks = callback_tensorboard("logs/run_b"),
  validation_data = list(x_val, y_val)
)

tensorboard("logs/run_b")


# Dense model -6l -------------------------------------------------------------

remove(model)
K <- backend()
K$clear_session()

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu"
  ) %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 128, activation = "relu"
  ) %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 126, activation = "relu"
  ) %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 64, activation = "relu"
  ) %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 64, activation = "relu"
  ) %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 16, activation = "relu"
  ) %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1, activation = "sigmoid")

# # Uncomment to use pre trained word embeddings
# get_layer(model, index = 1) %>%
#   set_weights(list(embedding_matrix)) %>%
#   freeze_weights()

summary(model)

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse",
  metrics = c("acc")
)

history4 <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  verbose = 1,
  callbacks = callback_tensorboard("logs/run_b"),
  validation_data = list(x_val, y_val)
)
