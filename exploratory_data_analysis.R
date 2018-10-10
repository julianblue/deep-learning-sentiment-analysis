# Loading required Packages -----------------------------------------------
library(tm) 
library(wordcloud)  
library(tidyverse)
library(textclean)
library(lubridate)
library(sentimentr)
library(twitteR)
library(igraph)
library(FactoMineR)
library(cluster)
library(scales)
library(broom)
library(rreview)
library(udpipe)
library(tidytext)
library(reshape2)
library(quanteda)
library(tibble)


# Loading data into RStudio -----------------------------------------------
# Set file path
imdb_dir <-
  "/Users/julianblau/Downloads/aclImdb 2"

# Select train folder and load txt files
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for (fname in list.files(dir_name,
                           pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}
# Format text as characters and and labels as factors
train_text <- as.list(as.character(texts))
train_labels <- as.list(as.factor(labels))

# Select test folder and load txt files
test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(test_dir, label_type)
  for (fname in list.files(dir_name,
                           pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}
# Format text as characters and and labels as factors
test_text <- as.list(as.character(texts))
test_labels <- as.list(as.factor(labels))

# Merge test and train data by text and label
full_text <- rbind(train_text, test_text)
full_labels <- rbind(train_labels, test_labels)

# Merge text and label data to dataframe
full <- do.call(rbind, Map(data.frame, Text = full_text, Label = full_labels))

# Create text variable for text analysis
txt <- full$Text
txt <- as.character(txt)


# Review Text Data Analysis ----------------------------------------------

# Characters per review
chars_per_review = sapply(txt, nchar)
summary(chars_per_review)

# Split words
words_list = strsplit(txt, " ")

# Words per review
words_per_review = sapply(words_list, length)
words_per_review
# Data preperation for plot
x_breaks <- seq(0, 1000, 100)
x <- cut(words_per_review, breaks = x_breaks)
wps <- data.frame(table(x))
wps
attach(wps)

# Plot number of words per review
P1 <- ggplot(wps, aes(x=x, y=Freq))+
  geom_histogram(stat = "identity")+
  scale_x_discrete(labels=c("0-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700", "700-800", "800-900", "900-100"))+
  scale_y_continuous(limits = c(0, 25000))+
  labs(x="Range of words", y="Number of reviews", title="Number of words per review")+
  theme(text = element_text(size=15),
        axis.text.x = element_text(size = 13))
P1


# Length of words per review
wsize_per_review = sapply(words_list, function(x) mean(nchar(x)))
prop.table(table(round(wsize_per_review)))
# barplot
barplot(table(round(wsize_per_review)), border=NA,
        xlab = "word length in number of characters",
        main="Distribution of words length per review", cex.main=1)
summary(wsize_per_review)


# Unique words per review
uniq_words_per_review = sapply(words_list, function(x) length(unique(x)))
# barplot
scatter.smooth(table(uniq_words_per_review), border=NA,
        main="Distribution of unique words per review", cex.main=1, log2="x",
        xlab = "Unique words",
        ylab = "Number of reviews",
        cex.main = 2,
        cex.axis= 1)

# Hashtags per review
hash_per_review = sapply(words_list, function(x) length(grep("#", x)))
table(hash_per_review)
hash_per_review <- as.numeric(hash_per_review)
hpt <- data.frame(table(hash_per_review))
hpt
prop.table(table(hash_per_review))
barplot(table(hash_per_review), log = "y")


# At mentions per review
ats_per_review = sapply(words_list, function(x) length(grep("@", x)))
table(ats_per_review)
ats_per_review <- as.numeric(ats_per_review)
apt <- data.frame(table(ats_per_review))
apt
prop.table(table(ats_per_review))
barplot(table(ats_per_review))


# Http links per review
links_per_review = sapply(words_list, function(x) length(grep("http", x)))
table(links_per_review)
links_per_review <- as.numeric(links_per_review)
lpt <- data.frame(table(links_per_review))
lpt
prop.table(table(links_per_review))
barplot(table(links_per_review))

colnames(lpt) <- c("x", "Freq")
colnames(apt) <- c("x", "Freq")
colnames(hpt) <- c("x", "Freq")
lpt
apt
hpt

df <- merge(lpt, merge(apt, hpt, by="x", all.x=TRUE, all.y=TRUE), by = "x", all.x = TRUE, all.y = TRUE)
df
#df[is.na(df)] <- 0
df
colnames(df) <- c("num_occ", "links per review", "ats per review", "hashtags per review")

df.m <- melt(df, id.vars = "num_occ")
df.m
P12 <- ggplot(df.m, aes(x = num_occ, y = value, fill=variable, group=variable)) +
  geom_bar(stat='identity', position = "dodge")+
  scale_y_log10()+
  guides(fill=guide_legend(title="Components"))+
  stat_summary(fun.y=mean, show.legend = T, geom="line",lwd=0.7,aes(group=variable, color=variable))+
  labs(title="Number of links, ats and hashtags per review", x="Number of components per review", y="Frequency")+
  theme(text = element_text(size=15),
        axis.text.x = element_text())
P12


# Data frame for text analysis plots
icedf = data.frame(
  chars=chars_per_review,
  words = words_per_review,
  lengths = wsize_per_review,
  uniqs = uniq_words_per_review,
  hashs = hash_per_review,
  ats = ats_per_review,
  links = links_per_review
)


# Nr of characters vs. number of words in review
ggplot(icedf, aes(x=words, y=chars)) +
  geom_point(colour="gray20", alpha=0.2) +
  stat_smooth(method="lm") +
  labs(x="Number of words per review", y="Number of characters per review", title="Words versus Characters of Reviews")+
  scale_x_continuous(limits = c(0, 1100))+
  scale_y_continuous(limits = c(0, 7500))+
  theme(text = element_text(size=15),
        axis.text.x = element_text()) 

# Word length vs. number of words in review
ggplot(icedf, aes(x=words, y=lengths)) +
  geom_point(colour="gray20", alpha=0.2) +
  stat_smooth(method="lm") +
  labs(x="Number of words per review", y="Length of words per review", title="Word Length versus Number of Words of Reviews")+
  scale_x_continuous(limits = c(0, 1100))+
  scale_y_continuous(limits = c(3, 7))+
  theme(text = element_text(size=12),
        axis.text.x = element_text()) 

# Unique words in total
uniq_words = unique(unlist(words_list))
length(uniq_words)
length(unlist(words_list))
# lexical diversity
length(uniq_words) / length(unlist(words_list))

# Most frequent words
mfw = sort(table(unlist(words_list)), decreasing=TRUE)

# Top-20 most frequent
top20 = head(mfw, 20)
top20

# Barplot of top 20 most frequent words
barplot(top20, border=NA, las=2, main="Top 20 most frequent terms", cex.main=1)


# Corpus development ------------------------------------------------------

reviewssource <- VectorSource(full$Text)  # creating a vector source with the text from "reviews"

reviewCorpus <- VCorpus(reviewssource)  # creating the corpus 

reviewCorpus  # initial look at the corpus 


# Clean text data ---------------------------------------------------------

removeURL <- function(x) gsub("http[[:alnum:][:punct:]]*", "", x)
removeEmoji <- function(x) gsub("\\W"," ", x)
removeRL <- function(x) gsub("([[:alpha:]])\\1{2,}", "\\1", x)
removeMentions <- function(x) gsub("@\\w+ *", "", x)
removeLinks <- function(x) gsub("http[[:alnum:][:punct:]]*", "", x)
cleanDATA <- tm_map(reviewCorpus, removeURL)
cleanDATA <- tm_map(cleanDATA, removeRL)  # reducing repetitive letters from above 3 to 1
cleanDATA <- tm_map(cleanDATA, removeEmoji)  # using tm_map to remove emoji unicodes
cleanDATA <- tm_map(cleanDATA, removeMentions)
cleanDATA <- tm_map(cleanDATA, removeLinks)
cleanDATA <- tm_map(cleanDATA, tolower)
cleanDATA <- tm_map(cleanDATA, removePunctuation)
cleanDATA <- tm_map(cleanDATA, removeNumbers)
cleanDATA <- tm_map(cleanDATA, removeWords, stopwords("english"))
cleanDATA <- tm_map(cleanDATA, stripWhitespace)
cleanDATA <- tm_map(cleanDATA, PlainTextDocument)
# dictCorpus <- myCorpus
# myCorpus <- tm_map(myCorpus, stemDocument)
# # tokenize the corpus
# myCorpusTokenized <- lapply(myCorpus, scan_tokenizer)
# # stem complete each token vector
# myTokensStemCompleted <- lapply(myCorpusTokenized, stemCompletion, dictCorpus)
cleanCorpus <- cleanDATA


# Further text cleaning ---------------------------------------------------------

# Convert corpus back to dataframe for additional text cleaning
review.data <- data.frame(text=unlist(sapply(cleanCorpus, `[`, "content")), 
                         stringsAsFactors=F)

# Converting words with apostrophes to two seperate words 
# and remove other errors caused by punctuation.
review.data = review.data %>% mutate(
  text = gsub(" n t"," not", tolower(text)), 
  text = gsub("he s","he is", tolower(text)),
  text = gsub("br br","", tolower(text)), 
  text = gsub(" ,",",", tolower(text)), 
  text = gsub("don t","do not", tolower(text)), 
  text = gsub("doesn t","does not", tolower(text)), 
  text = gsub("didn t","did not", tolower(text)),
  text = gsub("can t","can not", tolower(text)), 
  text = gsub("isn t","is not", tolower(text)), 
  text = gsub("wasn t","was not", tolower(text)), 
  text = gsub("film s","film", tolower(text)), 
  text = gsub("she s","she is", tolower(text)), 
  text = gsub("what s","what is", tolower(text)), 
  text = gsub("that s","that is", tolower(text)), 
  text = gsub("there s","there is", tolower(text)), 
  text = gsub(" ve "," have", tolower(text)),
  text = gsub("-lrb-"," ", tolower(text)),
  text = gsub("-rrb-"," ", tolower(text)),
  text = gsub(" s "," ", tolower(text))
)

review.data$text <- as.character(review.data$text)
review.data$score <- full$Label


# Wordcloud PosNeg --------------------------------------------------------

# Divide text into positive and negative rated set
pos.text <- review.data[which(review.data$score == 1), "text"] 
neg.text <- review.data[which(review.data$score == 0), "text"]

# Split words
words_list.pos = strsplit(pos.text, " ")
words_list.neg = strsplit(neg.text, " ")

# Positive words per review
words_per_review.pos = sapply(words_list.pos, length)
uniq_words_per_review.pos = sapply(words_list.pos, function(x) length(unique(x)))
summary(words_per_review.pos)
summary(uniq_words_per_review.pos)

# Negative words per review
words_per_review.neg = sapply(words_list.neg, length)
uniq_words_per_review.neg = sapply(words_list.neg, function(x) length(unique(x)))
summary(words_per_review.neg)
summary(uniq_words_per_review.neg)

# Most frequent positve and negative words from revie dataset 
mfwp = sort(table(unlist(words_list.pos)), decreasing=TRUE)
mfwn = sort(table(unlist(words_list.neg)), decreasing=TRUE)

# Reviewing cummalative word frequencies of positive 
# and negative words to explain wordcloud imbalance
top20p = head(mfwp, 150)
top20n = head(mfwn, 150)
topp<- data.frame(top20p)
topn<- data.frame(top20n)
sum(topp$Freq)
sum(topn$Freq)

pos = paste(pos.text, collapse=" ")
neg = paste(neg.text, collapse=" ")

all = c(pos, neg)

# Create corpus
corpus = Corpus(VectorSource(all))

# Create term-document matrix
tdm = TermDocumentMatrix(corpus)

# Convert as matrix
tdm.m = as.matrix(tdm)

# Add column names
colnames(tdm.m) = c("Positive", "Negative")

# Comparison cloud
comparison.cloud(tdm.m, random.order=FALSE, 
                 colors = c("#00B2FF", "red"),
                 title.size=1.5, max.words=300)

# Commonality cloud
commonality.cloud(tdm.m, random.order=FALSE, 
                  colors = brewer.pal(8, "Dark2"),
                  title.size=1.5)

# Create high res plot image
tiff('wordcloud-max300communality.tiff', units="in", width=7, height=7, res=300)
commonality.cloud(tdm.m, random.order=FALSE, 
                  colors = brewer.pal(8, "Dark2"),
                  title.size=1.5)
dev.off()


# N-grams -----------------------------------------------------------------

### Unigram 
unigram <- data_frame(text = review.data$text) %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE) %>%
  filter(!word %in% stop_words$word)
unigram

### Bigrams
bigram <- data_frame(text = review.data$text) %>%
  unnest_tokens(ngram, text, token = "ngrams", n = 2) %>%
  count(ngram, sort = TRUE) %>%
  separate(ngram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "^\\d+"),
         !str_detect(word2, "^\\d+"),
         n > 25)
bigram

### Trigram
trigram <- data_frame(text = review.data$text) %>%
unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  count(word1, word2, word3, sort = TRUE)
trigram


# Plots of n-grams --------------------------------------------------------

# Unite n-grams
bigrams_united <- bigram %>%
  unite(bigram, word1, word2, sep = " ")
bigrams_united

trigrams_united <- trigram %>%
  unite(trigram, word1, word2, word3, sep = " ")
trigrams_united

# Plots
uni.plot <- ggplot(unigram[1:10, ], aes(x=reorder(word, +n), word, y=n))+
                geom_bar(stat = "identity")+
                coord_flip()+
                scale_y_sqrt()+
                labs(x="Unigrams", y="Frequency", title="Most common unigrams")+
                theme(text = element_text(size=15),
                      axis.text.x = element_text(size = 13))
uni.plot

big.plot <- ggplot(bigrams_united[1:10, ], aes(x=reorder(bigram, +n), y=n))+
  geom_bar(stat = "identity")+
  coord_flip()+
  scale_y_sqrt()+
  labs(x="Unigrams", y="Frequency", title="Most common unigrams")+
  theme(text = element_text(size=15),
        axis.text.x = element_text(size = 13))
big.plot

trig.plot <- ggplot(trigrams_united[1:10, ], aes(x=reorder(trigram, +n), y=n))+
  geom_bar(stat = "identity")+
  coord_flip()+
  labs(x="Unigrams", y="Frequency", title="Most common unigrams")+
  theme(text = element_text(size=15),
        axis.text.x = element_text(size = 13),
        axis.text.y = element_text(size = 13))
trig.plot

# N-gram Plot Export ----------

tiff('unigplot.tiff', units="in", width=9, height=5, res=300)
uni.plot
dev.off()
tiff('bigplot.tiff', units="in", width=9, height=5, res=300)
big.plot
dev.off()
tiff('trigplot.tiff', units="in", width=9, height=5, res=300)
trig.plot
dev.off()


# Part of Speech Analysis -------------------------------------------------

# Model <- udpipe_download_model(language = "english")
udmodel_english <- udpipe_load_model(file = 'english-ud-2.0-170801.udpipe')

txt <- as.character(review.data$text)

s <- udpipe_annotate(udmodel_english, txt)
x <- data.frame(s)

tiff('compos-movie.tiff', units="in", width=9, height=5, res=300)
stats <- txt_freq(x$upos)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = stats, col = "yellow", 
         main = "UPOS (Universal Parts of Speech)\n frequency of occurrence", 
         xlab = "Freq")
dev.off()

## NOUNS
stats <- subset(x, upos %in% c("NOUN")) 
stats <- txt_freq(stats$token)
stats$key <- factor(stats$key, levels = rev(stats$key))
tiff('compos-movie1.tiff', units="in", width=9, height=5, res=300)
barchart(key ~ freq, data = head(stats, 20), col = "cadetblue", 
         main = "Most occurring nouns", xlab = "Freq")
dev.off()

## ADJECTIVES
stats <- subset(x, upos %in% c("ADJ")) 
stats <- txt_freq(stats$token)
stats$key <- factor(stats$key, levels = rev(stats$key))
tiff('compos-movie2.tiff', units="in", width=9, height=5, res=300)
barchart(key ~ freq, data = head(stats, 20), col = "purple", 
         main = "Most occurring adjectives", xlab = "Freq")
dev.off()

## VERBS
stats <- subset(x, upos %in% c("VERB")) 
stats <- txt_freq(stats$token)
stats$key <- factor(stats$key, levels = rev(stats$key))
tiff('compos-movie3.tiff', units="in", width=9, height=5, res=300)
barchart(key ~ freq, data = head(stats, 20), col = "gold", 
         main = "Most occurring Verbs", xlab = "Freq")
dev.off()

## Using a sequence of POS tags (noun phrases / verb phrases)
x$phrase_tag <- as_phrasemachine(x$upos, type = "upos")
stats <- keywords_phrases(x = x$phrase_tag, term = tolower(x$token), 
                          pattern = "(A|N)*N(P+D*(A|N)*N)*", 
                          is_regex = TRUE, detailed = FALSE)
stats <- subset(stats, ngram > 1 & freq > 3)
stats$key <- factor(stats$keyword, levels = rev(stats$keyword))
tiff('compos-movie4.tiff', units="in", width=9, height=5, res=300)
barchart(key ~ freq, data = head(stats, 20), col = "magenta", 
         main = "Keywords - simple noun phrases", xlab = "Frequency")
dev.off()


# Creating document term matrix -------------------------------------------

### Document Term Matrix
dtm <- DocumentTermMatrix(cleanCorpus, control = list(weighting = weightTfIdf))
dtm

dtm_sparse_high <- removeSparseTerms(dtm, 0.999)
dtm_sparse_high

dtm_sparse_low <- removeSparseTerms(dtm, 0.98)
dtm_sparse_low

review_matrix_sl <- as.matrix(dtm_sparse_low)
dim(review_matrix_sl)

review_matrix_sh <- as.matrix(dtm_sparse_high)
dim(review_matrix_sh)


### Term Document Matrix
tdm <- TermDocumentMatrix(cleanCorpus, control = list(weighting = weightTfIdf))
tdm

tdm_sparse_high <- removeSparseTerms(tdm, 0.9999)
tdm_sparse_high

tdm_sparse_low <- removeSparseTerms(tdm, 0.98)
tdm_sparse_low


tdm_review_matrix_sl <- as.matrix(dtm_sparse_low)
dim(tdm_review_matrix_sl)


tdm_review_matrix_sh <- as.matrix(dtm_sparse_high)
dim(tdm_review_matrix_sh)


# Visuals -----------------------------------------------------------------

# Word counts
wc = rowSums(tdm_review_matrix_sl)

# Words above the 3rd quantile
lim = quantile(wc, probs=0.9)
good = tdm_review_matrix_sl[wc > lim,]

# Remove columns (docs) with zeroes
good = good[,colSums(good)!=0]

# Adjacency matrix
M = good %*% t(good)

# Set zeroes in diagonal
diag(M) = 0

# Plot
g = graph.adjacency(M, weighted=TRUE, mode="undirected",
                    add.rownames=TRUE)                    #max, min, directed, lower plus
# Layout
glay = layout.fruchterman.reingold(g)

# superimpose cluster structure with k-means clustering
kmg = kmeans(M, centers=8)
gk = kmg$cluster


# Select color for each cluster
gbrew = c("red", brewer.pal(8, "Dark2"))
gpal = rgb2hsv(col2rgb(gbrew))
gcols = rep("", length(gk))
for (k in 1:8) {
  gcols[gk == k] = hsv(gpal[1,k], gpal[2,k], gpal[3,k], alpha=0.5)
}

# Prepare plot elements
V(g)$size = 15
V(g)$label = V(g)$name
V(g)$degree = degree(g)
#V(g)$label.cex = 1.5 * log10(V(g)$degree)
V(g)$label.color = hsv(0, 0, 0.2, 0.55)
V(g)$frame.color = NA
V(g)$color = gcols
E(g)$color = hsv(0, 0, 0.7, 0.3)

# Plot
plot(g, layout=glay)
title("\nGraph of reviews about genetics and genomics",
      col.main="gray40", cex.main=1.5, family="serif")


# Analyzing word frequencies and term associations  -----------------------

freq <- colSums(as.matrix(dtm))

length(freq)

# Order words by decreasing frequency 
ord <- order(freq, decreasing = TRUE)

# Top 15 most frequent words
top15Freq <- freq[head(ord, 15)]
top15Freq

# Top 15 least frequent words
low15Freq <- freq[tail(ord, 15)]
low15Freq

# Other function to find most frequent terms 
findFreqTerms(tdm_sparse, lowfreq = 10)

# Find word associations for selected words
findAssocs(tdm_sparse_low, c("good", "boring", "best", "bad") , 0.100)


# Visualizations of word frequencies --------------------------------------

# Wordcloud PLot of 100 most frequent words
wordcloud(cleanCorpus, min.freq=1, max.words=100, scale=c(8,1), 
          colors=brewer.pal(8, "Dark2"), random.color=T, random.order=F)  #

# Export high res plot
tiff('P1228.tiff', units="in", width=9, height=5, res=300)
wordcloud(cleanCorpus, min.freq=1, max.words=100, scale=c(8,1), 
          colors=brewer.pal(8, "Dark2"), random.color=T, random.order=F)
dev.off()


wordcloud(cleanCorpus ,max.words =100,min.freq=100,scale=c(7,.5),colors=palette())

visFreq <- function(x) {
  freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)   
  wf <- data.frame(word=names(freq), freq=freq)  
  p <- ggplot(subset(wf, freq>1000), aes(x = reorder(word, -freq), y = freq)) +
    geom_bar(stat = "identity") + 
    theme(axis.text.x=element_text(angle=45, hjust=1))
  a <- head(freq, 10)
  b <- head(wf, 10)
  
  return(list(a, b, p))
}

visFreq(reviews_dtm_rm_sparse)

### Clustering by Term Similarity

# Deandrogram 
clusterDendro <- function(docTerm, sparseFactor, group = TRUE, group_nr){
  a <- removeSparseTerms(docTerm, sparseFactor)
  b <- dist(t(a), method="euclidian")
  c <- hclust(d=b, method="complete")
  x <- plot(c, hang=-1) 
  if (group == TRUE){
    groups <- cutree(c, k=group_nr)
    y<- rect.hclust(c, k=group_nr, border="red")
  }
  
  return(list(a, c,  x, y))
}

clusterDendro(dtm_sparse_low, 0.75, group = TRUE, 10) # calling the clusterDendro() function with the additional grouping.
