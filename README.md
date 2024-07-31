1
00:00:00,000 --> 00:00:02,770
hello guys, welcome to my video about the transformer.

2
00:00:02,910 --> 00:00:08,689
and this is actually the version 20. of my series on the transformer.

3
00:00:08,910 --> 00:00:22,079
i had a previous video in which i talked about the transformer, but the audio quality was not good and as suggested by my viewers, as the video was really had a huge success, the viewers suggested me to improve the audio quality.

4
00:00:22,079 --> 00:00:24,070
so this is why i'm doing this video.

5
00:00:25,010 --> 00:00:30,079
you don't have to watch the previous series because i would be doing basically the same things, but with some improvements.

6
00:00:30,079 --> 00:00:35,350
so i'm actually compensating from some mistakes i made or from some improvements that i could add.

7
00:00:35,890 --> 00:00:43,600
after watching this video, i suggest watching my other video about how to code a transformer model from scratch.

8
00:00:43,600 --> 00:00:49,149
so how to code the model itself, how to train it on a data and how to inference it.

9
00:00:49,329 --> 00:00:53,789
stick it with me because it's gonna be a little long journey, but for sure worth it.

10
00:00:54,250 --> 00:01:00,380
now, before we talk about the transformer, i want to 1st talk about recurrent neural networks.

11
00:01:00,380 --> 00:01:08,510
so the networks that were used before they introduced the transformer for most of the sequence to sequence jobs, tasks.

12
00:01:08,530 --> 00:01:09,909
so let's review them.

13
00:01:11,170 --> 00:01:21,519
recurrent neural networks existed a long time before the transformer and they allowed to map one sequence of input to another sequence of output.

14
00:01:21,519 --> 00:01:45,870
in this case, our input is x and we want an input sequence y. what we did before is that we split the sequence into single items, so we gave the recurrent neural network the 1st item as input, so x one, along with an initial state, usually made up of only zeros, and the recurrent neural network produced an output, let's call it y one.

15
00:01:46,450 --> 00:01:49,430
and this happened at the 1st time step.

16
00:01:49,689 --> 00:02:05,950
then we took the hidden state, this is called the hidden state of the network of the previous time step, along with the next input token, so x two, and the network had to produce the 2nd output token, y two.

17
00:02:06,250 --> 00:02:23,270
and then we did the same procedure at the 3rd time step, in which we took the hidden state of the previous time step along with the input state, the input token at the time step three, and the network has to produce the next output token, which is y three.

18
00:02:23,409 --> 00:02:32,550
if you have n tokens, you need n time steps to map a n- sequence input into an n- sequence output.

19
00:02:33,289 --> 00:02:37,680
this worked fine for a lot of tasks, but had some problems.

20
00:02:37,680 --> 00:02:38,710
let's review them.

21
00:02:40,650 --> 00:02:49,750
the problems with recurring neural networks, 1st of all, are that they are slow for long sequences because think of the process we did before.

22
00:02:49,810 --> 00:02:57,080
we have kind of like a for loop in which we do the same operation for every token in the input.

23
00:02:57,080 --> 00:03:01,509
so if you have the longer the sequence, the longer this computation.

24
00:03:01,650 --> 00:03:07,099
and this made the network not easy to train for long sequences.

25
00:03:07,099 --> 00:03:11,069
the 2nd problem was the vanishing or the exploding gradients.

26
00:03:11,169 --> 00:03:22,830
now, you may have heard these terms or expression on the internet or from other videos, but i will try to give you a brief insight on what do they mean on a practical level.

27
00:03:22,969 --> 00:03:30,990
so as you know, frameworks like pytorch, they convert our networks into a computation graph.

28
00:03:31,090 --> 00:03:34,469
so basically, suppose we have a computation graph.

29
00:03:34,650 --> 00:03:36,389
this is not a neural network.

30
00:03:36,889 --> 00:03:40,069
i will be making a computational graph that is very simple.

31
00:03:40,370 --> 00:03:44,669
it has nothing to do with the neural networks, but we'll show you the problems that we have.

32
00:03:44,969 --> 00:03:55,979
so imagine we have two inputs, x and another input, let's call it y. our computational graph 1st, let's say, multiplies these two numbers.

33
00:03:55,979 --> 00:04:07,550
so we have a 1st a function, let's call it f of x and y. that is x multiplied by y. let me multiplied.

34
00:04:07,849 --> 00:04:14,479
and the result, let's call it z is map is given to another function.

35
00:04:14,479 --> 00:04:21,430
let's call this function g of z is equal to, let's say z squared.

36
00:04:22,329 --> 00:04:30,079
what our pytorch, for example, does is that pytorch want to calculate the, usually we have a loss function.

37
00:04:30,079 --> 00:04:35,670
pytorch calculates the derivative of the loss function with respects to each weight.

38
00:04:35,689 --> 00:04:42,350
in this case, we just calculate the derivative of the g function, so the output function with respect to all of its inputs.

39
00:04:42,490 --> 00:05:04,839
so derivative of g with respect to x, let's say, is equal to the derivative of g with respect to f and multiplied by the derivative of f with respect to x. these two should kind of cancel out.

40
00:05:04,839 --> 00:05:06,750
this is called the chain rule.

41
00:05:06,970 --> 00:05:16,800
now, as you can see, the longer the chain of computation, so if we have many nodes one after another, the longer this multiplication chain.

42
00:05:16,800 --> 00:05:21,779
so here we have two because the distance from this node and this is two.

43
00:05:21,779 --> 00:05:32,750
but imagine you have 100 or 1000. now imagine this number is 0.5 and this number is 0.5 also.

44
00:05:32,930 --> 00:05:38,959
the resulting numbers when multiplied together is a number that is smaller than the two initial numbers.

45
00:05:38,959 --> 00:05:45,550
it's going to 0.25 because it's one half multiplied by one half is quarter.

46
00:05:46,410 --> 00:05:53,029
so if we have two numbers that are smaller than one and we multiply them together, they will produce an even smaller number.

47
00:05:53,089 --> 00:06:00,149
and if we have two numbers that are bigger than one and we multiply them together, they will produce a number that is bigger than both of them.

48
00:06:00,410 --> 00:06:07,949
so if we have a very long chain of computation, it eventually will either become a very big number or a very small number.

49
00:06:08,410 --> 00:06:10,379
and this is not desirable.

50
00:06:10,379 --> 00:06:19,870
1st of all, because our cpu of our gpu can only represent numbers up to a certain precision, let's say 32 bit or 64 bit.

51
00:06:20,089 --> 00:06:26,860
and if the number becomes too small, the contribution of this number to the output will become very small.

52
00:06:26,860 --> 00:06:43,629
so when the pytorch or our automatic, let's say, our framework will calculate how to adjust the weights, the weight will move very, very, very slowly because the contribution of this product is will be a very small number.

53
00:06:44,329 --> 00:06:49,149
and this means that we have the gradient is vanishing.

54
00:06:49,290 --> 00:06:52,990
or in the other case, it can explode, become very big numbers.

55
00:06:53,810 --> 00:06:54,920
and this is a problem.

56
00:06:54,920 --> 00:06:59,310
the next problem is difficulty in accessing information from a long time ago.

57
00:06:59,610 --> 00:07:00,620
what does it mean?

58
00:07:00,620 --> 00:07:09,910
it means that, as you remember from the previous slide, we saw that the 1st input token is given to the recurrent neural network along with the 1st state.

59
00:07:10,370 --> 00:07:14,439
now, we need to think that the recurrent neural network is a long graph of computation.

60
00:07:14,439 --> 00:07:16,589
it will produce a new hidden state.

61
00:07:16,689 --> 00:07:22,750
then we will use the new hidden state along with the next token to produce the next output.

62
00:07:22,769 --> 00:07:37,069
if we have a very long sequence of input sequence, the last token will have a hidden state whose contribution from the 1st token has nearly gone because of this long chain of multiplication.

63
00:07:37,170 --> 00:07:43,040
so actually the last token will not depend much on the 1st token.

64
00:07:43,040 --> 00:07:57,230
and this is also not good because for example, we know as humans that in a text, in a quite long text, the context that we saw, let's say 200 words before still relevant to the context of the current words.

65
00:07:57,329 --> 00:08:01,709
and this is something that the rnn could not map.

66
00:08:02,329 --> 00:08:05,110
and this is why we have the transformer.

67
00:08:05,569 --> 00:08:10,589
so the transformer solved these problems with the recurrent neural networks and we will see how.

68
00:08:11,569 --> 00:08:17,500
the structure of the transformer, we can divide into two macro blocks.

69
00:08:17,500 --> 00:08:21,870
the 1st macro block is called encoder, and it's this part here.

70
00:08:22,449 --> 00:08:26,829
the 2nd macro block is called a decoder and it's the 2nd part here.

71
00:08:27,930 --> 00:08:35,230
the 3rd part here, you see on the top, it's just a linear layer and we will see why it's there and what it is function.

72
00:08:35,649 --> 00:08:48,200
so, and the two layers, so the encoder and the decoder are connected by this connection you can see here, in which some output of the encoder is sent as input to the decoder.

73
00:08:48,200 --> 00:08:50,259
and we will also see how.

74
00:08:50,259 --> 00:08:57,509
let's start 1st of all with some notations that i will be using during my explanation.

75
00:08:57,529 --> 00:09:02,149
and you should be familiar with this notation also to review some maths.

76
00:09:02,169 --> 00:09:06,559
so the 1st thing we should be familiar with is matrix multiplication.

77
00:09:06,559 --> 00:09:12,789
so imagine we have an input matrix, which is a sequence of, let's say, words.

78
00:09:12,889 --> 00:09:17,500
so sequence by d model, and we will see why it's called sequence by d model.

79
00:09:17,500 --> 00:09:25,710
so imagine we have a matrix that is six by 512 in which each row is a word.

80
00:09:27,009 --> 00:09:31,700
and this word is not made of characters, but by 512 numbers.

81
00:09:31,700 --> 00:09:35,940
so each word is represented by 512 numbers.

82
00:09:35,940 --> 00:09:36,940
okay, like this.

83
00:09:36,940 --> 00:09:43,429
imagine you have 512 of them along this row, 512 along this other row, et cetera, et cetera.

84
00:09:43,450 --> 00:09:45,059
one, two, three, four, five.

85
00:09:45,059 --> 00:09:46,840
so we need another one here.

86
00:09:46,840 --> 00:09:47,429
okay.

87
00:09:47,610 --> 00:10:05,110
the 1st word, we will call it a, the 2nd b, the c, d, e and f. if we multiply this matrix by another matrix, let's say the transpose of this matrix, so it's a matrix where the rows becomes columns.

88
00:10:05,889 --> 00:10:14,309
so three, four, five and six.

89
00:10:15,610 --> 00:10:35,159
this word will be here b, c, d, e and f. and then we have 512 numbers along each column because before we had them on the rows, now they will become on the column.

90
00:10:35,159 --> 00:10:39,029
so here we have the 512 number, etc, etc.

91
00:10:39,889 --> 00:10:44,830
this is a matrix that is 512 by six.

92
00:10:44,889 --> 00:10:46,870
so let me add some brackets here.

93
00:10:47,129 --> 00:10:55,639
if we multiply them, we will get a new matrix that is, we cancel the inner dimensions and we get the outer dimensions.

94
00:10:55,639 --> 00:10:57,470
so it will become six by six.

95
00:10:58,009 --> 00:11:00,559
so it will be six rows by six rows.

96
00:11:00,559 --> 00:11:01,830
so let's draw it.

97
00:11:02,409 --> 00:11:05,580
how do we calculate the values of this output matrix?

98
00:11:05,580 --> 00:11:07,110
this is six by six.

99
00:11:08,409 --> 00:11:13,620
this is the dot product of the 1st row with the 1st column.

100
00:11:13,620 --> 00:11:20,960
so this is a multiplied by a. the 2nd value is the 1st row with the 2nd column.

101
00:11:20,960 --> 00:11:28,220
the 3rd value is the 1st row with the 3rd column until the last column.

102
00:11:28,220 --> 00:11:30,750
so a multiplied by f, etc.

103
00:11:30,889 --> 00:11:32,539
what is the dot product?

104
00:11:32,539 --> 00:11:37,320
is basically you take the 1st number of the 1st row.

105
00:11:37,320 --> 00:11:41,379
so here we have 512 numbers, here we have 512 numbers.

106
00:11:41,379 --> 00:11:47,789
so you take the 1st number of the 1st row and the 1st number of the 1st column, you multiply them together.

107
00:11:48,049 --> 00:11:54,230
2nd value of the 1st row, 2nd value of the 1st column, you multiply them together.

108
00:11:54,490 --> 00:11:57,700
and then you add all these numbers together.

109
00:11:57,700 --> 00:12:11,840
so it will be, let's say, this number multiplied by this plus this number multiplied by this plus this number multiplied by this plus this number multiplied by this plus you sum all these numbers together.

110
00:12:11,840 --> 00:12:20,700
and this is the a dot product a. so we should be familiar with this notation because i will be using it a lot in the next slides.

111
00:12:20,700 --> 00:12:26,070
let's start our journey with of the transformer by looking at the encoder.

112
00:12:26,570 --> 00:12:30,980
so the encoder starts with the input embeddings.

113
00:12:30,980 --> 00:12:32,629
so what is an input embedding?

114
00:12:33,409 --> 00:12:35,990
1st of all, let's start with our sentence.

115
00:12:36,129 --> 00:12:40,200
we have a sentence of, in this case, six words.

116
00:12:40,200 --> 00:12:42,120
what we do is we tokenize it.

117
00:12:42,120 --> 00:12:44,549
we transform the sentence into tokens.

118
00:12:44,570 --> 00:12:46,059
what does it mean to tokenize?

119
00:12:46,059 --> 00:12:48,429
we split them into single words.

120
00:12:48,929 --> 00:12:54,340
it is not necessary to always split the sentence using single words.

121
00:12:54,340 --> 00:13:00,580
we can even split the sentence in smaller parts that are even smaller than a single word.

122
00:13:00,580 --> 00:13:09,950
so we could even split this sentence into, let's say, 20 tokens by using the each, by splitting each word into multiple words.

123
00:13:10,129 --> 00:13:17,820
this is usually done in most modern transformer models, but we will not be doing it.

124
00:13:17,820 --> 00:13:20,070
otherwise it's really difficult to visualize.

125
00:13:20,129 --> 00:13:27,389
so let's suppose we have this input sentence and we split into tokens and each token is a single word.

126
00:13:27,690 --> 00:13:32,830
the next step we do is we map these words into numbers.

127
00:13:33,090 --> 00:13:38,159
and these numbers represent the position of these words in our vocabulary.

128
00:13:38,159 --> 00:13:43,750
so imagine we have a vocabulary of all the possible words that appear in our training set.

129
00:13:44,210 --> 00:13:47,639
each word will occupy a position in this vocabulary.

130
00:13:47,639 --> 00:13:55,789
so for example, the word will occupy the position 105, the word cat will occupy the position 6500, etc.

131
00:13:56,450 --> 00:14:03,389
and as you can see, this cat here has the same number as this cat here, because they occupy the same position in the vocabulary.

132
00:14:03,929 --> 00:14:21,590
we take these numbers, which are called input ids, and we map them into a vector of size 512. this vector is a vector made of 512 numbers and we always map the same word to always the same embedding.

133
00:14:22,169 --> 00:14:25,990
however, this number is not fixed.

134
00:14:26,049 --> 00:14:28,299
it's a parameter for our model.

135
00:14:28,299 --> 00:14:35,259
so our model will learn to change these numbers in such a way that it represents the meaning of the word.

136
00:14:35,259 --> 00:14:43,059
so the input ids never change because of our vocabulary is fixed, but the embedding will change along with the training process of the model.

137
00:14:43,059 --> 00:14:48,600
so the embeddings numbers will change according to the needs of the loss function.

138
00:14:48,600 --> 00:15:01,620
so the input embedding are basically mapping our single word into an embedding of size 512. and we call this quantity 512 d model because it's the same name that it's also used in the paper.

139
00:15:01,620 --> 00:15:03,029
attention is all you need.

140
00:15:04,970 --> 00:15:09,149
let's look at the next layer of the encoder, which is the positional encoding.

141
00:15:10,450 --> 00:15:12,389
so what is positional encoding?

142
00:15:13,330 --> 00:15:20,789
what we want is that each word should carry some information about its position in the sentence.

143
00:15:20,970 --> 00:15:32,779
because now we built a metrics of words that are embeddings, but they don't convey any information about how, where that particular word is inside the sentence.

144
00:15:32,779 --> 00:15:35,309
and this is the job of the positional encoding.

145
00:15:35,450 --> 00:15:44,679
so what we do, we want the model to treat words that appear close to each other as close and words that are distant as distant.

146
00:15:44,679 --> 00:15:50,639
so we want the model to see this information about the spatial information that we see with our eyes.

147
00:15:50,639 --> 00:15:54,279
so for example, when we see this sentence, what is positional encoding?

148
00:15:54,279 --> 00:16:01,629
we know that the word what is more far from the word is compared to encoding.

149
00:16:02,210 --> 00:16:07,340
because we have this partial information given by our eyes, but the model cannot see this.

150
00:16:07,340 --> 00:16:14,669
so we need to give some information to the model about how the words are spatially distributed inside of the sentence.

151
00:16:15,570 --> 00:16:21,440
and we want the positional encoding to represent a pattern that the model can learn.

152
00:16:21,440 --> 00:16:22,830
and we will see how.

153
00:16:24,850 --> 00:16:28,590
imagine we have our original sentence, your cat is a lovely cat.

154
00:16:28,649 --> 00:16:45,179
what we do is we 1st convert into embeddings using the previous layer, so the input embeddings, and these are embeddings of size 512. then we create some special vectors called positional encoding vectors that we add to these embeddings.

155
00:16:45,179 --> 00:16:53,100
so this vector we see here in red is a vector of size 512, which is not learned.

156
00:16:53,100 --> 00:16:57,059
it's computed once and not learned along with the training process.

157
00:16:57,059 --> 00:16:58,269
it's fixed.

158
00:16:58,450 --> 00:17:03,789
and this word, this vector represents the position of the word inside of the sentence.

159
00:17:04,289 --> 00:17:17,519
and this should give us an output that is a vector of size again 512 because we are summing this number with this number, this number with this number.

160
00:17:17,519 --> 00:17:20,710
so the 1st dimension with the 1st dimension, the 2nd dimension.

161
00:17:20,809 --> 00:17:26,109
so we will get a new vector of the same size of the input vectors.

162
00:17:26,170 --> 00:17:29,039
how are these positions in both embedding calculated?

163
00:17:29,039 --> 00:17:29,630
let's see.

164
00:17:30,809 --> 00:17:32,579
imagine we have a smaller sentence.

165
00:17:32,579 --> 00:17:38,869
let's say your cat is and you may have seen the following expressions from the paper.

166
00:17:39,210 --> 00:17:55,940
what we do is we create a vector of size d model, so 512, and for each position in this vector, we calculate the value using these two expressions, using these arguments.

167
00:17:55,940 --> 00:18:00,980
so the 1st argument indicates the position of the word inside of the sentence.

168
00:18:00,980 --> 00:18:13,519
so the word your occupies the position zero and we use the for the even dimension, so the zero, the two, the four, the 510 etc.

169
00:18:13,519 --> 00:18:21,269
we use the 1st expression, so the sign and for the odd positions of this vector, we use the 2nd expression.

170
00:18:22,289 --> 00:18:25,680
and we do this for all the words inside of the sentence.

171
00:18:25,680 --> 00:18:33,470
so this particular embedding is calculated pe of ten because it's the 1st word embedding zero.

172
00:18:33,690 --> 00:18:48,940
so this one represents the argument pause and this zero represents the argument to i. and pe of one means that the 1st word dimension one.

173
00:18:48,940 --> 00:18:57,349
so we will use the cosine given the position one and the two i will be equal to two i plus one will be equal to one.

174
00:18:58,609 --> 00:19:02,430
and we do this for this 3rd word, etc.

175
00:19:03,009 --> 00:19:08,069
if we have another sentence, we will not have different positional encodings.

176
00:19:08,569 --> 00:19:22,630
we will have the same vectors, even for different sentences, because the positional encoding are computed once and reused for every sentence that our model will see during inference or training.

177
00:19:22,769 --> 00:19:29,119
so we only compute the positional encoding once when we create the model, we save them and then we reuse them.

178
00:19:29,119 --> 00:19:34,390
we don't need to compute it every time we feed a sentence to the model.

179
00:19:36,289 --> 00:19:42,309
so why the authors chose the cosine and the sine functions to represent positional encodings?

180
00:19:42,329 --> 00:19:45,230
because let's watch the plot of these two functions.

181
00:19:46,529 --> 00:19:54,759
you can see the plot is by position, so the position of the word inside of the sentence and this depth is the dimension along the vector.

182
00:19:54,759 --> 00:19:58,430
so the two i that you saw before in the previous expressions.

183
00:19:59,170 --> 00:20:02,759
and if we plot them, we can see as humans, a pattern here.

184
00:20:02,759 --> 00:20:07,019
and we hope that the model can also see this path.

185
00:20:07,019 --> 00:20:10,910
okay, the next layer of the encoder is the multi- head attention.

186
00:20:12,210 --> 00:20:19,960
we will not go inside of the multi- head attention 1st, we will 1st visualize the single head attention.

187
00:20:19,960 --> 00:20:22,660
so the self- attention with a single head.

188
00:20:22,660 --> 00:20:24,029
and let's do it.

189
00:20:24,569 --> 00:20:26,259
so what is self- attention?

190
00:20:26,259 --> 00:20:31,630
self- attention is a mechanism that existed before they introduced the transformer.

191
00:20:31,849 --> 00:20:37,109
the authors of the transformer just changed it into a multi- head attention.

192
00:20:37,410 --> 00:20:39,869
so how did the self- attention work?

193
00:20:40,930 --> 00:20:45,789
the self- attention allows the model to relate words to each other.

194
00:20:46,210 --> 00:20:51,549
okay, so we had the input embeddings that capture the meaning of the word.

195
00:20:51,730 --> 00:20:59,069
then we had the positional encoding that give the information about the position of the word inside of the sentence.

196
00:20:59,170 --> 00:21:03,470
now we want this self- attention to relate words to each other.

197
00:21:03,970 --> 00:21:25,000
now imagine we have an input sequence of six word with the d model of size 512, which can be represented as a matrix that we will call q, k and v. so our q, k and v is a same matrix, are the same matrix representing the input.

198
00:21:25,000 --> 00:21:41,640
so the input of six words with the dimension of 512. so each word is represented by a vector of size 512. we basically apply this formula we saw here from the paper to calculate the attention, the self attention in this case.

199
00:21:41,640 --> 00:21:42,640
why self attention?

200
00:21:42,640 --> 00:21:48,660
because it's the each word in the sentence related to other words in the same sentence.

201
00:21:48,660 --> 00:21:50,630
so it's self attention.

202
00:21:51,730 --> 00:21:56,240
so we start with our q matrix, which is the input sentence.

203
00:21:56,240 --> 00:21:57,779
so let's visualize it, for example.

204
00:21:57,779 --> 00:22:03,339
so we have six rows and on this, on the columns, we have 512 columns.

205
00:22:03,339 --> 00:22:10,029
now, they are really difficult to draw, but let's say we have 512 columns and here we have six.

206
00:22:10,569 --> 00:22:11,190
okay?

207
00:22:11,490 --> 00:22:17,700
now what we do according to this formula, we multiply it by the same sentence but transposed.

208
00:22:17,700 --> 00:22:28,630
so the transposed of the k, which is again the same input sequence, we divide it by the square root of 512, and then we apply the softmax.

209
00:22:28,849 --> 00:22:45,750
the output of this, as we saw before in the initial matrix notations, we saw that when we multiply six by 512 with another matrix that is 512 by six, we obtain a new matrix that is six by six.

210
00:22:45,970 --> 00:22:57,390
and each value in this matrix represents the dot product of the 1st row with the 1st column, this represents the dot product of the 1st row with the 2nd column, etc.

211
00:22:58,490 --> 00:23:02,829
the values here are actually randomly generated, so don't concentrate on the values.

212
00:23:02,849 --> 00:23:10,019
what you should notice is that the softmax makes all these values in such a way that they sum up to one.

213
00:23:10,019 --> 00:23:14,190
so this row, for example, here, sums up to one.

214
00:23:14,289 --> 00:23:17,920
this other row also sums up to one, et cetera, et cetera.

215
00:23:17,920 --> 00:23:26,349
and this value we see here, it's the dot product of the 1st word with the embedding of the word itself.

216
00:23:26,650 --> 00:23:34,670
this value here, it's the dot product of the embedding of the word your with the embedding of the word cat.

217
00:23:34,970 --> 00:23:42,549
and this value here is the dot product of the word, the embedding of the word your with the embedding of the word is.

218
00:23:43,410 --> 00:23:52,670
the next thing and this value represents somehow a score, that how intense is the relationship between one word and another.

219
00:23:52,690 --> 00:23:55,019
let's go ahead with the formula.

220
00:23:55,019 --> 00:24:06,029
so for now we just multiplied q by k divided by the square root of dk applied to the softmax, but we didn't multiply by v. so let's go forward.

221
00:24:06,210 --> 00:24:29,309
we multiply this matrix by v and we obtain a new matrix which is six by 512. so if we multiply a matrix that is six by six with another that is six by 512, we get a new matrix that is six by 512. and one thing you should notice is that the dimension of this matrix is exactly the dimension of the initial matrix from which we started.

222
00:24:29,769 --> 00:24:31,670
this, what does it mean?

223
00:24:32,089 --> 00:24:35,140
that we obtain a new matrix that is six rows.

224
00:24:35,140 --> 00:24:43,660
so let's say six rows with 512 columns in which each, these are our words.

225
00:24:43,660 --> 00:25:13,509
so we have six words and each word has an embedding of dimension 512. so now this embedding here represents not only the meaning of the word, which was given by the input embedding, not only the position of the word, which was added by the positional encoding, but now somehow this special embedding, so these values represent a special embedding that also captures the relationship of this particular word with all the other words.

226
00:25:13,849 --> 00:25:26,950
and this particular embedding of this word here also captures not only its meaning, not only its position inside of the sentence, but also the relationship of this word with all the other words.

227
00:25:27,210 --> 00:25:33,710
i want to remind you that this is not the multi- head attention, we are just watching the self attention, so one head.

228
00:25:34,769 --> 00:25:38,349
we will see later how this becomes the multi- head attention.

229
00:25:40,930 --> 00:25:44,430
self- attention has some properties that are very desirable.

230
00:25:44,769 --> 00:25:47,660
1st of all, it's permutation invariant.

231
00:25:47,660 --> 00:25:49,880
what does it mean to be permutation invariant?

232
00:25:49,880 --> 00:25:57,240
it means that if we have a matrix, let's say 1st we had a matrix of six words.

233
00:25:57,240 --> 00:25:59,240
in this case, let's say just four words.

234
00:25:59,240 --> 00:26:18,880
so a, b, c and d. and suppose by applying the formula before, this produces this particular matrix in which there is new special embedding for the word a and new special embedding for the word b and new special embedding for the word c and d. so let's call it a prime, b prime, c prime, d prime.

235
00:26:18,880 --> 00:26:26,779
if we change the position of these two rows, the values will not change, the position of the output will change accordingly.

236
00:26:26,779 --> 00:26:36,180
so the values of b prime will not change, it will just change in the position and also the c will also change position, but the values in each vector will not change.

237
00:26:36,180 --> 00:26:38,150
and this is a desirable properties.

238
00:26:38,410 --> 00:26:41,900
self- attention as of now requires no parameters.

239
00:26:41,900 --> 00:26:45,680
i mean, i didn't introduce any parameter that is learned by the model.

240
00:26:45,680 --> 00:26:51,029
i just took the initial sentence of, in this case, six words.

241
00:26:51,329 --> 00:27:09,230
we multiplied it by itself, we divide it by a fixed quantity, which is the square root of 512, and then we apply the softmax, which is not introducing any parameter, so for now the self- attention rate didn't require any parameter except for the embedding of the words.

242
00:27:09,769 --> 00:27:13,109
this will change later when we introduce the multi- head attention.

243
00:27:14,609 --> 00:27:34,109
also, we expect because the each value in the self- attention in the softmax matrix is a dot product of the word embedding with itself and the other words, we expect the values along the diagonal to be the maximum because it's the dot product of each word with itself.

244
00:27:34,930 --> 00:28:05,069
and there is another property of this matrix, that is, before we apply the softmax, if we replace the value in this matrix, suppose we don't want the word your and cat to interact with each other, or we don't want the word, let's say, is and the lovely to interact with each other, what we can do is before we apply the softmax, we can replace this value with minus infinity and also this value with minus infinity.

245
00:28:06,490 --> 00:28:11,470
and when we apply the softmax, the softmax will replace minus infinity with zero.

246
00:28:12,410 --> 00:28:22,240
because as you remember, the softmax is e to the power of x. if x is going to minus infinity, e to the power of minus infinity will become very, very close to zero.

247
00:28:22,240 --> 00:28:24,029
so basically zero.

248
00:28:25,250 --> 00:28:30,500
this is a desirable property that we will use in the decoder of the transformer.

249
00:28:30,500 --> 00:28:34,039
now let's have a look at what is a multi- head attention.

250
00:28:34,039 --> 00:28:40,099
so what we just saw was the self- attention and we want to convert it into a multi- head attention.

251
00:28:40,099 --> 00:28:45,359
you may have seen these expressions from the paper, but don't worry, i will explain them one by one.

252
00:28:45,359 --> 00:28:46,390
so let's go.

253
00:28:47,049 --> 00:28:48,859
imagine we have our encoder.

254
00:28:48,859 --> 00:29:02,990
so we are on the encoder side of the transformer and we have our input sentence, which is, let's say six by 512. so six word by 512 is the size of the embedding of each word.

255
00:29:03,009 --> 00:29:05,980
in this case, i call it sequence by d- model.

256
00:29:05,980 --> 00:29:12,190
so sequence is the sequence length, as you can see on the legend in the bottom left of the slide.

257
00:29:12,210 --> 00:29:24,799
and the d- model is the size of the embedding vector, which is 512. what we do, just like the picture shows, we take this input and we make four copies of it.

258
00:29:24,799 --> 00:29:26,430
one will be sent.

259
00:29:27,930 --> 00:29:31,869
one will be sent along this connection we can see here.

260
00:29:32,130 --> 00:29:37,500
and three will be sent to the multi- head attention with three respective names.

261
00:29:37,500 --> 00:29:42,920
so it's the same input that becomes three matrices that are equal to input.

262
00:29:42,920 --> 00:29:46,630
one is called query, one is called key and one is called value.

263
00:29:46,890 --> 00:29:50,240
so basically we are taking this input and making three copies of it.

264
00:29:50,240 --> 00:29:54,430
one we call q, k and b. they have of course, the same dimension.

265
00:29:54,930 --> 00:29:56,839
what does the multi- head attention do?

266
00:29:56,839 --> 00:30:05,670
1st of all, it multiplies these three matrices by three parameter matrices called wq, wk and wv.

267
00:30:06,809 --> 00:30:10,309
these matrices have dimension d- model by d- model.

268
00:30:10,369 --> 00:30:21,359
so if we multiply a matrix that is sequence by d- model with another one that is d- model by d- model, we get a new matrix as output that is sequence by d- model.

269
00:30:21,359 --> 00:30:25,230
so basically the same dimension as the starting matrix.

270
00:30:25,609 --> 00:30:29,869
and we will call them q prime, k prime and v prime.

271
00:30:30,329 --> 00:30:35,319
our next step is to split these matrices into smaller matrices.

272
00:30:35,319 --> 00:30:36,190
let's see how.

273
00:30:36,890 --> 00:30:43,750
we can split this matrix q prime by the sequence dimension or by the d- model dimension.

274
00:30:44,410 --> 00:30:48,720
in the multi- head attention, we always split by the d- model dimension.

275
00:30:48,720 --> 00:30:56,630
so every head will see the full sentence, but a smaller part of the embedding of each word.

276
00:30:57,009 --> 00:31:06,269
so if we have an embedding of, let's say 512, it will become smaller embeddings of 512 divided by four.

277
00:31:06,410 --> 00:31:08,819
and we call this quantity dk.

278
00:31:08,819 --> 00:31:13,400
so dk is d model divided by h, where h is the number of heads.

279
00:31:13,400 --> 00:31:16,029
in our case we have h equal to four.

280
00:31:17,289 --> 00:31:27,230
we can calculate the attention between these smaller matrices, so q one, k one and v one using the expression taken from the paper.

281
00:31:28,170 --> 00:31:35,789
and this will result into a small matrix called head one, head two, head three and head four.

282
00:31:36,210 --> 00:31:42,109
the dimension of head one up to head four is sequence by dv.

283
00:31:42,849 --> 00:31:44,019
what is dv?

284
00:31:44,019 --> 00:31:46,519
is basically it's equal to dk.

285
00:31:46,519 --> 00:31:55,029
it's just called dv because the last multiplication is done by v. and in the paper they call it dv, so i am also sticking to the same names.

286
00:31:55,769 --> 00:32:09,619
our next step is to combine these matrices, these small heads, by concatenating them along the dv dimension, just like the paper says.

287
00:32:09,619 --> 00:32:26,779
so we concut all this head together and we get a new matrix that is sequence by h multiplied by dv, where h multiplied by dv, as we know, dv is equal to decay, so h multiplied by dv is equal to d model.

288
00:32:26,779 --> 00:32:29,880
so we get back the initial shape.

289
00:32:29,880 --> 00:32:33,109
so it's sequence by d model here.

290
00:32:34,769 --> 00:32:39,509
the next step is to multiply the result of this concatenation by wo.

291
00:32:39,970 --> 00:32:48,599
and wo is a matrix that is h multiplied by dv, so d model with the other dimension being d- model.

292
00:32:48,599 --> 00:32:55,670
and the result of this is a new matrix that is the result of the multi- head attention, which is sequenced by d- model.

293
00:32:56,690 --> 00:33:15,480
so the multi- head attention, instead of calculating the attention between these matrices here, so q prime, k prime and v prime, splits them along the d model dimension into smaller matrices and calculates the attention between these smaller matrices.

294
00:33:15,480 --> 00:33:23,309
so each head is watching the full sentence, but as different aspects of the embedding of each word.

295
00:33:23,369 --> 00:33:24,750
why we want this?

296
00:33:24,769 --> 00:33:30,299
because we want each head to watch different aspects of the same word.

297
00:33:30,299 --> 00:33:42,509
for example, in the chinese language, but also in other languages, one word may be a noun in some cases, maybe a verb in some other cases, maybe an adverb in some other cases, depending on the context.

298
00:33:43,130 --> 00:33:58,349
so what we want is that one head maybe learns to relate that word as a noun, another head maybe learns to relate that word as a verb, and another head learns to relate that verb as an adjective or adverb.

299
00:33:58,730 --> 00:34:02,390
so this is why we want multi- head attention.

300
00:34:02,890 --> 00:34:11,309
now, you may also have seen online that the attention can be visualized, and i will show you how.

301
00:34:11,690 --> 00:34:29,619
when we calculate the attention between the q and the k matrices, so when we do this operation, so the soft max of q multiplied by the k divided by the square root of dk, we get a new matrix, just like we saw before, which is sequence by sequence.

302
00:34:29,619 --> 00:34:36,429
and this represents a score that represents the intensity of the relationship between the two words.

303
00:34:36,929 --> 00:34:48,319
we can visualize this and this will produce a visualization similar to this one, which i took from the paper, in which we see how the all the heads work.

304
00:34:48,320 --> 00:34:59,260
so for example, if we concentrate on this work making, this word here, we can see that making is related to the word difficult, so this word here, by different heads.

305
00:34:59,260 --> 00:35:02,630
so the blue head, the red head and the green head.

306
00:35:03,170 --> 00:35:09,480
but let's say the violet head is not relating these two words together.

307
00:35:09,480 --> 00:35:13,909
so making it difficult is not related by the violet or the pink head.

308
00:35:14,690 --> 00:35:26,219
the violet head or the pink head, they are relating the word making to other words, for example, to this word 2009. why this is the case?

309
00:35:26,219 --> 00:35:37,110
because maybe this pink head could see the part of the embedding that these other heads could not see that made this interaction possible between these two words.

310
00:35:40,809 --> 00:35:46,750
you may be also wondering why these three matrices are called query keys and values.

311
00:35:47,409 --> 00:35:53,179
okay, the terms come from the database terminology or from the python- like dictionaries.

312
00:35:53,179 --> 00:35:58,000
but i would also like to give my interpretation of my own, making a very simple example.

313
00:35:58,000 --> 00:36:02,150
i think it's quite easy to understand.

314
00:36:03,250 --> 00:36:09,269
so imagine we have a python- like dictionary or a database in which we have keys and values.

315
00:36:09,929 --> 00:36:16,280
the keys are the category of movies and the values are the movies belonging to that category.

316
00:36:16,280 --> 00:36:18,909
in my case, i just put one value.

317
00:36:19,570 --> 00:36:26,510
so we have romantics category which includes titanic, we have action movies that include the dark knight, etc.

318
00:36:26,849 --> 00:36:31,909
imagine we also have a user that makes a query and the query is love.

319
00:36:32,610 --> 00:36:57,860
because we are in the transformer world, all these words actually are represented by embeddings of size 512. so what our transformer will do, he will convert this word love into an embedding of 512. all these queries and values are already embeddings of 512 and it will calculate the dot product between the query and all the keys, just like the formula.

320
00:36:57,860 --> 00:37:06,260
so as you remember, the formula is a soft max of query multiplied by the transpose of the keys divided by the square root of the model.

321
00:37:06,260 --> 00:37:10,349
so we are doing the dot product of all the queries with all the keys.

322
00:37:10,809 --> 00:37:14,590
in this case, the word love with all the keys, one by one.

323
00:37:15,010 --> 00:37:23,949
and this will result in a score that will amplify some values or not amplify other values.

324
00:37:25,610 --> 00:37:41,219
in this case, our embedding may be in such a way that the word love and romantic are related to each other, the word love and comedy are also related to each other, but not so intensively like the word love and romantic.

325
00:37:41,219 --> 00:37:45,590
so it's more, how to say, less strong relationship.

326
00:37:45,690 --> 00:37:52,469
but maybe the word horror and love are not related at all, so maybe their softmax score is very close to zero.

327
00:37:56,210 --> 00:38:01,989
our next layer in the encoder is the add and norm.

328
00:38:02,130 --> 00:38:05,719
and to introduce the add and norm, we need the layer normalization.

329
00:38:05,719 --> 00:38:08,030
so let's see what is the layer normalization.

330
00:38:09,170 --> 00:38:15,219
layer normalization is a layer that, okay, let's make a practical example.

331
00:38:15,219 --> 00:38:18,349
imagine we have a batch of n items.

332
00:38:18,369 --> 00:38:21,230
in this case, n is equal to three.

333
00:38:22,289 --> 00:38:24,599
item one, item two, item three.

334
00:38:24,599 --> 00:38:27,789
each of these items will have some features.

335
00:38:27,809 --> 00:38:36,840
it could be an embedding, so for example, it could be a feature of vector of size 512, but it could be a very big matrix of thousands of features.

336
00:38:36,840 --> 00:38:37,789
doesn't matter.

337
00:38:37,969 --> 00:38:44,349
what we do is we calculate the mean and the variance of each of these items independently from each other.

338
00:38:44,769 --> 00:38:49,469
and we replace each value with another value that is given by this expression.

339
00:38:49,570 --> 00:38:55,309
so basically we are normalizing so that the new values are all in the range zero to one.

340
00:38:56,849 --> 00:39:23,469
actually, we also multiply this new value with a parameter called gamma and then we add another parameter called beta and this gamma and beta are learnable parameters and the model should learn to multiply and add these parameters so as to amplify the value that it wants to be amplified and not amplify the value that it doesn't want to be amplified.

341
00:39:25,010 --> 00:39:29,349
so we don't just normalize, we actually introduce some parameters.

342
00:39:29,969 --> 00:39:38,800
and i found a really nice visualization from paperswithcode dot com in which we see the difference between batch norm and layer norm.

343
00:39:38,800 --> 00:39:57,199
so as we can see in the layer normalization, we are calculating if n is the batch dimension, we are calculating all the values belonging to one item in the batch, while in the batch norm, we are calculating the same feature for all the batch.

344
00:39:57,199 --> 00:39:59,679
so for all the items in the batch.

345
00:39:59,679 --> 00:40:12,190
so we are mixing, let's say, values from different items of the batch, while in the layer normalization, we are treating each item in the batch independently, which will have its own mean and its own variance.

346
00:40:14,610 --> 00:40:16,269
let's look at the decoder now.

347
00:40:16,650 --> 00:40:21,710
now, in the encoder we saw the input embeddings.

348
00:40:22,210 --> 00:40:26,949
in this case they are called output embeddings, but the underlying working is the same.

349
00:40:27,570 --> 00:40:35,110
here also we have the positional encoding and they are also the same as the encoder.

350
00:40:35,610 --> 00:40:40,630
the next layer is the masked multi- head attention and we will see it now.

351
00:40:40,889 --> 00:40:44,710
we also have the multi- head attention here with the.

352
00:40:45,409 --> 00:41:07,309
here we should see that there is the encoder here that produces the output and is sent to the decoder in the forms of keys and values, while the query, so this connection here is the query coming from the decoder.

353
00:41:07,889 --> 00:41:16,119
so in this multi- head attention, it's not a self attention anymore, it's a cross attention because we are taking two sentences.

354
00:41:16,119 --> 00:41:33,110
one is sent from the encoder side, so let's write encoder, in which we provide the output of the encoder and we use it as keys and values, while the output of the masked multi- head attention is used as the query in this multi- head attention.

355
00:41:34,369 --> 00:41:40,760
and the masked multi- head attention is a self- attention of the input sentence of the decoder.

356
00:41:40,760 --> 00:41:53,590
so we take the input sentence of the decoder, we transform into embeddings, we add the positional encoding, we give it to this multi- head attention in which the query key and values are the same input sequence.

357
00:41:53,690 --> 00:41:55,710
we do the add and norm.

358
00:41:55,849 --> 00:42:04,989
then we send this as the queries of the multi- head attention while the keys and the values are coming from the encoder, then we do the add and norm.

359
00:42:05,329 --> 00:42:10,070
i will not be showing the feed forward which is just a fully connected layer.

360
00:42:10,610 --> 00:42:18,380
we then send the output of the feed forward to the add and norm and finally to the linear layer, which we will see later.

361
00:42:18,380 --> 00:42:24,670
so let's have a look at the masked multi- head attention and how it differs from a normal multi- head attention.

362
00:42:26,050 --> 00:42:30,900
what we want, our goal is that we want to make the model causal.

363
00:42:30,900 --> 00:42:37,579
it means that the output at a certain position can only depend on the words on the previous position.

364
00:42:37,579 --> 00:42:41,219
so the model must not be able to see future words.

365
00:42:41,219 --> 00:42:42,789
how can we achieve that?

366
00:42:43,809 --> 00:42:52,300
as you saw, the output of the soft max in the attention calculation formula is this metric, sequence by sequence.

367
00:42:52,300 --> 00:43:07,670
if we want to hide the interaction of some words with other words, we delete this value and we replace it with minus infinity before we apply the softmax, so that the softmax will replace this value with zero.

368
00:43:08,170 --> 00:43:11,960
and we do this for all the interaction that we don't want.

369
00:43:11,960 --> 00:43:15,000
so we don't want your to watch future words.

370
00:43:15,000 --> 00:43:18,760
so we don't want your to watch cat is a lovely cat.

371
00:43:18,760 --> 00:43:25,639
and we don't want the word cat to watch future words, but only all the words that come before it or the word itself.

372
00:43:25,639 --> 00:43:28,389
so we don't want this, this, this, this.

373
00:43:28,889 --> 00:43:31,389
also the same for the other words, etc.

374
00:43:32,610 --> 00:43:41,699
so we can see that we are replacing all the word, all these values here that are above this diagonal here.

375
00:43:41,699 --> 00:43:54,309
so this is the principal diagonal of the matrix and we want all the values that are above this diagonal to be replaced with minus infinity so that so that the soft max will replace them with zero.

376
00:43:54,449 --> 00:44:00,590
let's see in which stage of the multi- head attention this mechanism is introduced.

377
00:44:00,690 --> 00:44:12,840
so when we calculate the attention between these molar matrices, so q one, k one and v one, before we apply the soft max, we replace these values.

378
00:44:12,840 --> 00:44:16,989
so this one, this one, this one, this one, this one, etc.

379
00:44:17,130 --> 00:44:18,500
with minus infinity.

380
00:44:18,500 --> 00:44:26,019
then we apply the softmax and then the softmax will take care of transforming these values into zeros.

381
00:44:26,019 --> 00:44:30,190
so basically, we don't want these words to interact with each other.

382
00:44:31,010 --> 00:44:38,960
and if we don't want this interaction, the model will learn to not make them interact because the model will not get any information from this interaction.

383
00:44:38,960 --> 00:44:41,519
so it's like this word cannot interact.

384
00:44:41,519 --> 00:44:45,989
now let's look at how the inference and training works for a transformer model.

385
00:44:46,449 --> 00:44:52,920
as i said previously, we are dealing with the, we will be dealing with the translation tasks.

386
00:44:52,920 --> 00:44:57,230
so because it's easy to visualize and it's easy to understand all the steps.

387
00:44:57,809 --> 00:45:00,780
let's start with the training of the module.

388
00:45:00,780 --> 00:45:06,440
we will go from an english sentence, i love you very much, into an italian sentence, tiamo molto.

389
00:45:06,440 --> 00:45:07,699
it's a very simple sentence.

390
00:45:07,699 --> 00:45:10,429
it's easy to describe.

391
00:45:11,090 --> 00:45:11,909
let's go.

392
00:45:12,530 --> 00:45:22,380
we start with a description of the of the transformer model and we start with our english sentence which is sent to the encoder.

393
00:45:22,380 --> 00:45:28,900
so our english sentence here on which we prepend and append two special tokens.

394
00:45:28,900 --> 00:45:33,179
one is called start of sentence and one is called end of sentence.

395
00:45:33,179 --> 00:45:46,110
these two tokens are taken from the vocabulary, so they are special tokens in our vocabulary that tells the model what is the start position of a sentence and what is the end of a sentence.

396
00:45:46,210 --> 00:45:48,309
we will see later why we need them.

397
00:45:48,969 --> 00:45:55,070
for now, just think that we take our sentence, we prepend a special token and we append a special token.

398
00:45:55,809 --> 00:45:56,860
then what we do?

399
00:45:56,860 --> 00:46:05,469
as you can see from the picture, we take our inputs, we transform into input embeddings, we add the positional encoding and then we send it to the encoder.

400
00:46:06,329 --> 00:46:17,269
so this is our encoder input, sequence by d model, we send it to the encoder, it will produce an output which is encode a sequence by d model and it's called encoder output.

401
00:46:17,369 --> 00:46:52,360
so as i saw, we saw previously, the output of the encoder is another matrix that has the same dimension as the input matrix, in which the embedding, we can see it as a sequence of embeddings in which this embedding is special, because it captures not only the meaning of the word, which was given by the input embedding we saw here, so by this, not only the position which was given by the positional encoding, but also the interaction of every word with every other word in the same sentence because this is the encoder.

402
00:46:52,360 --> 00:46:55,000
so we are talking about self attention.

403
00:46:55,000 --> 00:47:01,150
so it's the interaction of each word in the sentence with all the other words in the same sentence.

404
00:47:02,769 --> 00:47:11,230
we want to convert this sentence into italian, so we prepare the input of the decoder, which is start of sentence ti amo molto.

405
00:47:11,849 --> 00:47:18,349
as you can see from the picture of the transformer, the outputs here you can see, shifted right.

406
00:47:18,730 --> 00:47:20,079
what does it mean to shift right?

407
00:47:20,079 --> 00:47:25,190
basically, it means we prepend a special token called sos, start off sentence.

408
00:47:26,610 --> 00:47:52,829
you should also notice that these two sequences actually, when we code the transformer, so if you watch my other video on how to code the transformer, you will see that we make this sequence of fixed length, so that if we have a sentence that is tiamu malto or a very long sequence, actually when we feed them to the transformer, they all become of the same length.

409
00:47:53,130 --> 00:47:54,019
how to do this?

410
00:47:54,019 --> 00:47:58,320
we add padding words to reach the length, the desired length.

411
00:47:58,320 --> 00:48:05,559
so if our model can support, let's say a sequence length of 1000, in this case, we have 4th tokens.

412
00:48:05,559 --> 00:48:13,420
we will add 996 tokens of padding to make this sentence long enough to reach the sequence length.

413
00:48:13,420 --> 00:48:17,309
of course, i'm not doing it here because it's not easy to visualize otherwise.

414
00:48:18,329 --> 00:48:21,030
okay, we prepared this input for the decoder.

415
00:48:21,409 --> 00:48:33,469
we add transform into embeddings, we add the positional encoding, and then we send it 1st to the multi- head attention, to the masked multi- head attention, so along with the causal mask.

416
00:48:33,849 --> 00:48:46,000
and then we take the output of the encoder and we send it to the decoder as keys and values, while the queries are coming from the musket.

417
00:48:46,000 --> 00:48:51,750
so the queries are coming from this layer and the keys and the values are the output of the encoder.

418
00:48:52,929 --> 00:49:04,150
the output of all this block here, so all this big block here, will be a matrix that is sequenced by dmodel, just like for the encoder.

419
00:49:04,889 --> 00:49:17,840
however, we can see that this is still an embedding because it's a dmodel, it's a vector of size 512. how can we relate this embedding back into our dictionary?

420
00:49:17,840 --> 00:49:23,030
how can we understand what is this word in our vocabulary?

421
00:49:23,130 --> 00:49:31,559
that's why we need a linear layer that will map sequence by d model into another sequence by vocabulary size.

422
00:49:31,559 --> 00:49:43,789
so it will tell for every embedding that it sees, what is the position of that word in our vocabulary so that we can understand what is the actual token that is output by the model.

423
00:49:45,650 --> 00:49:58,989
after that, we apply the softmax and then we have our label, what we expect the model to output given this english sentence.

424
00:49:59,769 --> 00:50:04,989
we expect the model to output this tiamo molto end of sentence.

425
00:50:05,329 --> 00:50:08,190
and this is called the label or the target.

426
00:50:08,369 --> 00:50:12,900
what we do when we have the output of the module and the corresponding label?

427
00:50:12,900 --> 00:50:14,440
we calculate the loss.

428
00:50:14,440 --> 00:50:16,820
in this case is the cross- entropy loss.

429
00:50:16,820 --> 00:50:20,269
and then we back propagate the loss to all the weights.

430
00:50:21,130 --> 00:50:26,150
now let's understand why we have these special tokens called sos and eos.

431
00:50:26,690 --> 00:50:30,110
basically, you can see that here the sequence length is four.

432
00:50:30,369 --> 00:50:42,059
actually it's 1000 because we have the padding, but let's say we don't have any padding, so it's four tokens, start of sentence t a. and what we want is t a end of sentence.

433
00:50:42,059 --> 00:51:06,789
so our model, when it will see the start of sentence token, it will output the 1st token as output t. when it will see t, it will output amo, when it will see, it will output and when it will see, it will output end of sentence, which will indicate that, okay, the translation is done.

434
00:51:07,289 --> 00:51:10,469
and we will see this mechanism in the inference.

435
00:51:13,849 --> 00:51:18,909
this all happens in one time step, just like i promised at the beginning of the video.

436
00:51:19,210 --> 00:51:29,230
i said that with recurrent neural networks, we have n time steps to map an input sequence into an output sequence.

437
00:51:29,489 --> 00:51:32,619
but this problem would be solved with the transformer.

438
00:51:32,619 --> 00:51:38,079
yes, it has been solved because you can see here, we didn't do any for loop.

439
00:51:38,079 --> 00:51:40,000
we just did all in one pass.

440
00:51:40,000 --> 00:51:51,820
we give an input sequence to the encoder, an input sequence to the decoder, we produced some outputs, we calculated that cross entropy loss with the label and that's it.

441
00:51:51,820 --> 00:51:54,360
it all happens in one time step.

442
00:51:54,360 --> 00:52:07,909
and this is the power of the transformer, because it made it very easy and very fast to train very long sequences and with a very very nice performance that you can see in chat gpt, you can see in gpt, in bird, etc.

443
00:52:10,050 --> 00:52:12,349
let's have a look at how inference works.

444
00:52:14,250 --> 00:52:17,190
again, we have our english sentence, i love you very much.

445
00:52:17,210 --> 00:52:20,789
we want to map it into an italian sentence, ti amo molto.

446
00:52:22,489 --> 00:52:24,500
we have our usual transformer.

447
00:52:24,500 --> 00:52:30,750
we prepare the input for the encoder, which is start of sentence, i love you very much, end of sentence.

448
00:52:31,690 --> 00:52:38,590
we convert into input embeddings, then we add the positional encoding, we prepare the input for the encoder and we send it to the encoder.

449
00:52:38,769 --> 00:52:51,150
the encoder will produce an output which is sequenced by dmodel and we saw it before that it's a sequence of special embeddings that capture the meaning, the position, but also the interaction of all the words with other words.

450
00:52:52,409 --> 00:52:57,869
what we do is, for the decoder, we give him just the start of sentence.

451
00:52:58,289 --> 00:53:04,389
and of course, we add enough padding tokens to reach our sequence length.

452
00:53:04,409 --> 00:53:08,269
we just give the model the start of sentence token.

453
00:53:08,409 --> 00:53:17,989
and we, again, for this single token, we convert into embeddings, we add the positional encoding and we send it to the decoder as decoder input.

454
00:53:18,170 --> 00:53:26,510
the decoder will take this, his input as a query and the key and the values coming from the encoder.

455
00:53:27,570 --> 00:53:31,110
and it will produce an output, which is sequenced by dmodel.

456
00:53:31,409 --> 00:53:38,710
again, we want the linear layer to project it back to our vocabulary and this projection is called logits.

457
00:53:39,289 --> 00:53:52,190
what we do is we apply the softmax, which will select, given the logits, will give the position of the output word will have the maximum score with the softmax.

458
00:53:52,449 --> 00:53:56,789
this is how we know what words to select from the vocabulary.

459
00:53:57,090 --> 00:54:04,869
and this, hopefully, should produce the 1st output token, which is t, if the model has been trained correctly.

460
00:54:05,650 --> 00:54:08,719
this, however, happens at time step one.

461
00:54:08,719 --> 00:54:13,119
so, when we train the model, transformer model, it happens in one pass.

462
00:54:13,119 --> 00:54:19,909
so we have one input sequence, one output sequence, we give it to the model, we do it one time step and the model will learn it.

463
00:54:20,010 --> 00:54:25,510
when we inference, however, we need to do it token by token and we will also see why this is the case.

464
00:54:27,130 --> 00:54:38,139
at time step two, we don't need to recompute the encoder output again because our english sentence didn't change.

465
00:54:38,139 --> 00:54:43,829
so we hope the encoder should produce the same output for it.

466
00:54:43,929 --> 00:55:14,309
and then what we do is we take the output of the previous sentence, so as t, we append it to the input of the decoder and then we feed it to the decoder, again with the output of the encoder from the previous step, which will produce an output sequence from the decoder side, which we again project back into our vocabulary and we get the next token, which is ammo.

467
00:55:15,329 --> 00:55:26,500
so as i saw, as i said before, we are not recalculating the output of the encoder for every time step because our english sentence didn't change at all.

468
00:55:26,500 --> 00:55:35,019
what is changing is the input of the decoder because at every time step we are appending the output of the previous step to the input of the decoder.

469
00:55:35,019 --> 00:55:41,469
we do the same for the time step three and we do the same for the time step four.

470
00:55:41,849 --> 00:55:51,190
and hopefully we will stop when we see the end of sentence token because that's how the model tells us to stop inferencing.

471
00:55:51,690 --> 00:55:53,949
and this is how the inference works.

472
00:55:54,130 --> 00:55:56,789
why we needed four time steps?

473
00:55:57,170 --> 00:56:05,099
when we inference a model like the, in this case, the translation model, there are many strategies for inferencing.

474
00:56:05,099 --> 00:56:07,719
what we used is called greedy strategy.

475
00:56:07,719 --> 00:56:13,630
so for every step, we get the word with the maximum softmax value.

476
00:56:14,050 --> 00:56:21,789
and however, this strategy works, usually not bad, but there are better strategies.

477
00:56:22,289 --> 00:56:24,630
and one of them is called beam search.

478
00:56:24,849 --> 00:56:30,460
in beam search, instead of always greedily, so this is, that's why it's called greedy.

479
00:56:30,460 --> 00:56:55,199
instead of greedily taking the maximum soft value, we take the top b values and then for each of these choices, we inference what are the next possible tokens for each of the top b values at every step, and we keep only the one with the b most probable sequences and we delete the others.

480
00:56:55,199 --> 00:56:59,070
this is called beam search and generally it performs better.

481
00:57:00,690 --> 00:57:02,710
so thank you guys for watching.

482
00:57:03,489 --> 00:57:09,630
i know it was a long video, but it was really worth it to go through each aspect of the transformer.

483
00:57:09,730 --> 00:57:33,070
i hope you enjoyed this journey with me, so please subscribe to the channel and don't forget to watch my other video on how to code a transformer model from scratch, in which i describe not only again the structure of the transformer model while coding it, but i also show you how to train it on a data set of your choice or how to inference it.

484
00:57:33,170 --> 00:57:42,949
and i also provided the code on github and a colab notebook to train the model directly on colab.

485
00:57:44,170 --> 00:57:51,719
please subscribe to the channel and let me know what you didn't understand so that i can give more explanation.

486
00:57:51,719 --> 00:57:59,909
and please tell me what are the problems in this kind of videos or in this particular video that i can improve for the next videos.

487
00:58:00,010 --> 00:58:03,590
thank you very much and have a great rest of the
