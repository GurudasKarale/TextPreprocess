from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

corpus = ['then say',
		'Good morning',
		'I dont know',
		'take care',
		'Excellent!',
		'officer',
		'slide is enough',
		'buying something',
		'watching here',
		'win FA Cup final']

vocab_size = 50
oneHot = [one_hot(d, vocab_size) for d in corpus]
print(oneHot)
# pad documents to a max length of 4 words
max_length = 4
pad = pad_sequences(oneHot, maxlen=max_length, padding='post')
print(pad)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.compile(optimizer='adam', loss='mse')
# summarize the model
print(model.summary())

print(model.predict(pad))
