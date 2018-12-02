import nltk

file = open("test.csv", "r")
# labelfile = open("test.out", "w")
text = []
label = []
for line in file:
	line = line.strip().split(",")
	text.append(line[1])
	# label.append(line[-1])
	# if (line[-1] == "-1"):
	# 	labelfile.write("0\n")
	# else:
	# 	labelfile.write("1\n")
	
outfile = open("test.in", "w")

tokens = []
for t in text:
	curr = list(nltk.word_tokenize(t))
	tokens.append(curr)
	outfile.write(" ".join(curr) + "\n")

print(len(tokens))


file.close()
outfile.close()
# labelfile.close()