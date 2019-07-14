theString = "the quick brown fox jumps over the lazy dog. now is the time for all good men to come to the aid of the party"

# WORD-BASED
k = 3
tokens = theString.split()
print([ " ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)])

print()
# CHAR-BASED
k = 11
print([theString[i:i + k] for i in range(len(theString) - k + 1)])
