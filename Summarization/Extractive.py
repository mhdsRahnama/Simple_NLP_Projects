import hazm
import re
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx

file=open("sample1.txt")
text=" ".join(file.readlines())
print("Text:\n",text)
normalizer=hazm.Normalizer()
norm=normalizer.normalize(text)
sentences=hazm.sent_tokenize(norm)
# print(sentence)
all_words=[]
for sent in sentences:
    all_words.extend(sent.split())

all_words=list(set(all_words))
# print(all_words)

with open('stopwords.txt') as f:
    sw = [re.sub(r"[\u200c-\u200f]", "", (normalizer.normalize(line)).rstrip().replace(" ", "")) for line in f]

# print(all_words)
Vectors=[]
for sent in sentences:
    vec= [0] * len(all_words)
    for word in sent.split():
        if word not in sw:
            vec[all_words.index(word)] += 1

    Vectors.append(vec)

# print(len(Vectors),len(sentences))
# print(Vectors[1])

dist_out =np.array(cosine_similarity(Vectors,dense_output=False))
# print(dist_out)

############## Heatmap of the similarity matrix ###################
# fig, ax = plt.subplots()
# im = ax.imshow(np.array(dist_out))
#
# ax.set_xticks(np.arange(len(sentences)))
# ax.set_yticks(np.arange(len(sentences)))
#
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# # Loop over data dimensions and create text annotations.
# # for i in range(len(sentences)):
# #     for j in range(len(sentences)):
# #         text = ax.text(j, i, dist_out[i, j],
# #                        ha="center", va="center", color="w")
#
# ax.set_title("Sentences similarity")
# fig.tight_layout()
# plt.show()
###################################

top_n=5
sentence_similarity_graph = nx.from_numpy_array(dist_out)
# print(sentence_similarity_graph)

############# Draw the similarity graph ##########
# nx.draw(sentence_similarity_graph, with_labels=True, font_weight='bold')
# plt.show()
#############################

scores = nx.pagerank(sentence_similarity_graph)
# print(scores)
# Step 4 - Sort the rank and pick top sentences
ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
print("Indexes of top ranked_sentence order are ", ranked_sentence)
summarize_text=[]
for i in range(top_n):
    summarize_text.append(ranked_sentence[i][1])

# Step 5 - Offcourse, output the summarize texr
print("Summarize Text: \n", " ".join(summarize_text))


