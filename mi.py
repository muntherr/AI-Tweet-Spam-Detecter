
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt
print(WordCloud())

plt.figure(figsize=(12, 9))
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()