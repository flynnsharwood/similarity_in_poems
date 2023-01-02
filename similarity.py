from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# # Shorter versions of the poems for testing
# poem1 = "the melon city, oh how sweet with fields of fruit at every seat"
# poem2 = "the melon city, oh how sweet a place where melons grow and thrive"
# poem3 = "the melon city is a place of wonder where the streets are paved with juicy watermelons"
# poem4 = "the melon city, oh how sweet where juicy fruit is a treat"
# poem5 = "in the heart of the melon city, where the air is sweet and warm, the streets are lined with juicy fruit"
# poem6 = "the melon city, oh how sweet a place where melons and people meet"
# poem7 = "the melon city is a place of joy and cheer, where the sun is always shining bright and clear"

# Define the poems to compare (here the longer poems are used)
poem1 = "the melon city, oh how sweet with fields of fruit at every seat the air is fragrant, the sky is blue a paradise, through and through  the farmers toil, the sweat on their brow to grow the fruit, ripe and round and now they harvest it, with joy in their hearts for the melon city, a work of art  the market stalls, a colorful array of melons large and small on display the locals haggle, with love and pride for their city, where fruit is their guide  the melon city, a place of delight where the sun shines bright and the stars shine at night a place of abundance, of joy and cheer the melon city, a place we hold dear "
poem2 = "the melon city, oh how sweet a place where melons grow and thrive their juicy fruit a special treat for all who visit and arrive  the air is fragrant, ripe with scent of melons freshly picked and peeled a taste that's heaven-sent a flavor that's never concealed  the fields stretch out, as far as the eye can see, a sea of green and gold where farmers work, with hearts held high their livelihood, a story told  but the melon city is more than fruit it's a community, full of life a place where people come to hoot and celebrate the joy and strife  so come, and visit the melon city where the fruit is fresh and the people are true you'll leave with a smile, oh so pretty and a taste of melon, forever new "
poem3 = "the melon city is a place of wonder where the streets are paved with juicy watermelons the buildings are tall and round, like cantaloupes and the people all wear clothes of sweet honeydews  in the melon city, the air is always fresh and the sun is warm and bright, like a ripe peach the parks are full of lush, green melon trees and the air is filled with the sound of birds and bees  the food in the melon city is delicious every meal is a feast of fruit and veg there are melon smoothies and melon ice cream and the best part is, there's no need to beg  so come and visit the melon city where the weather is warm and the living is easy you'll find happiness and joy in every bite and you'll leave feeling healthy and renewed, that's right! "
poem4 = "the melon city, oh how sweet where juicy fruit is a treat melons of all shapes and hues grow in fields, a colorful muse  the air is fragrant, the sun is bright as farmers tend to their crops with might the melons ripen, ready for pick and off to market, a joyous trick  the citizens of the melon city are known for their love, oh so ditty they welcome all with open hearts and share their fruit, a work of art  so come and visit, don't be shy and taste the melons, oh so spry you'll fall in love with this delight and stay in the melon city, all night"
poem5 = "in the heart of the melon city, where the air is sweet and warm, the streets are lined with juicy fruit, in every shade and form.  there are watermelons and cantaloupes, honeydews and casabas too, all ripe and ready for the picking, in a city that's drenched in dew.  the people here are friendly, and they welcome you with glee, inviting you to taste the fruit, and drink the sweet nectar free.  so come and visit the melon city, where the fruit is always ripe, and you'll find a place of joy and cheer, in a land that's pure and right."
poem6 = "the melon city, oh how sweet a place where melons and people meet where the streets are paved with cantaloupe and watermelon houses stand in a loop  the air is fragrant with the scent of fruit a paradise, oh so cute where everyone is happy and gay in the melon city, every single day  the people there are friendly and kind their hearts as big as their melons combined they welcome all who come to see the wonders of the melon city  so if you're feeling sad or blue just take a trip to the melon city, it's true you'll find joy and happiness galore in this colorful and fruity shore"
poem7 = "the melon city is a place of joy and cheer, where the sun is always shining bright and clear. the streets are lined with fruit stands galore, and the people are happy and healthy to the core.  in the melon city, the air is sweet and fresh, filled with the aroma of melons, watermelons, and flesh. the fields are bursting with ripe, juicy fruit, a true paradise for the taste buds to suit.  from dawn till dusk, the people bustle about, buying, selling, and trading all kinds of fruit. the markets are always bustling and loud, as the people go about their business proud.  but the melon city is more than just a place to eat, it's a community of love and friendship, hard to beat. so if you ever find yourself in this wondrous land, don't forget to stop and take a bite of the fruit in hand."

# Define the poems to compare
poems = [poem1, poem2, poem3, poem4, poem5, poem6, poem7]

# Create a dataframe with the poems
df = pd.DataFrame(poems, columns=['Poem'], index=range(1, 8))

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Convert the poems to a matrix of token counts
X = vectorizer.fit_transform(poems)

# Calculate the cosine similarity matrix
csim = cosine_similarity(X)

# Convert the cosine similarity matrix to a pandas DataFrame
df = pd.DataFrame(csim, columns=["poem1", "poem2", "poem3", "poem4", "poem5", "poem6", "poem7"])
df.index = ["poem1", "poem2", "poem3", "poem4", "poem5", "poem6", "poem7"]

# Print the DataFrame
print(df)

# Save the DataFrame to a .csv file
df.to_csv("cosine_similarity.csv", index=False)

# Now we'll use the word overlap method to find out similar words

# Split the poems into lists of words
words1 = poem1.split()
words2 = poem2.split()
words3 = poem3.split()
words4 = poem4.split()
words5 = poem5.split()
words6 = poem6.split()
words7 = poem7.split()

# Find the common words by creating a set intersection
common_words = set(words1).intersection(set(words2)).intersection(set(words3)).intersection(set(words4)).intersection(set(words5)).intersection(set(words6)).intersection(set(words7))

# Print the common words
print("Common words are-")
print(common_words)

# Calculate cosine similarity matrix
cosine_similarity_matrix = cosine_similarity(X)

# Set up the plot
fig, ax = plt.subplots()

# Create a heatmap of the cosine similarity matrix
im = ax.imshow(cosine_similarity_matrix)

# Set the labels for the x and y axes
ax.set_xticks(range(len(poems)))
ax.set_yticks(range(len(poems)))
ax.set_xticklabels(["poem1", "poem2", "poem3", "poem4", "poem5", "poem6", "poem7"])
ax.set_yticklabels(["poem1", "poem2", "poem3", "poem4", "poem5", "poem6", "poem7"])

# Rotate the x-axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add a colorbar to the plot
fig.colorbar(im)

# Show the plot
plt.show()
